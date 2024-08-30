from nemo.collections.asr.models.confidence_ensemble import ConfidenceEnsembleModel
from nemo.collections.asr.models.confidence_ensemble import compute_confidence
from typing import List, Optional, Union
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
# from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from omegaconf import DictConfig, MISSING
from dataclasses import dataclass, field
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from nemo.collections.asr.parts.utils.transcribe_utils import TextProcessingConfig
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.utils import logging
import nemo.collections.asr as nemo_asr
import soundfile as sf
from segments import Segment
import tempfile
import json
from tqdm import tqdm
import os

@dataclass
class EvalBeamSearchNGramConfig:
    """
    Evaluate an ASR model with beam search decoding and n-gram KenLM language model.
    """
    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    nemo_model_file: str = MISSING

    # File paths
    input_manifest: str = MISSING  # The manifest file of the evaluation set
    kenlm_model_file: Optional[str] = None  # The path of the KenLM binary model file
    preds_output_folder: Optional[str] = None  # The optional folder where the predictions are stored
    probs_cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    acoustic_batch_size: int = 16  # The batch size to calculate log probabilities
    beam_batch_size: int = 128  # The batch size to be used for beam search decoding
    # device: str = "cpu"  # The device to load the model onto to calculate log probabilities
    device: str = "cuda"
    use_amp: bool = False  # Whether to use AMP if available to calculate log probabilities

    # Beam Search hyperparameters

    # The decoding scheme to be used for evaluation.
    # Can be one of ["greedy", "beamsearch", "beamsearch_ngram"]
    decoding_mode: str = "greedy"

    beam_width: List[int] = field(default_factory=lambda: [32])  # The width or list of the widths for the beam search decoding
    beam_alpha: List[float] = field(default_factory=lambda: [0.7])  # The alpha parameter or list of the alphas for the beam search decoding
    beam_beta: List[float] = field(default_factory=lambda: [1.0])  # The beta parameter or list of the betas for the beam search decoding

    # Can be one of ["flashlight", "pyctcdecode", "beam"]
    decoding_strategy: str = "flashlight"
    decoding: ctc_beam_decoding.BeamCTCInferConfig = field(default_factory=lambda: ctc_beam_decoding.BeamCTCInferConfig(beam_size=128))
    
    text_processing: Optional[TextProcessingConfig] = field(default_factory=lambda: TextProcessingConfig(
        punctuation_marks = "",
        separate_punctuation = False,
        do_lowercase = False,
        rm_punctuation = False,
    ))

class FastConformerASR(EncDecCTCModelBPE):

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
    
    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        If modify this function, please remember update transcribe_partial_audio() in
        nemo/collections/asr/parts/utils/trancribe_utils.py

        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)
        # We will store transcriptions here
        hypotheses = []
        all_hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    'channel_selector': channel_selector,
                }

                if augmentor:
                    config['augmentor'] = augmentor

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=not verbose):
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    probs = []
                    for idx in range(logits.shape[0]):
                        lg = logits[idx][: logits_len[idx]]
                        probs.append(lg.cpu().numpy())
                    else:
                        current_hypotheses, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
                            logits, decoder_lengths=logits_len, return_hypotheses=return_hypotheses,
                        )
                        logits = logits.cpu()
                
                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                                if current_hypotheses[idx].alignments is None:
                                    current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence
                        if all_hyp is None:
                            hypotheses += current_hypotheses
                        else:
                            hypotheses += all_hyp
                    
                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)
        return hypotheses,probs

class FastConformerWithLM:

    def __init__(
        self, cfg: EvalBeamSearchNGramConfig,
        beam_width=32, beam_alpha=1.0, beam_beta=1.0,
        return_best_hypothesis=False,
    ):  
        model = cfg.nemo_model_file
        cfg.decoding.beam_size = beam_width
        cfg.decoding.beam_alpha = beam_alpha
        cfg.decoding.beam_beta = beam_beta
        cfg.decoding.return_best_hypothesis = return_best_hypothesis
        cfg.decoding.kenlm_path = cfg.kenlm_model_file
        cfg.decoding.preserve_word_confidence = True
        self.cfg = cfg
        self.asr_model = model
    
    def _disable_logging(self):
        logging.set_verbosity(logging.CRITICAL)
    
    def _enable_logging(self):
        logging.set_verbosity(logging.INFO)
        
    def _enable_preserve_alignments(self):
        self.cfg.decoding_mode = "greedy"
        self.cfg.decoding_strategy = "greedy"
        decoding_cfg = self.asr_model.cfg.decoding
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.strategy = self.cfg.decoding_strategy
        decoding_cfg.beam = self.cfg.decoding
        decoding_cfg.confidence_cfg.preserve_word_confidence = True
        decoding_cfg.confidence_cfg.preserve_token_confidence = True
        decoding_cfg.confidence_cfg.preserve_frame_confidence = True
        decoding_cfg.confidence_cfg.method_cfg.name = "max_prob"
        self._disable_logging()
        self.asr_model.change_decoding_strategy(decoding_cfg)
        self._enable_logging()

    def _enable_beamsearch(self):
        self.cfg.decoding_mode = "beamsearch_ngram"
        self.cfg.decoding_strategy = "beam"
        decoding_cfg = self.asr_model.cfg.decoding
        decoding_cfg.preserve_alignments = False
        decoding_cfg.compute_timestamps = False
        decoding_cfg.confidence_cfg.preserve_word_confidence = False
        decoding_cfg.confidence_cfg.preserve_token_confidence = False
        decoding_cfg.confidence_cfg.preserve_frame_confidence = False
        decoding_cfg.strategy = self.cfg.decoding_strategy
        decoding_cfg.beam = self.cfg.decoding
        self._disable_logging()
        # print(decoding_cfg)
        self.asr_model.change_decoding_strategy(decoding_cfg)
        self._enable_logging()    
      
    def transcribe_ensemble(self, audio_file: str,batch_size=3,return_hypotheses=None):
        self._enable_preserve_alignments()
        outputs= self.asr_model.transcribe(audio_file,return_hypotheses=return_hypotheses,batch_size=batch_size)
        return outputs
    
    def get_results_beamSearch(self,output_fromModel):
        outputs_text=[]
        for index,output in enumerate(output_fromModel):
            timesteps, _, probs_batch = output_fromModel[index].timestep,output_fromModel[index].timestep,[output_fromModel[index].y_sequence]
            # print(len(timesteps["word"]))
            # print(len(output.word_confidence))
            raw_segments = [Segment(w["word"], w["start_offset"], w["end_offset"],output.word_confidence[i]) for i,w in enumerate(timesteps["word"])]
            raw_segments = [seg for seg in raw_segments if seg.word != ""]
            self._enable_beamsearch()
            probs_lens = torch.tensor([prob.shape[0] for prob in probs_batch])
            with torch.no_grad():
                packed_batch = torch.zeros(len(probs_batch), max(probs_lens), probs_batch[0].shape[-1], device='cpu')

                for prob_index in range(len(probs_batch)):
                    packed_batch[prob_index, : probs_lens[prob_index], :] = torch.tensor(
                        probs_batch[prob_index], device=packed_batch.device, dtype=packed_batch.dtype
                    )
                _, beams_batch = self.asr_model.decoding.ctc_decoder_predictions_tensor(
                    packed_batch, decoder_lengths=probs_lens, return_hypotheses=True,
                )
            kenlm_text = beams_batch[0][0].text
            outputs_text.append([raw_segments,kenlm_text])
        return outputs_text  
################################################################################################################################################################################

class Infer_ASR(ConfidenceEnsembleModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def customer_transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        Lm_path = None,
        return_hypotheses = False
    ) -> List[str]:
        """Confidence-ensemble transcribe method.

        Consists of the following steps:

            1. Run all models (TODO: in parallel)
            2. Compute confidence for each model
            3. Use logistic regression to pick the "most confident" model
            4. Return the output of that model
        """

        confidences = []
        all_transcriptions = []
        final_transcriptions = []
        # always requiring to return hypothesis
        # TODO: make sure to return text only if was False originally
        # print(self.num_models)

        model_results = {
            "model_0": [],
            "model_1": []
        }

        for model_idx in range(self.num_models):
            model = getattr(self, f"model{model_idx}")
            cfg = EvalBeamSearchNGramConfig(beam_alpha=0.7, beam_beta=1.0, beam_width=32)
            cfg.nemo_model_file = model
            cfg.kenlm_model_file = Lm_path[model_idx]
            fast_conformer = FastConformerWithLM(cfg=cfg, beam_alpha=0.5, beam_beta=2.0, beam_width=64) 
            transcriptions = fast_conformer.transcribe_ensemble(audio_file=paths2audio_files,batch_size=batch_size,return_hypotheses=return_hypotheses)
            beam_search_results = fast_conformer.get_results_beamSearch(transcriptions)

            for beam_search_result in beam_search_results:
                model_results[f"model_{model_idx}"].append(beam_search_result[1])

            if isinstance(transcriptions, tuple):  # transducers return a tuple
                transcriptions = transcriptions[0]
            model_confidences = []
            for transcription in transcriptions:
                model_confidences.append(compute_confidence(transcription, self.confidence_cfg))
            confidences.append(model_confidences)
            all_transcriptions.append(transcriptions)

        # transposing with zip(*list)
        features = np.array(list(zip(*confidences)))
        model_indices = self.model_selection_block.predict(features)
        for audio_index, model_index in enumerate(model_indices):
            final_transcriptions.append(model_results[f"model_{model_index}"][audio_index])

        return final_transcriptions
 
    
if __name__ == '__main__':
    model_nemo = "/home/pdnguyen/Ensemble_confidence_Nemo/Ensemble-model-STT/NEMO/dang_nguyen_ensembles/models/Ensemble/Ensemble_E_fubong.nemo"
    KenLM_path = ["/home/pdnguyen/Ensemble_confidence_Nemo/Ensemble-model-STT/NEMO/dang_nguyen_ensembles/models/kenLM/model_fubong_23_8_2024_English_4","/home/pdnguyen/fast_confomer_finetun/train_kenLM/scripts/asr_language_modeling/kenLM_output/model_fubong_23_8_2024_5"]
    
    model = Infer_ASR.restore_from(model_nemo, map_location=torch.device("cuda"))

    text_infer = model.customer_transcribe(
        paths2audio_files=["/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/infer_N/fubong/20240222164216_giọng nữ miền Bắc 1/chunk_13_normalized.wav",
                "/home/pdnguyen/Ensemble_confidence_Nemo/English_data/3.wav",
                "/mnt/driver/pdnguyen/studen_annoted/data_telesale_extract_10_dir/data_tele_1/notannoted/group_1/92.wav"
                ],
        Lm_path=KenLM_path,
        batch_size=2,
        return_hypotheses=True,
    )
    print(text_infer)

