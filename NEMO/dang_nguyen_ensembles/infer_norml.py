from nemo.collections.asr.models.confidence_ensemble import ConfidenceEnsembleModel
from nemo.collections.asr.models.confidence_ensemble import compute_confidence
import torch
from typing import Dict, List, Optional
import numpy as np
import torch
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from omegaconf import DictConfig
class Infer_ASR(ConfidenceEnsembleModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def customer_transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        **kwargs,  # any other model specific parameters are passed directly
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
        # always requiring to return hypothesis
        # TODO: make sure to return text only if was False originally
        return_hypotheses = True
        for model_idx in range(self.num_models):
            model = getattr(self, f"model{model_idx}")
            transcriptions = model.transcribe(
                audio=paths2audio_files,
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                **kwargs,
            )
            print(transcriptions)
            print("--------------------------------------------------------------------------------------------------------------------")
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
        final_transcriptions = []
        for transcrption_idx in range(len(all_transcriptions[0])):
            final_transcriptions.append(all_transcriptions[model_indices[transcrption_idx]][transcrption_idx])
        return final_transcriptions
    
    
if __name__ == '__main__':
    model_nemo = "/home/pdnguyen/Ensemble_confidence_Nemo/confidence-ensembles-tutorial/NeMo/scripts/dang_nguyen_ensembles/1.nemo"
    model = Infer_ASR.restore_from(model_nemo, map_location=torch.device("cuda"))
    text_infer = model.customer_transcribe(["/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/infer_N/fubong/20240222164216_giọng nữ miền Bắc 1/chunk_13_normalized.wav","/mnt/driver/pdnguyen/data_record/telesale/T7/shipper.wav"],batch_size=2,return_hypotheses=True)
    print(text_infer)
 

    
