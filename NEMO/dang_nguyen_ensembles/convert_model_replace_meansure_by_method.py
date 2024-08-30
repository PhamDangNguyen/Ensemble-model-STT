from dataclasses import dataclass, field
from typing import List, Optional
import soundfile as sf
import logging
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
import torch
from omegaconf import DictConfig, MISSING
import tempfile
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from nemo.collections.asr.parts.utils.transcribe_utils import TextProcessingConfig
from nemo.utils import logging
from nemo.collections.asr.models import EncDecCTCModelBPE
import json
from tqdm.auto import tqdm
import os

# File path to the .nemo model
filepath = "/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/models_convert/asr_model.nemo"

# Load the ASR model and its config
asr_model = nemo_asr.models.EncDecCTCModelBPE
cfg = asr_model.restore_from(filepath, return_config=True)

# Replace 'measure_cfg' with 'method_cfg' in cfg.decoding.confidence_cfg
if 'measure_cfg' in cfg.decoding.confidence_cfg:
    cfg.decoding.confidence_cfg.method_cfg = cfg.decoding.confidence_cfg.measure_cfg
    del cfg.decoding.confidence_cfg.measure_cfg

# Restore the model with the modified config
Model = asr_model.restore_from(filepath, override_config_path=cfg)

# Save the modified model to a new file
Model.save_to("/home/pdnguyen/Ensemble_confidence_Nemo/confidence-ensembles-tutorial/NeMo/scripts/dang_nguyen_ensembles/Overwrite_model.nemo")