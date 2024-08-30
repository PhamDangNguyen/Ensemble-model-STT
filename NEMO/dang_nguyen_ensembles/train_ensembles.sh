python build_ensemble.py --config-path=. --config-name=ensemble_config.yaml \
    ensemble.0.model=/home/pdnguyen/Ensemble_confidence_Nemo/Ensemble-model-STT/NEMO/dang_nguyen_ensembles/models/Conformer/English.nemo \
    ensemble.0.training_manifest=/home/pdnguyen/Ensemble_confidence_Nemo/Ensemble-model-STT/NEMO/dang_nguyen_ensembles/English_metadata.json \
    ensemble.1.model=/home/pdnguyen/Ensemble_confidence_Nemo/Ensemble-model-STT/NEMO/dang_nguyen_ensembles/models/Conformer/Fubong.nemo \
    ensemble.1.training_manifest=/home/pdnguyen/Ensemble_confidence_Nemo/Ensemble-model-STT/NEMO/dang_nguyen_ensembles/fubon_meta.json \
    # ensemble.2.model=/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/backup/7_10_2024/nemo_experiments/FastConformer-CTC-BPE/checkpoints/175.nemo \
    # ensemble.2.training_manifest=/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/metadata_train/test_0.json \
    output_path=confidence_build.nemo
