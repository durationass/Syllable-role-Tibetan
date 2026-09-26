# Transcript-Guided Syllable-Role Regularization for Generative Tibetan Speech Enhancement

This repository provides code for Tibetan speech enhancement with a transcript-guided syllable-role regularization strategy. The system combines a SepFormer front end with GECO/Fast-GeCo generative correction.


## Environment

The public environment targets Linux with an NVIDIA GPU, Python 3.10, and CUDA 12.4.

```bash
conda env create -f environment-public.yml
conda activate zangyu
pip install -e ./speechbrain
pip install -e ./score_models
```

W&B logging is disabled by default. Use `--wandb` only after configuring W&B credentials. Use `--nolog` to disable local logging as well.

## Models and Data

The model and dataset repositories are public. Local paths match the commands below.

| Resource | Local path / usage | Contents | Download |
| --- | --- | --- | --- |
| SepFormer | Download the files as described below. | SepFormer configuration, encoder/masknet/decoder weights, and fine-tuned `best_pesq.pt`. | https://huggingface.co/Myrail/trained_sepformer |
| Fast-GeCo | Select a checkpoint with `--ckpt`. | Fast-GeCo baseline and syllable-role Fast-GeCo results. | https://huggingface.co/Myrail/zang_ckpt |
| `zang_data` | `zang_data/` | Provides the training, validation, and test splits for the full training pipeline. | https://huggingface.co/datasets/Myrail/zang_data |


For SepFormer, download `hyperparams.yaml`, `encoder.ckpt`, `masknet.ckpt`, and `decoder.ckpt` from `Myrail/trained_sepformer` into `pretrained_models/sepformer-whamr/`; installing the Python packages does not download these files. To use the provided fine-tuned model, also download `best_pesq.pt` into `ckpt/sepformer_ft/` and skip Step 1 (Fine-tune SepFormer).

For downloaded Fast-GeCo checkpoints, replace `ckpt/fastgeco/last.ckpt` in the evaluation example with the selected checkpoint's local path.

## Data Preparation

The input data is organized into train, validation, and test splits:

```text
zang_data/
├── train/
│   ├── metadata.csv
│   ├── <id>_mix.wav
│   └── <id>_source1.wav
├── val/
└── test/
```

Audio files should be 8 kHz mono waveforms. After SepFormer processing, each split also contains:

```text
<id>_source1hatP.wav
```

Here, `*_mix.wav` is the mixture, `*_source1.wav` is the clean target, and `*_source1hatP.wav` is the initial estimate used by GECO/Fast-GeCo. `label.txt` contains the transcript labels used to build syllable-role masks. The mask generator writes one `*_role_mask.npz` file per training example.

## Running the Pipeline

Run the stages in this order.

### 1. Fine-tune SepFormer

```bash
python function_script/sepformer/finetune_sepformer.py \
  --train_dir zang_data/train \
  --valid_dir zang_data/val \
  --source pretrained_models/sepformer-whamr \
  --savedir pretrained_models/sepformer-whamr \
  --output_dir ckpt/sepformer_ft \
  --device cuda:0 \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-5 \
  --segment_seconds 4 \
  --num_eval_files 50 \
  --num_workers 0 \
  --nolog
```

### 2. Generate SepFormer Estimates

```bash
python function_script/sepformer/generate_sepformer_ft.py \
  --input_root zang_data \
  --output_root zang_data_sepformer \
  --source pretrained_models/sepformer-whamr \
  --savedir pretrained_models/sepformer-whamr \
  --checkpoint ckpt/sepformer_ft/best_pesq.pt \
  --device cuda:0 \
  --splits train,val,test
```

### 3. Generate Syllable-Role Masks

First create `label.txt` in the format expected by the mask generator. Then run:

```bash
python function_script/syllable/build_mask/tibetan_syllable_mask.py \
  --split_dir zang_data/train \
  --label_file zang_data/label.txt \
  --metadata_file zang_data/train/metadata.csv \
  --output_dir zang_data_sepformer/train/role_masks \
  --num_visualize 0
```

### 4. Train GECO

```bash
python train-geco.py \
  --backbone ncsnpp \
  --lr 5e-5 \
  --warmup_ratio 0.1 \
  --train_dir zang_data_sepformer/train \
  --val_dir zang_data_sepformer/val \
  --test_dir zang_data_sepformer/test \
  --checkpoint_dir ckpt/geco \
  --gpus 1 \
  --batch_size 4 \
  --max_epochs 30
```

### 5. Train Role-Regularized Fast-GeCo

```bash
python train-fastgeco.py \
  --pre_ckpt ckpt/geco/last.ckpt \
  --train_dir zang_data_sepformer/train \
  --val_dir zang_data_sepformer/val \
  --test_dir zang_data_sepformer/test \
  --checkpoint_dir ckpt/fastgeco \
  --gpus 1 \
  --batch_size 4 \
  --max_epochs 30 \
  --lr 5e-5 \
  --loss_type default \
  --role_loss_weight 0.05 \
  --role_loss_warmup_steps 0 \
  --role_mask_dir zang_data_sepformer/train/role_masks \
  --role_onset_weight 3.0 \
  --role_nucleus_weight 2.0 \
  --role_transition_weight 0.0 \
  --role_highband_start_hz 1800
```

`--role_mask_dir` is required when `--role_loss_weight` is greater than zero. Set `--role_loss_weight 0` to train the ordinary Fast-GeCo objective.

### 6. Evaluate Fast-GeCo

```bash
python eval-fastgeco.py \
  --test_dir zang_data_sepformer/test \
  --ckpt ckpt/fastgeco/last.ckpt \
  --destination_folder fastgeco_eval \
  --max_index 0
```


## Outputs and Logging

- Training checkpoints are written to the directory passed with `--checkpoint_dir`.
- Evaluation outputs are written to `Libri2mix/<destination_folder>/` unless `--output_dir` is provided.
- Evaluation writes `_results.csv`, `_avg_results.txt`, `_settings.txt`, and enhanced/reference audio under `files/`.
- TensorBoard logs are stored under `logs/`. View them with `tensorboard --logdir logs`.
- W&B is opt-in through `--wandb` and is never given a key by the code.

## Attribution and Data

This project is based on [Fast-GeCo](https://github.com/WangHelin1997/Fast-GeCo),
licensed under the MIT License. It modifies the original GECO/Fast-GeCo training
and evaluation code for Tibetan speech enhancement and adds transcript-guided
syllable-role mask generation and role-regularized training.

Please cite the Fast-GeCo paper when using this code:

```bibtex
@inproceedings{wang24i_interspeech,
  title     = {Noise-robust Speech Separation with Fast Generative Correction},
  author    = {Helin Wang and Jes{\'u}s Villalba and Laureano Moro-Velazquez and
               Jiarui Hai and Thomas Thebaud and Najim Dehak},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2165--2169},
  doi       = {10.21437/Interspeech-2024-327},
}
```

## License

Original project-specific additions are released under the MIT License. See
[LICENSE](LICENSE) and the attribution notes above for third-party components.
