# SET

Official PyTorch implementation of the Superpixel-Embedded Transformer (SET) published in *Medical Image Analysis 2025*. SET targets 2D medical image segmentation (ISIC skin-lesion benchmarks by default) and augments encoder–decoder models with pre-computed superpixel graphs that act as structural priors during training and inference.

## Repository Layout

- `datasets/`: dataset loaders (`isic18.py` by default), metadata utilities, and the `save_graph` helper for superpixel generation.
- `model/`: encoder/decoder modules (Swin/HRNet variants, UNETR-style decoders, ensemble heads).
- `utils/`: training utilities, loss functions, and evaluation metrics.
- `workers/`: high-level `train_net.py` and `test_net.py` orchestration scripts invoked by `run.py`.
- `configs.yaml`: single source of truth for experiment hyperparameters.

## Dependencies

| Library | Notes |
| --- | --- |
| Python ≥ 3.9 | Tested with CPython 3.10 |
| PyTorch ≥ 2.1 & torchvision | CUDA 11.8 build recommended |
| pytorch-lightning ≥ 2.2 | `lightning.pytorch` module |
| scikit-learn | K-fold splits |
| scikit-image | `slic` superpixel generator |
| OpenCV (`opencv-python`) | I/O + augmentations |
| Pillow, imageio | Image conversion & debug exports |
| numpy, scipy, tqdm, einops | Array ops and progress bars |
| wandb, tensorboard (optional) | Experiment tracking |

Install everything with pip (CUDA wheel omitted here):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install lightning wandb tensorboard scikit-learn scikit-image opencv-python pillow imageio einops tqdm pyyaml
```

## Datasets & Offline Superpixel Maps

SET expects the ISIC2018 structure used in `datasets/isic18.py` (update the glob paths in `load_name()` to match your filesystem):

```
/path/to/ISIC2018/
├── Training_Data/*.jpg
└── Training_GroundTruth/{image_id}_segmentation.png
```

### Superpixel Directory Layout

`load_group_associations` looks for `.npy` files under

```
/path/to/superpixels/{SIZE}_{SP_NUM}/T_sp/{SEED}/stage{1-5}/IMAGE_ID.npy
```

Each stage stores increasingly coarser associations (pixels → sp, sp → clusters, …). You can pre-compute them with the `save_graph` helper defined in `datasets/superpixel_generator.py`.

### Generating Superpixel Maps Offline

1. Either export `SET_SUPERPIXEL_ROOT=/path/to/ISIC2018` (the directory that will hold the `{size}_{sp_num}/T_sp/...` folders) or pass `output_root=...` to `save_graph`.
2. Run the helper on every training image:

```bash
python - <<'PY'
from glob import glob
from pathlib import Path
from datasets.superpixel_generator import save_graph

data_root = Path("/path/to/ISIC2018")
images = sorted((data_root / "Training_Data").glob("*.jpg"))
for img in images:
    save_graph(img, seed=10, output_root=data_root)  # seed controls SLIC compactness + folder name
PY
```

`save_graph` will:

- crop/resize each RGB image,
- run `skimage.segmentation.slic`,
- iteratively merge superpixels to produce stage2–stage5 relations via `format_sub_sp`,
- create the required `stage*/` folders under `output_root/{size}_{sp_num}/`, and
- write `.npy` files that `datasets/isic18.py` loads at training time.

If superpixels already exist, the function simply copies them into the `_v2` layout. Regenerate whenever you change `size`, `sp_num`, or the compactness seed.

## Configuring Experiments

All runtime knobs live in `configs.yaml`. The keys mirror the argparse options in `run.py`, so CLI flags override YAML values when provided.

- **Dataset paths**: edit `datasets/isic18.py::load_name` (and other dataset loaders if you plan to train on ISIC16/17/Archive/HAM) so that `input_pattern` and `targetlist` point to your local dataset directories. Likewise, ensure the superpixel directories passed to `load_group_associations` exist on your system.
- **System**: `workers`, `gpu`, `batch_size`, `test_batch_size`, `epoch`, `lr`, `optimizer`, `resume`, `seed`.
- **Dataset**: `dataset_name` (default `ISIC18`), `percent`, `size`, `sp_num`, `stage` list, `ensemble`, `inpainting`.
- **Model**: `model`, `transformer`, `patch_size`, `window_size`, `embed_dim`, `depths`, `num_heads`.
- **Experiment metadata**: `log_name`, `tags`, `description`, `save_name`, `save_results`, `dice_loss`, `kfold`, `evaluate`, `smoke_test`.
- **Logging**: `resume_wandb`, `use_tensorboard`, `id`, `update_config`.

Typical workflow:

1. Duplicate `configs.yaml` if you need multiple runs, or edit in-place.
2. Update dataset paths in `datasets/isic18.py::load_name` if your ISIC root differs.
3. Point `save_name` to a writable directory (checkpoints & logs land there).

## Training & Evaluation

`run.py` wraps the full lifecycle: train → test. Example commands assume superpixel maps already exist.

### Train (then auto-test)

```bash
python run.py --cfg configs.yaml --gpu 0
```

Key optional flags:

- `--workers 8` to control dataloader workers.
- `--batch_size 32 --test_batch_size 4` to override YAML values.
- `--resume 3` to continue from the 4th CV fold checkpoint.
- `--no_crop`, `--dice_loss`, `--transformer` etc. to toggle model behavior.

### Evaluation Only

```bash
python run.py --cfg configs.yaml --gpu 0 --evaluate
```

This bypasses training and immediately runs `workers/test_net.py` with the checkpoints referenced in `save_name`.

### Logging Options

- Default: Weights & Biases (`wandb`). Set `resume_wandb` or `use_tensorboard` in the config to switch modes.
- Offline/debug: enable `smoke_test` to use a CSV logger and skip remote tracking.

## Troubleshooting

- **Missing superpixel files**: verify the directory naming scheme (`{size}_{sp_num}/T_sp/{seed}/stageX`). The loader prints the missing path when a `.npy` is absent.
- **CUDA OOM**: lower `batch_size` and consider reducing `size` or `sp_num` in the config (which also affects the path looked up by `load_group_associations`).
- **Different dataset root**: edit `datasets/isic18.py::load_name` input/target glob patterns or symlink your data to the expected path.

## To-Do

- Release pretrained weights of SET on ISIC2018.

## Citation

If you find SET useful or build upon this codebase, please cite our Medical Image Analysis paper:

```bibtex
@article{wang2025set,
  title={SET: Superpixel Embedded Transformer for skin lesion segmentation},
  author={Wang, Zhonghua and Lyu, Junyan and Tang, Xiaoying},
  journal={Medical Image Analysis},
  pages={103738},
  year={2025},
  publisher={Elsevier}
}
```
