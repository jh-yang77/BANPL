# BANPL

Code for **BANPL: Learning to Reject with Background-Aware Negative Prompts for Out-of-Distribution Detection**.

BANPL learns negative prompts from context-preserving reconstructions of few-shot ID images. This repository provides the ImageNet-1K workflow: pseudo-negative construction, positive- and negative-prompt learning, and evaluation on Standard, Near-OOD, and Far-OOD benchmarks.

## Installation

The code uses Python 3.10, PyTorch 2.3.1, torchvision 0.18.1, and CUDA. Install the dependencies and the bundled Dassl library:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ./Dassl.pytorch
```

CLIP and LaMa weights are downloaded by their respective loaders on first use.

## Data and few-shot splits

Arrange the datasets as follows. Dataset directories can also be symbolic links.

```text
data/
  imagenet/
    train/<wnid>/*.JPEG
    val/<wnid>/*.JPEG
    classnames.txt
    split_fewshot/
  inaturalist/
  SUN/
  Places/
  texture/
  ssb_hard/
  ninco/
  openimage_o/
```

The three experiment splits are included in `resources/splits/`:

```text
shot_16-seed_1.pkl
shot_16-seed_2.pkl
shot_16-seed_3.pkl
```

Each file contains the same 16,000 image records used in our main experiments: 16 images for each of the 1,000 classes. Image paths are relative to the ImageNet directory, for example `train/n01440764/n01440764_2196.JPEG`. The accompanying CSV files provide readable copies of the sample lists. The PKLs contain paths and labels, not image pixels.

Install the splits for your data location with:

```bash
export BANPL_DATA_ROOT=/absolute/path/to/data
python tools/install_splits.py --root "$BANPL_DATA_ROOT"
```

The installer verifies the PKLs against the recorded hashes and CSV lists, checks that all selected images exist, and joins each relative path to your ImageNet root. Image identities, labels, class names, and sample order stay unchanged. It stops if an existing split contains different records. If you move the dataset, rebuild its `preprocessed.pkl` cache for the new location.

`resources/splits_provenance.json` records both the original experiment-file hashes and the relative-path release hashes. The exported files retain the Dassl `Datum` record format; only the image-path strings change.

You can verify the bundled splits without loading a model or accessing the images:

```bash
python tools/install_splits.py --verify-only
```

## Run the main experiment

The default configuration uses CLIP ViT-B/16, 16-shot training, 16 context tokens, a cumulative-evidence masking threshold of 0.95, and one LaMa reconstruction per ID image. The negative-prompt stage uses a prompt-separation weight of 1.0. Evaluation uses top-2 confidence.

For each seed, generate its pseudo-negatives, train the prompts, and evaluate the resulting checkpoint:

```bash
for seed in 1 2 3; do
  export BANPL_NEG_DIR="$PWD/runs/negatives/seed${seed}"

  python run.py generate -- \
    --root "$BANPL_DATA_ROOT" --seed "$seed" \
    --output-dir "runs/generation/seed${seed}" --neg-dir "$BANPL_NEG_DIR"

  python run.py train -- \
    --root "$BANPL_DATA_ROOT" --seed "$seed" \
    --output-dir "runs/banpl/seed${seed}" --skip-eval

  python run.py eval -- \
    --root "$BANPL_DATA_ROOT" --seed "$seed" \
    --model-dir "runs/banpl/seed${seed}" --load-epoch 90 \
    --output-dir "runs/banpl/seed${seed}/eval"
done
```

Training runs the positive-prompt stage for 30 epochs and the negative-prompt stage for 60 epochs. The reported runs initialized the latter from an existing epoch-30 positive-prompt checkpoint. To use that two-invocation workflow, pass its directory and epoch when starting negative-prompt training:

```bash
python run.py train -- \
  --root "$BANPL_DATA_ROOT" --seed 1 \
  --model-dir /path/to/positive_checkpoint_directory --load-epoch 30 \
  --output-dir runs/banpl/seed1 --skip-eval \
  TRAINER.BANPL.STAGE_EPOCHS 0 OPTIM.MAX_EPOCH 60
```

Evaluate a run trained with this command at epoch 60. Always use the split, positive checkpoint, and pseudo-negative directory from the same seed.

## Evaluation and aggregation

The evaluation uses 50,000 ImageNet-1K validation images as ID and the following OOD groups:

| Benchmark | OOD datasets |
| --- | --- |
| Standard | iNaturalist, SUN, Places, Textures |
| Near-OOD | SSB-Hard, NINCO |
| Far-OOD | iNaturalist, Textures, OpenImage-O |

The evaluator writes per-image scores and reports FPR95 and AUROC. Files ending in `_2_scores.txt` contain the top-2 score used in the main experiment; larger values indicate OOD. The `2` denotes the score variant, not the random seed.

After evaluating all three seeds, summarize their results with:

```bash
python tools/collect_metrics.py \
  --manifest configs/score_manifest.json \
  --output-dir runs/results
```

The collector checks dataset sizes, averages datasets within each benchmark for each seed, and reports the three-seed mean and sample standard deviation. Metrics are percentages; lower FPR95 and higher AUROC are better.

## Code structure

`trainers/banpl.py` implements prompt learning and OOD scoring. `trainers/coop.py` and `scripts/generate_banpl_neg_images.py` construct pseudo-negatives. The training configuration is `configs/trainers/BANPL/imagenet1k.yaml`. `run.py` provides the generation, training, and evaluation entry points.

Run the lightweight checks with `python -m unittest discover -s tests -v`. Use `python run.py --dry-run train -- --seed 1 --skip-eval` to inspect a command without starting training.

## Acknowledgments

This implementation builds on [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp), [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), and [LaMa](https://github.com/advimman/lama), using the [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) wrapper for reconstruction. The bundled third-party components retain their respective license terms.
