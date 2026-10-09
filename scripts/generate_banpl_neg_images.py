import argparse
import os
import sys

import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger
from yacs.config import CfgNode as CN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import datasets.imagenet  # noqa: E402,F401
import trainers.coop  # noqa: E402,F401


def extend_cfg(cfg):
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "amp"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATASET.ROOT = args.root
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SEED = args.seed
    cfg.TRAINER.NAME = "CoOp"
    cfg.MODEL.BACKBONE.NAME = args.backbone
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("BANPL negative-image generation")
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}")

    trainer = build_trainer(cfg)
    split_file = f"shot_{args.nshot}-seed_{args.seed}.pkl"
    trainer.model.create_images_from_split(
        split_file=split_file,
        neg_dir=args.neg_dir,
        mask_ratio_range=(args.mask_low, args.mask_high),
        mask_mode=args.mask_mode,
        cam_mass_levels=args.cam_mass_levels,
        batch_size=args.batch_size,
        num_neg_per_image=args.num_neg,
        num_workers=args.num_workers,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        inpaint_method=args.inpaint_method,
        mask_source=args.mask_source,
        skip_existing=args.skip_existing,
        overwrite_labels=not args.skip_existing,
        max_images=args.max_images,
    )
    print(f"Done: {args.neg_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nshot", type=int, default=16)
    parser.add_argument("--config-file", type=str, default="configs/trainers/BANPL/vit_b16.yaml")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/imagenet.yaml")
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--neg-dir", type=str, required=True)
    parser.add_argument("--num-neg", type=int, default=1)
    parser.add_argument("--mask-low", type=float, default=0.3)
    parser.add_argument("--mask-high", type=float, default=0.7)
    parser.add_argument("--mask-mode", type=str, default="mass", choices=["ratio", "mass"])
    parser.add_argument("--cam-mass-levels", type=float, nargs="+", default=[0.95])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-format", type=str, default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--inpaint-method", type=str, default="lama")
    parser.add_argument(
        "--mask-source",
        type=str,
        default="cam",
        choices=[
            "cam",
            "attention_rollout",
            "attention_rollout_area_matched",
            "gradcam_area_matched",
            "gradcampp_area_matched",
        ],
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    main(parser.parse_args())
