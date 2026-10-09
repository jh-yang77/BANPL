import argparse
import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import trainers.banpl
import datasets.imagenet

import numpy as np
from utils.train_eval_util import set_val_loader, set_ood_loader_ImageNet
from utils.detection_util import get_and_print_results
from utils.plot_util import plot_distribution

import os

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed is not None:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    cfg.TRAIN_MOD = args.train


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.BANPL = CN()
    cfg.TRAINER.BANPL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.BANPL.CSC = False  # class-specific context
    cfg.TRAINER.BANPL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.BANPL.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.BANPL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.BANPL.STAGE_EPOCHS = 40
    cfg.TRAINER.BANPL.POS_LR = 0.002
    cfg.TRAINER.BANPL.NEG_LR = 0.002
    cfg.TRAINER.BANPL.TWO_STAGE_LR = True
    cfg.TRAINER.BANPL.NEG_SELECT_BATCH_SIZE = 1024
    cfg.TRAINER.BANPL.NEG_LOADER_NUM_WORKERS = 8
    cfg.TRAINER.BANPL.HARD_NEG_FRACTION = 0.25
    cfg.TRAINER.BANPL.HARD_NEG_MODE = "low_prob_fraction"
    cfg.TRAINER.BANPL.SEP_WEIGHT = 2.0
    cfg.TRAINER.BANPL.ID_CROP_SCALE_MIN = 0.08
    cfg.TRAINER.BANPL.NEG_CROP_SCALE_MIN = 0.08
    cfg.TRAINER.BANPL.POS_RRCROP_SCALE = (0.08, 1.0)
    cfg.TRAINER.BANPL.NEG_RRCROP_SCALE = (0.08, 1.0)
    cfg.TRAINER.BANPL.AUG_MODE = "rrcrop"
    cfg.TRAINER.BANPL.LEGACY_COSINE_TMAX = 100
    cfg.TRAINER.BANPL.LEGACY_COSINE_ETA_MIN = 1e-5
    cfg.TRAINER.BANPL.RISE_FALL_SCHEDULER = False
    cfg.TRAINER.BANPL.RISE_FALL_HALF_CYCLE = 30
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    import clip
    cfg = setup_cfg(args)
    _, preprocess = clip.load(cfg.MODEL.BACKBONE.NAME)

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    trainer.load_model(args.model_dir, epoch=args.load_epoch)

    # if args.eval_only:
    #     trainer.load_model(args.model_dir, epoch=args.load_epoch)
    #     trainer.test()
    #     return

    if args.train:
        trainer.train()
        if args.skip_eval:
            print("Skipping OOD evaluation as requested")
            return
        # ----------------- 定义分组 -----------------
    oodv1_ood = ["iNaturalist", "SUN", "Places", "Textures"]
    near_ood    = ["SSB-Hard", "NINCO"]
    far_ood     = ["iNaturalist", "Textures", "OpenImage-O"]

    out_datasets = ["SSB-Hard", "NINCO", "iNaturalist", "Textures", "OpenImage-O", "SUN", "Places"]

    # ID 数据
    id_data_loader = set_val_loader(args, preprocess)
    in_score_1, in_score_2, in_score_3, in_score_4 = trainer.test_ood(id_data_loader, is_list=True)

    # 三个基准分别的指标列表
    results = {
        "oodv1": {"fpr": [[], [], [], []], "auroc": [[], [], [], []], "aupr": [[], [], [], []]},
        "near":    {"fpr": [[], [], [], []], "auroc": [[], [], [], []], "aupr": [[], [], [], []]},
        "far":     {"fpr": [[], [], [], []], "auroc": [[], [], [], []], "aupr": [[], [], [], []]},
    }

    # # 所有 OOD loaders 一次性取好
    # ood_loader_dict = set_ood_loader_ImageNet(args, preprocess)

    for out_dataset in out_datasets:
        print(f"Evaluating OOD dataset: {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args,out_dataset, preprocess)
        out_score_1, out_score_2, out_score_3, out_score_4 = trainer.test_ood(ood_loader, is_list=True)

        score_types = {
            "1": (in_score_1, out_score_1, 0),
            "2": (in_score_2, out_score_2, 1),
            "3": (in_score_3, out_score_3, 2),
            "4": (in_score_4, out_score_4, 3),
        }

        os.makedirs(args.output_dir, exist_ok=True)

        for score_name, (in_scores, out_scores, idx) in score_types.items():
            print(f"{score_name} score")
            auroc_list, aupr_list, fpr_list = [], [], []
            get_and_print_results(args, in_scores, out_scores, auroc_list, aupr_list, fpr_list)

            # ----------------- 多重分组逻辑 -----------------
            groups = []
            if out_dataset in oodv1_ood:
                groups.append("oodv1")
            if out_dataset in near_ood:
                groups.append("near")
            if out_dataset in far_ood:
                groups.append("far")

            for group in groups:
                results[group]["fpr"][idx].extend(fpr_list)
                results[group]["auroc"][idx].extend(auroc_list)
                results[group]["aupr"][idx].extend(aupr_list)

            # 保存分数日志
            save_path = f"{args.output_dir}/{out_dataset}_{score_name}_scores.txt"
            with open(save_path, "w") as f:
                for s in in_scores:
                    f.write(f"in\t{s:.10f}\n")
                for s in out_scores:
                    f.write(f"out\t{s:.10f}\n")

            plot_distribution(args, in_scores, out_scores, out_dataset, score=score_name)

    # ----------------- 打印分基准的平均结果 -----------------
    for group in ["oodv1", "near", "far"]:
        print(f"\n==== {group.upper()} OOD RESULTS ====")
        for i in range(4):
            fpr   = np.mean(results[group]["fpr"][i])   if results[group]["fpr"][i] else 0
            auroc = np.mean(results[group]["auroc"][i]) if results[group]["auroc"][i] else 0
            aupr  = np.mean(results[group]["aupr"][i])  if results[group]["aupr"][i] else 0
            print(f"[OOD{i+1}]  FPR: {fpr:.4f}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=os.environ.get("BANPL_DATA_ROOT", "data"), help="path to dataset")
    parser.add_argument('--in_dataset', default='imagenet', type=str,
                        choices=['imagenet'], help='in-distribution dataset')
    parser.add_argument("--output-dir", type=str, default="runs/outputs/banpl_seed1", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/BANPL/imagenet1k.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/imagenet.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="BANPL", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", default="", type=str)
    parser.add_argument(
        "--load-epoch", type=int,default=None, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--train", type=int,default=False,help="do not call trainer.train()"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="finish after training without evaluating real OOD datasets",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for ID/OOD score extraction")
    parser.add_argument("--num-workers", type=int, default=8, help="num workers for ID/OOD score extraction")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
   
    # augment for BANPL

    args = parser.parse_args()
    main(args)
