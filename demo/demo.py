# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Qasim Khan from: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import pickle
# fmt: off
import sys  # noqa

sys.path.insert(1, os.path.join(sys.path[0], '..')) # noqa
# fmt: on

import time

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config

from predictor import VisualizationDemo


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="The path to input iamge",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations."
    )
    parser.add_argument(
        "--preds_dest",
        help="Where to save predictions",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        path = args.input
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(
                    len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(
                    args.output, os.path.basename(path))
            else:
                assert len(
                    args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        if args.preds_dest:
            dirname = os.path.dirname(args.preds_dest)
            if os.path.isdir(dirname):
                dest = args.preds_dest
                filename = args.preds_dest
                filename = filename if filename.endswith('.pkl') else filename + ".pkl" 
                with open(filename, "wb") as f:
                    pickle.dump(predictions, f)
            else:
                print("Directory doesn't exist")
