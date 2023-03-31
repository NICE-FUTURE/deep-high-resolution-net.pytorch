import argparse
import os
from pathlib import Path
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.function import inference
from utils.utils import create_logger

import models
import dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument("--save_batch_images_gt", action="store_true")
    parser.add_argument("--save_heatmaps_gt", action="store_true")
    parser.add_argument("--model_file", help="model file (.pth)", type=str, required=True)
    parser.add_argument("--path", help="video path or directory", type=str, required=True)

    args = parser.parse_args()
    return args


def main(video_path):

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    ### check cudnn related settings
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    ### load model
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # prepare data loader
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(cfg.MODEL.IMAGE_SIZE), 
        normalize,
    ])
    dataset = eval('dataset.'+cfg.DATASET.DATASET+"inference")(
        cfg, video_path, transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    ### inference on video file
    save_dir = os.path.join("predicted_frames", Path(video_path).stem)
    os.makedirs(save_dir, exist_ok=True)
    inference(cfg, dataloader, dataset, model, final_output_dir, device, save_dir)


if __name__ == '__main__':
    ### load configuration
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'inference')
    main(args.path)
