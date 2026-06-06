import os
import argparse
import torch
from monai.networks.nets import SwinUNETR
from trainer import run_training
from utils.data_utils import get_loader

def main():
    parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline for BRATS Challenge")
    parser.add_argument("--data_dir", default="./dataset", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="./jsons/brats21_folds.json", type=str, help="dataset json file")
    parser.add_argument("--logdir", default="./runs", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--max_epochs", default=200, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="AdamW", type=str, help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
    parser.add_argument("--fold", default=0, type=int, help="data fold (0-based) to use for validation")
    parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
    parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="feature size")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=96, type=int)
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float)
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float)
    parser.add_argument("--RandScaleIntensityd_prob", default=0.2, type=float)
    parser.add_argument("--RandShiftIntensityd_prob", default=0.2, type=float)
    parser.add_argument("--smooth_nr", default=1e-5, type=float)
    parser.add_argument("--smooth_dr", default=1e-5, type=float)
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    args = parser.parse_args()
    
    # 损失函数：交叉熵
    loss_func = torch.nn.CrossEntropyLoss()
    
    # 创建模型
    model = SwinUNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
    )
    model.cuda(args.gpu)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    
    # 加载数据
    loader = get_loader(args)
    
    # 训练
    best_acc = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss_func,
        args=args,
    )
    print(f"Training Finished !, Best Dice: {best_acc}")

if __name__ == "__main__":
    main()
