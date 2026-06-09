#!/usr/bin/env python3
"""
配置文件生成器
将 YAML 配置与命令行参数合并，生成临时配置文件供训练脚本使用。

Usage:
    python config_generator.py --config configs/default.yaml --max_epoch 20 --batch_size 4
"""

import argparse
import os
import sys
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='SegMamba Config Generator')

    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to YAML config file')
    parser.add_argument('-o', '--output', type=str, default='./temp_config.yaml',
                        help='Output config file path')

    # 数据配置
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--logdir', type=str)

    # 训练配置
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_every', type=int)
    parser.add_argument('--num_gpus', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--seed', type=int)

    # 模型配置
    parser.add_argument('--roi_size', type=int, nargs='+')

    # 模型架构参数
    parser.add_argument('--in_chans', type=int)
    parser.add_argument('--out_chans', type=int)
    parser.add_argument('--depths', type=int, nargs='+')
    parser.add_argument('--feat_size', type=int, nargs='+')

    # 优化器参数
    parser.add_argument('--optimizer_type', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--nesterov', type=bool)

    # 学习率调度器
    parser.add_argument('--scheduler_type', type=str)

    # 其他
    parser.add_argument('--augmentation', type=bool)
    parser.add_argument('--small_lesion_threshold', type=float)

    return parser.parse_args()


def flatten_dict(d, parent_key='', sep='_'):
    """将嵌套字典展平为单层字典，key 用下划线连接"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='_'):
    """将单层字典还原为嵌套字典"""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_inner = result
        for part in parts[:-1]:
            if part not in d_inner:
                d_inner[part] = {}
            d_inner = d_inner[part]
        d_inner[parts[-1]] = value
    return result


def merge_config(base_config, cli_args):
    """合并 YAML 配置与命令行参数"""
    merged = base_config.copy()

    # 将 CLI 参数转换为扁平字典
    cli_dict = {k: v for k, v in vars(cli_args).items()
                if v is not None and k != 'config' and k != 'output'}

    # 处理嵌套参数 (如 model_depths -> model.depths)
    for key, value in cli_dict.items():
        # 处理带下划线的嵌套参数
        if '_' in key:
            parts = key.split('_', 1)
            if parts[0] in merged and isinstance(merged.get(parts[0]), dict):
                merged[parts[0]][parts[1]] = value
            else:
                merged[key] = value
        else:
            merged[key] = value

    return merged


def save_config(config, output_path):
    """保存配置到 YAML 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Config saved to: {output_path}")


def main():
    args = parse_args()

    # 1. 加载基础配置
    base_config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f) or {}
        print(f"Loaded config from: {args.config}")
    else:
        print("No config file specified, using defaults")

    # 2. 合并命令行覆盖参数
    final_config = merge_config(base_config, args)

    # 3. 保存合并后的配置
    save_config(final_config, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
