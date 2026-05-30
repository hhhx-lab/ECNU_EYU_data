#!/usr/bin/env python3
"""
生成带 fold 字段的 JSON 文件，用于固定比例划分训练/验证集。
用法: python generate_json.py <data_ready_dir> --val_ratio 0.1
"""
import os
import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='预处理后的数据目录')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例，默认0.1')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    cases = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    if not cases:
        print(f"错误: {args.data_dir} 中没有病例文件夹")
        return

    random.seed(args.seed)
    random.shuffle(cases)

    split_idx = max(1, int(len(cases) * (1 - args.val_ratio)))
    train_cases = cases[:split_idx]
    val_cases = cases[split_idx:]

    data_list = []
    for c in train_cases:
        data_list.append({"image": c, "label": c, "fold": 0})
    for c in val_cases:
        data_list.append({"image": c, "label": c, "fold": 1})

    with open("brats25_full.json", "w") as f:
        json.dump({"training": data_list}, f, indent=2)

    print(f"生成完成: {len(train_cases)} 训练 (fold=0), {len(val_cases)} 验证 (fold=1) -> brats25_full.json")
    print(f"验证集比例: {len(val_cases)/len(cases):.2%}")

if __name__ == '__main__':
    main()
