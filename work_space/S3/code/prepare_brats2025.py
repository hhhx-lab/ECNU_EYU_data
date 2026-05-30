#!/usr/bin/env python3
"""将原始 BraTS 2025 数据整理为 Swin UNETR 所需格式"""
import os, sys, argparse, shutil, glob

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True, help='原始 Training 目录')
    p.add_argument('--label_dir', required=True, help='corrected-labels 目录')
    p.add_argument('--out_dir', required=True, help='输出目录')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cases = [d for d in os.listdir(args.train_dir) if d.startswith('BraTS-MET') and os.path.isdir(os.path.join(args.train_dir, d))]
    for case in cases:
        src = os.path.join(args.train_dir, case)
        dst = os.path.join(args.out_dir, case)
        os.makedirs(dst, exist_ok=True)
        # 图像重命名
        for old, new in [('t1n.nii.gz', 't1.nii.gz'), ('t1c.nii.gz', 't1ce.nii.gz'), ('t2w.nii.gz', 't2.nii.gz'), ('t2f.nii.gz', 'flair.nii.gz')]:
            f = os.path.join(src, f"{case}-{old}")
            if not os.path.exists(f):
                f = os.path.join(src, old)
            if os.path.exists(f):
                shutil.copy2(f, os.path.join(dst, new))
        # 标签：优先使用 corrected-labels 中的，否则使用原始 seg
        seg_src = os.path.join(args.label_dir, f"{case}-seg.nii.gz")
        if not os.path.exists(seg_src):
            seg_src = os.path.join(src, f"{case}-seg.nii.gz")
        if not os.path.exists(seg_src):
            seg_src = os.path.join(src, "seg.nii.gz")
        if os.path.exists(seg_src):
            shutil.copy2(seg_src, os.path.join(dst, "seg.nii.gz"))
        else:
            print(f"警告: {case} 缺少分割标签")
    print("数据整理完成，输出目录:", args.out_dir)

if __name__ == '__main__':
    main()
