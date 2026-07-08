import torch
import os
import numpy as np
from scipy.ndimage import label, find_objects
from monai.metrics import DiceMetric
from tensorboardX import SummaryWriter
import time

def extract_lesions(seg, spacing=(1.,1.,1.)):
    labeled, num = label(seg.astype(np.int32))
    lesions = []
    for i in range(1, num+1):
        mask = (labeled == i)
        volume = np.sum(mask) * np.prod(spacing)
        bbox = find_objects(mask.astype(np.int32))[0]
        lesions.append({'mask': mask, 'bbox': bbox, 'volume': volume})
    return lesions

def match_lesions(pred_lesions, gt_lesions, iou_thresh=0.5):
    if not pred_lesions and not gt_lesions:
        return 0, 0, 0
    if not pred_lesions:
        return 0, 0, len(gt_lesions)
    if not gt_lesions:
        return 0, len(pred_lesions), 0
    iou_matrix = np.zeros((len(pred_lesions), len(gt_lesions)))
    for i, p in enumerate(pred_lesions):
        for j, g in enumerate(gt_lesions):
            inter = np.logical_and(p['mask'], g['mask']).sum()
            union = np.logical_or(p['mask'], g['mask']).sum()
            iou = inter / union if union > 0 else 0.0
            iou_matrix[i,j] = iou
    matched_pred = set()
    matched_gt = set()
    pairs = sorted([(i,j) for i in range(len(pred_lesions)) for j in range(len(gt_lesions))], key=lambda x: -iou_matrix[x[0],x[1]])
    for i,j in pairs:
        if i not in matched_pred and j not in matched_gt and iou_matrix[i,j] >= iou_thresh:
            matched_pred.add(i)
            matched_gt.add(j)
    tp = len(matched_pred)
    fp = len(pred_lesions) - tp
    fn = len(gt_lesions) - tp
    return tp, fp, fn

def compute_detection_metrics(pred_prob, gt_label, spacing=(1.,1.,1.), thresholds=[0.3,0.4,0.5,0.6,0.7,0.8,0.9], volume_thresh=275.0):
    gt_lesions = extract_lesions(gt_label, spacing)
    gt_small = [l for l in gt_lesions if l['volume'] < volume_thresh]
    f1_scores = []
    for th in thresholds:
        pred_bin = (pred_prob > th).astype(np.int64)
        pred_lesions = extract_lesions(pred_bin, spacing)
        pred_small = [l for l in pred_lesions if l['volume'] < volume_thresh]
        tp, fp, fn = match_lesions(pred_small, gt_small)
        prec = tp / (tp + fp) if (tp+fp) > 0 else 0
        rec = tp / (tp + fn) if (tp+fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        f1_scores.append(f1)
    if len(f1_scores) > 1:
        auc = np.trapezoid(f1_scores, thresholds)
    else:
        auc = f1_scores[0] if f1_scores else 0
    return f1_scores, auc

def train_epoch(model, loader, optimizer, loss_func, epoch, args, scaler):
    model.train()
    running_loss = 0.0
    for idx, batch_data in enumerate(loader):
        inputs = batch_data["image"].cuda()
        labels = batch_data["label"].cuda().long()
        if labels.dim() == 5 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = torch.clamp(labels, 0, 4)
        if idx == 0 and epoch == 0:
            print("Labels unique values after clamp:", torch.unique(labels))
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        if idx % 10 == 0:
            print(f"Epoch {epoch} batch {idx} loss: {loss.item():.4f}")
    return running_loss / len(loader)

def val_epoch(model, loader, loss_func, epoch, args):
    model.eval()
    total_loss = 0.0
    # 使用 reduction="none" 获取每个类别的 Dice
    dice_metric = DiceMetric(include_background=True, reduction="none")
    # 用于记录验证集中出现的类别
    present_classes = set()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["image"].cuda()
            y = batch["label"].cuda()
            if y.dim() == 5 and y.shape[1] == 1:
                y = y.squeeze(1)
            y = y.long()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_func(logits, y)
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1, keepdim=True)
            # 更新 Dice 指标（每个类别单独）
            dice_metric(y_pred=pred.long(), y=y)
            # 记录当前 batch 中出现的类别
            batch_classes = torch.unique(y).cpu().numpy()
            present_classes.update(batch_classes)
            if batch_idx == 0:
                print(f"Debug: pred shape {pred.shape}, y shape {y.shape}, unique pred {torch.unique(pred)}, unique y {torch.unique(y)}")
    val_loss = total_loss / len(loader)

    # 获取每个类别的 Dice 值（形状: (num_classes,)）
    dice_per_class = dice_metric.aggregate()  # 这是一个 tensor，长度为 num_classes
    dice_metric.reset()

    # 只对验证集中实际出现的类别计算平均 Dice
    present_classes = sorted(list(present_classes))  # 确保顺序
    valid_dice = [dice_per_class[c] for c in present_classes if c < len(dice_per_class)]
    if valid_dice:
        mean_dice = torch.mean(torch.tensor(valid_dice)).item()
    else:
        mean_dice = 0.0

    print(f"Validation loss: {val_loss:.4f}, Average Dice (over present classes {present_classes}): {mean_dice:.4f}")
    # 可选：打印每个类别的 Dice
    for c in present_classes:
        if c < len(dice_per_class):
            print(f"  Class {c} Dice: {dice_per_class[c].item():.4f}")

    return val_loss, mean_dice

def run_training(model, train_loader, val_loader, optimizer, loss_func, args):
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(logdir=args.logdir)
    scaler = torch.cuda.amp.GradScaler()
    best_dice = 0.0
    for epoch in range(args.max_epochs):
        print(f"{time.asctime()} Epoch: {epoch}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_func, epoch, args, scaler)
        val_loss, val_dice = val_epoch(model, val_loader, loss_func, epoch, args)
        print(f"Epoch {epoch} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_dice: {val_dice:.4f}")
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_dice", val_dice, epoch)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.logdir, "best_model_dice.pth"))
            print(f"Saved new best model with dice {best_dice:.4f}")
    writer.close()
    return best_dice
