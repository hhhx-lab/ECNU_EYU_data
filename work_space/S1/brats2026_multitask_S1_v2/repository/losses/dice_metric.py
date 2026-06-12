import torch


def dice_score(pred, target, eps=1e-6):

    pred = pred.float()
    target = target.float()

    inter = torch.sum(pred * target)

    union = (
        torch.sum(pred)
        +
        torch.sum(target)
    )

    return (
        2.0 * inter + eps
    ) / (
        union + eps
    )
