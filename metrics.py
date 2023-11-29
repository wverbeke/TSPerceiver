from mapillary_data_loader.make_class_list import mapillary_class_list
from torchmetrics.classification import MulticlassConfusionMatrix

import numpy as np 
import torch

TINY = 10*10
SMALL = 30*30
MEDIUM = 120*120
LARGE = 200*200
ODD_RATIO = 0.3


def is_tiny(h, w):
    return h*w <= TINY

def is_small(h, w):
    return torch.logical_and(h*w <= SMALL, h*w > TINY)

def is_medium(h, w):
    return torch.logical_and(h*w <= MEDIUM, h*w > SMALL)

def is_large(h, w):
    return torch.logical_and(h*w <= LARGE, h*w > MEDIUM)

def is_huge(h, w):
    return h*w > LARGE

def is_odd(h, w):
    return torch.logical_or(h/w < ODD_RATIO, w/h < ODD_RATIO)


def confusion_matrix(pred_classes, labels):
    cm_computer = MulticlassConfusionMatrix(num_classes=len(mapillary_class_list()))
    return cm_computer(pred_classes, labels).numpy()


def n_true_positives(cm):
    return torch.trace(cm)

def tp(cm, cls_index):
    return cm[cls_index, cls_index]

def fn(cm, cls_index):
    #neg = cm[cls_index]
    #neg = torch.cat(neg[:cls_index], neg[cls_index + 1:])
    #return torch.sum(neg)
    return np.sum(np.delete(cm[cls_index], cls_index))

def fp(cm, cls_index):
    #pos = cm[:, cls_index]
    #pos = torch.cat(pos[: cls_index], pos[cls_index + 1:])
    #return torch.sum(pos)
    return np.sum(np.delete(cm, cls_index, axis=0)[:, cls_index])

def divide_safe(numerator, denominator):
    """Numericall safe division."""
    numerator = numerator.astype(np.float64)
    denominator = denominator.astype(np.float64)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator > 1e-6))

def precision_recall(cm, cls_index):
    tp_ = tp(cm, cls_index)
    fn_ = fn(cm, cls_index)
    fp_ = fp(cm, cls_index)
    recall = divide_safe(tp_, tp_ + fn_)
    precision = divide_safe(tp_, tp_ + fp_)
    return precision, recall


def f1(pr, re):
    num = 2*pr*re
    den = pr + re
    return divide_safe(num, den)


def compute_metrics(pred_classes, labels):
    cm = confusion_matrix(pred_classes, labels)
    n_total = np.sum(cm)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for cls_index in range(cm.shape[0]):
        n_class = np.sum(cm[cls_index])
        cls_w = divide_safe(n_class, n_total)
        pr, re = precision_recall(cm, cls_index)
        f1_score = f1(pr, re)
        total_precision += cls_w*pr
        total_recall += cls_w*re
        total_f1 += cls_w*f1_score
    return total_precision, total_recall, total_f1

def add_metrics_to_dict(current_dict, pred_classes, labels, postfix):
    pr, re, f1 = compute_metrics(pred_classes, labels)
    current_dict[f"precision_{postfix}"] = pr
    current_dict[f"recall_{postfix}"] = re
    current_dict[f"f1_{postfix}"] = f1


def compute_all_metrics(pred_classes, true_classes, heights, widths):
    pred_classes = torch.cat(pred_classes).cpu()
    true_classes = torch.cat(true_classes).cpu()
    heights = torch.cat(heights).cpu()
    widths = torch.cat(widths).cpu()

    # Output dict
    out = {}

    # Total metrics:
    add_metrics_to_dict(out, pred_classes, true_classes, "")

    # Tiny metrics:
    tiny_mask = is_tiny(heights, widths)
    add_metrics_to_dict(out, pred_classes[tiny_mask], true_classes[tiny_mask], "tiny")

    # Small
    small_mask = is_small(heights, widths)
    add_metrics_to_dict(out, pred_classes[small_mask], true_classes[small_mask], "small")

    # Medium
    medium_mask = is_medium(heights, widths)
    add_metrics_to_dict(out, pred_classes[medium_mask], true_classes[medium_mask], "medium")

    # Large
    large_mask = is_large(heights, widths)
    add_metrics_to_dict(out, pred_classes[large_mask], true_classes[large_mask], "large")

    # Huge 
    huge_mask = is_huge(heights, widths)
    add_metrics_to_dict(out, pred_classes[huge_mask], true_classes[huge_mask], "huge")

    # Odd
    odd_mask = is_odd(heights, widths)
    add_metrics_to_dict(out, pred_classes[odd_mask], true_classes[odd_mask], "odd")

    return out


if __name__ == "__main__":
    #heights = [torch.arange(100), torch.arange(100)]
    #widths = heights
    #a = torch.rand(200, len(mapillary_class_list()))
    #pred_classes = [torch.argmax(a[:100], dim=-1), torch.argmax(a[100:], dim=-1)]
    #true_classes = pred_classes

    #out = compute_all_metrics(pred_classes, true_classes, heights, widths)
    #print(out)
    pred_classes = torch.tensor([0]*20 + [0]*20 + [0]*20 + [1]*20 + [2]*20)
    true_classes = torch.tensor([0]*20 + [1]*20 + [2]*60)
    heights = torch.zeros(100)
    widths = torch.zeros(100)
    print(compute_all_metrics(pred_classes, true_classes, heights, widths))

    cm = np.array([[20, 0, 0],
                   [20, 0, 0],
                   [20, 20, 20]])
    print(precision_recall(cm, 0))

    n_total = np.sum(cm)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for cls_index in range(cm.shape[0]):
        n_class = np.sum(cm[cls_index])
        cls_w = divide_safe(n_class, n_total)
        pr, re = precision_recall(cm, cls_index)
        f1_score = f1(pr, re)
        total_precision += cls_w*pr
        total_recall += cls_w*re
        total_f1 += cls_w*f1_score
    print(total_precision, total_recall, total_f1)

