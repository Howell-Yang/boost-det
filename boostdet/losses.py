import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from utils import distance2bbox
import torch.nn.functional as F
import numpy as np
import math

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert (
        len(target) == 2
    ), """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = torch.nonzero((label >= 0) & (label < bg_class_ind), as_tuple=False).squeeze(
        1
    )
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def distribution_focal_loss(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = (
        F.cross_entropy(pred, dis_left, reduction="none") * weight_left
        + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )
    return loss


class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid in QFL supported now."
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            raise NotImplementedError
        return loss_cls


class DistributionFocalLoss(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls

def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])
        >>> bbox_overlaps(bboxes1, bboxes2, mode='giou', eps=1e-7)
        tensor([[0.5000, 0.0000, -0.5000],
                [-0.2500, -0.0500, 1.0000],
                [-0.8371, -0.8766, -0.8214]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(
            bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
        )  # [B, rows, cols, 2]
        rb = torch.min(
            bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
        )  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(
                bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
            )
            enclosed_rb = torch.max(
                bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
            )

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps),
        torch.zeros_like(dx),
    )
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps),
        torch.zeros_like(dy),
    )
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(
        loss_dx.size(0), -1
    )

    loss = torch.where(
        loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta
    ).sum(dim=-1)
    return loss


@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


@weighted_loss
def diou_loss(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


@weighted_loss
def ciou_loss(pred, target, eps=1e-7):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
    loss = 1 - cious
    return loss


class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if (
            (weight is not None)
            and (not torch.any(weight > 0))
            and (reduction != "none")
        ):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


class BoundedIoULoss(nn.Module):
    def __init__(self, beta=0.2, eps=1e-3, reduction="mean", loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * bounded_iou_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


class GIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


class DIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(DIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * diou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


class CIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(CIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * ciou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(-self.reg_max, self.reg_max, 2 * self.reg_max + 1)
        )

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = F.softmax(x.reshape(-1, 2 * self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x

class GeneralizedFocalLoss(nn.Module):
    def __init__(self, reg_max = 7, use_sigmoid=True, num_classes=4):
        super(GeneralizedFocalLoss, self).__init__()
        # 模型输出(多个list): [x16_logits], [x16_reg];
        # 原始targets bbox: B, N, 5 动态shape
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid
        self.regression_scale = torch.nn.Parameter(torch.ones((1, self.num_classes, 1))) # # NHW, self.num_classes, 4
        self.distribution_project = Integral(self.reg_max)
        self.loss_qfl = QualityFocalLoss(
            use_sigmoid=self.use_sigmoid,
            beta=2,
            loss_weight=0.25,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=0.50
        )
        self.loss_bbox = GIoULoss(loss_weight=0.25)
    
    
    def assign_targets(self,
        label_infos, # bbox_targets: B, N, 5
        stride = 16,
        image_size = (256, 256),
        ):

        B = len(label_infos)
        feat_size = (image_size[0] // stride, image_size[1] // stride)
        cls_targets = []
        bbox_targets = []
        corner_targets = []
        num_total_samples = 0
        for i, label_info in enumerate(label_infos):
            cls_target = torch.ones((feat_size[0], feat_size[1]), dtype=torch.long) * self.num_classes # H, W
            corner_target = torch.zeros((feat_size[0], feat_size[1], self.num_classes,  4), dtype=torch.float32) # H, W, num_cls, 4
            bbox_target = torch.zeros((feat_size[0], feat_size[1], self.num_classes,  4), dtype=torch.float32) # H, W, num_cls, 4
            for bbox_info in label_info:
                x1, y1, x2, y2 = bbox_info['bbox']
                category = bbox_info['category']
                cx = (x1 + x2)/2.0
                cy = (y1 + y2)/2.0
                # 每个bbox，会被分配给最近的点 ----> 一个点可能会分配多个bbox
                for offset_x in range(0, 2):
                    for offset_y in range(0, 2):
                        grid_cx = int(cx / stride) + offset_x
                        grid_cy = int(cy / stride) + offset_y
                        if grid_cx >= feat_size[0] or grid_cy >= feat_size[1]:
                            continue
                        if grid_cy < 0 or grid_cy < 0:
                            continue

                        # print(grid_cx, grid_cy)
                        cls_target[grid_cx, grid_cy] = category # 0 - num_cls - 1 for objects, num_cls for background

                        # 计算每个位置, 相对中心位置的偏移量 ---> 由于中心点不一定在bbox内，所以回归坐标是存在负值的情况的
                        left_offset = grid_cx - x1 / stride  # 上下左右坐标偏移量，从当前点，向边界点移动；符合直觉
                        right_offset = x2 / stride - grid_cx
                        top_offset = grid_cy - y1/stride
                        bottom_offset = y2/stride - grid_cy
                        corner_target[grid_cx, grid_cy, category, 0] = left_offset
                        corner_target[grid_cx, grid_cy, category, 1] = right_offset
                        corner_target[grid_cx, grid_cy, category, 2] = top_offset
                        corner_target[grid_cx, grid_cy, category, 3] = bottom_offset

                        # 当前的bbox ---> 原图的位置 ---> 方便画图
                        bbox_target[grid_cx, grid_cy, category, 0] = x1
                        bbox_target[grid_cx, grid_cy, category, 1] = y1
                        bbox_target[grid_cx, grid_cy, category, 2] = x2
                        bbox_target[grid_cx, grid_cy, category, 3] = y2

                        num_total_samples += 1
                        # 回归的目标是每个坐标线，相对中心点的偏移量
                        # 但是，中间还涉及到target层面的scale
                        # 如果target层面有可学习的target scale参数，能够更好地处理
            cls_targets.append(cls_target)
            bbox_targets.append(bbox_target)
            corner_targets.append(corner_target)

        cls_targets = torch.stack(cls_targets)
        bbox_targets = torch.stack(bbox_targets)
        corner_targets = torch.stack(corner_targets)
        # print("cls_targets", np.shape(cls_targets)) # 1, 16, 16
        # print("bbox_targets", np.shape(bbox_targets)) # 1, 16, 16, 4, 4
        # import matplotlib.pyplot as plt
        # plt.imshow(bbox_targets[0, :, :,1, :3].numpy())
        # plt.show()
        return cls_targets, bbox_targets, corner_targets, num_total_samples

    def loss_single(
        self,
        score_pre, # logits before sigmoid: N, num_classes, H, W
        corner_pred, # logits before sigmoid: N, 4 * num_classes *(reg_max + 1), H, W
        cls_target, # N, H, W
        bbox_targets, # N, H, W, num_classes, 4
        corner_targets,
        stride,
        num_total_samples,
        image_size,
    ):

        # score_pre = score_pre.reshape(-1, self.num_classes)  # N, num_classes, H, W
        # corner_pred = corner_pred.reshape(-1, self.num_classes, 4 * (self.reg_max + 1)) # regrssion ---> 不同类别用的相同的回归，这里是不合理的
        corner_pred = corner_pred.permute([0, 2, 3, 1]).reshape(-1, self.num_classes, 4 * (2 * self.reg_max + 1)) # N, C, H, W ---> N, H, W, C
        score_pre = score_pre.permute([0, 2, 3, 1]).reshape(-1, self.num_classes)

        cls_target = cls_target.reshape(-1,) # N x H x W --> NHW, -1 每个位置的类别index
        bbox_targets = bbox_targets.reshape(-1, self.num_classes, 4) # NHW, self.num_classes, 4
        corner_targets = corner_targets.reshape(-1, self.num_classes, 4) # NHW, self.num_classes, 4
        corner_targets = self.regression_scale * corner_targets # (NHW, self.num_classes, 4) x (1, num_classes, 1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes # 每个位置会分配一个类别Index
        pos_index = torch.nonzero(
            (cls_target >= 0) & (cls_target < bg_class_ind), as_tuple=False
        ).squeeze(1) # N x 1

        IoU_score = corner_pred.new_zeros(cls_target.shape)

        if len(pos_index) > 0: # 有正样本
            pos_cls_target = cls_target[pos_index] # 仅训练正样本？

            # 回归时，只回归正样本的bbox坐标
            pos_corner_pred = corner_pred[pos_index]  # (n, self.num_classes, 4 * (reg_max + 1))
            pos_bbox_targets = bbox_targets[pos_index] # (n, self.num_classes, 4) # 这里的regression target是相对于feature map的偏移量
            # 注意: 这里的偏移量，左右/上下的方向是相反的；所以通常情况下，两个都是正数
            pos_corner_targets = corner_targets[pos_index] # (n, self.num_classes, 4) # 这里的regression target是相对于feature map的偏移量

            pos_cls_target = pos_cls_target.unsqueeze(1).unsqueeze(2) # 8x1x60
            pos_corner_pred = pos_corner_pred.gather(dim=1, index=pos_cls_target.expand(-1, -1, pos_corner_pred.size(-1)))     # 8x4x60
            pos_bbox_targets=pos_bbox_targets.gather(dim=1,index=pos_cls_target.expand(-1, -1, pos_bbox_targets.size(-1)))
            pos_corner_targets = pos_corner_targets.gather(dim=1, index=pos_cls_target.expand(-1, -1, pos_corner_targets.size(-1)))
            # print("===selected====", np.shape(pos_cls_target), pos_cls_target) # 8x1x1 # 预期 8x60， 实际8x8x60
            pos_corner_pred = pos_corner_pred.view(-1, 2 * self.reg_max + 1)
            pos_corner_targets = pos_corner_targets.view(-1)

            # 获取score预测值，作为weight
            weight_targets = score_pre.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_index]
            # weight_targets = weight_targets[:, None].expand(-1, 4).reshape(-1)
            # print(np.shape(pos_corner_pred))
            # print(np.shape(pos_bbox_targets), pos_bbox_targets)
            # print(np.shape(pos_corner_targets), pos_corner_targets)
            # print(np.shape(weight_targets), weight_targets) # 8x1

            # dfl loss
            loss_dfl = self.loss_dfl(
                pos_corner_pred, # 8 x 60 --> 8x4x15 --> 32x15
                pos_corner_targets + self.reg_max, # 8 x 4 = 32 x 1, 需要正确的映射0的位置
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1), # 32 x 1
                avg_factor=4.0,
            )

            # 计算IoU损失
            B = score_pre.size(0)
            x_coord = torch.arange(image_size[0]//stride, dtype=torch.int32).unsqueeze(1) # 32 x 1
            y_coord = torch.arange(image_size[1]//stride, dtype=torch.int32).unsqueeze(0) # 1 x 32
            x_coord = x_coord.expand(-1, image_size[1]//stride) # 32 x 32 ---> x[0][0] != x[1][0]
            y_coord = y_coord.expand(image_size[0]//stride, -1)
            xy_coord = torch.stack([x_coord, y_coord], dim=2) # 32 x 32 x 2
            xy_coord = xy_coord.unsqueeze(0).expand(B, -1, -1, -1)
    
            grid_centers = xy_coord.reshape(-1, 2) # x, y 中心点坐标
            pos_grid_centers = grid_centers[pos_index]
            pos_corner_pred = self.distribution_project(pos_corner_pred) # 32x15 ---> 8x4
            # print("pos_corner_pred", pos_corner_pred)
            pos_decode_bbox_pred = distance2bbox(
                pos_grid_centers, pos_corner_pred
            )
            pos_decode_bbox_targets = pos_bbox_targets.view(-1, 4) / stride # 从原图到feature map
            # print("pos_decode_bbox_targets", np.shape(pos_decode_bbox_targets)) # 8,4 
            # print("pos_decode_bbox_pred", np.shape(pos_decode_bbox_pred)) # 8, 4
            # exit(0)
            # 一对一，计算IoU
            IoU_score[pos_index] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            )
            # print(IoU_score[pos_index])
            # pred_corners = pos_corner_pred.reshape(-1, self.reg_max + 1)
            # target_corners = bbox2distance(
            #     pos_grid_cell_centers, pos_decode_bbox_targets, self.reg_max
            # ).reshape(-1)

            # # IoU loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred, # 8 x 4
                pos_decode_bbox_targets, # 8 x 4
                weight=weight_targets[:, None], # 32 x 1
                avg_factor=1.0,
            )

            # qfl loss
            loss_qfl = self.loss_qfl(
                score_pre,
                (cls_target, IoU_score), # 这里只会计算正样本的IoU?
                weight=None,
                avg_factor=num_total_samples,
            )
        else: # 没有正样本
            loss_bbox = corner_pred.sum() * 0
            loss_dfl = corner_pred.sum() * 0
            loss_qfl = corner_pred.sum() * 0
            weight_targets = torch.tensor(0).to(score_pre.device)



        return loss_qfl, loss_bbox, loss_dfl, weight_targets.sum()

    def forward(self, outputs, label_info):
        score_pre, corner_pred = outputs # 假设只有一个level的输出
        stride = 16 # 这个level的stride
        image_size = (256, 256)

        # 第一步，分配target
        cls_targets, bbox_targets, corner_targets, num_total_samples = self.assign_targets(label_info, stride, image_size)
        num_total_samples = max(num_total_samples, 1.0)

        # 第二步，计算每部分的loss
        loss_qfl, loss_bbox, loss_dfl, avg_factor = self.loss_single(
            score_pre, # logits before sigmoid: N, num_classes, H, W
            corner_pred, # logits before sigmoid: N, 4 * num_classes *(reg_max + 1), H, W
            cls_targets, #
            bbox_targets,
            corner_targets,
            stride,
            num_total_samples,
            image_size,
        )

        # 第三步，计算总loss
        avg_factor = max(1.0, avg_factor.item())
        loss_bbox = loss_bbox / avg_factor  #IoU损失，与bbox个数有关
        loss_dfl = loss_dfl / avg_factor # DFL损失，与bbox个数有关
        return loss_qfl, loss_bbox, loss_dfl
    

if __name__ == "__main__":
    # outputs
    num_classes = 4
    stride = 16
    reg_max = 7 # 正负各7
    image_size = (256, 256)
    score_pre = torch.zeros((1, num_classes, image_size[0]//stride, image_size[1]//stride)) # N, C, H, W
    corner_pred = torch.zeros((1, num_classes * 4 * (2 * reg_max + 1), image_size[0]//stride, image_size[1]//stride))
    outputs = (score_pre, corner_pred) # 模型输出的是每个位置的偏移量
    # targets
    label_info = [
        [{"bbox": [32, 32, 64, 64], "category": 0}, {"bbox": [64, 64, 128, 128], "category": 1}]
    ]

    # loss
    loss_module = GeneralizedFocalLoss(reg_max=reg_max)
    loss_qfl, loss_bbox, loss_dfl = loss_module.forward(outputs, label_info)
    print("Losses:")
    print("\tloss_qfl", loss_qfl) 
    # cls loss ---> predicted IoU ----> 这里理论上应该对应每个anchor的IoU
    # 分类和回归，拆分成两个任务；直接预测bbox输出的IoU，导致实际这里的loss包含了bbox预测
    print("\tloss_bbox", loss_bbox)
    print("\tloss_dfl", loss_dfl) # corner loss