import torch

from sali_pycce.losses import WeightedBCEDiceLoss, WeightedMSELoss, make_heatmap_loss


def test_weighted_bce_dice_penalizes_missed_spin_more_than_background():
    target = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    target[..., 7, 9] = 1.0

    missed_spin = torch.zeros_like(target)
    missed_spin[..., 7, 9] = 0.01
    empty_target = torch.zeros_like(target)
    false_background = torch.zeros_like(target)
    false_background[..., 0, 0] = 0.01

    loss_fn = WeightedBCEDiceLoss(pos_weight=40.0, dice_weight=0.0)

    assert loss_fn(missed_spin, target) > 100 * loss_fn(false_background, empty_target)


def test_make_heatmap_loss_uses_detection_loss_by_default():
    loss_fn = make_heatmap_loss("weighted-bce-dice", pos_weight=30.0, dice_weight=0.5)

    assert isinstance(loss_fn, WeightedBCEDiceLoss)


def test_weighted_mse_penalizes_missed_spin_more_than_background():
    target = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    target[..., 7, 9] = 1.0
    missed_spin = torch.zeros_like(target)
    empty_target = torch.zeros_like(target)
    false_background = torch.zeros_like(target)
    false_background[..., 0, 0] = 0.01

    loss_fn = WeightedMSELoss(pos_weight=200.0)

    assert loss_fn(missed_spin, target) > 1000 * loss_fn(false_background, empty_target)


def test_make_heatmap_loss_uses_weighted_mse_by_name():
    loss_fn = make_heatmap_loss("weighted-mse", pos_weight=150.0)

    assert isinstance(loss_fn, WeightedMSELoss)
