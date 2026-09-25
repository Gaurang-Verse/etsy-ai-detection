import torch

from src.losses import FocalLoss


def test_focal_loss_confident_correct_is_near_zero():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    # Very confident, correct prediction (logit strongly positive, target 1)
    logits = torch.tensor([[10.0]])
    targets = torch.tensor([[1.0]])
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01


def test_focal_loss_confident_wrong_is_larger_than_correct():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    logits = torch.tensor([[10.0]])
    correct_target = torch.tensor([[1.0]])
    wrong_target = torch.tensor([[0.0]])

    correct_loss = loss_fn(logits, correct_target)
    wrong_loss = loss_fn(logits, wrong_target)
    assert wrong_loss.item() > correct_loss.item()


def test_focal_loss_matches_bce_shape():
    loss_fn = FocalLoss()
    logits = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8, 1)).float()
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # scalar (mean-reduced)
