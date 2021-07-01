import torch


def softmax(inputs: torch.Tensor, mask: torch.BoolTensor, dim: int, epsilon: float = 1e-5):
    exps = torch.exp(inputs)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums


def sigmoid(inputs: torch.Tensor, mask: torch.BoolTensor):
    res = torch.sigmoid(inputs)
    return res * mask.float()
