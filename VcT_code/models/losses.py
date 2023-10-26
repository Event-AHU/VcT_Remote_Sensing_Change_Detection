import torch
import torch.nn.functional as F


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W  torch.Size([8, 2, 64, 64])
    :param target: torch.Tensor, N*1*H*W,/ N*H*W  torch.Size([8, 1, 256, 256])
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)  # torch.Size([8, 256, 256])
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)  # torch.Size([8, 2, 256, 256])

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def cross_entropy_2(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W  torch.Size([8, 1, 64, 64])
    :param target: torch.Tensor, N*1*H*W,/ N*H*W  torch.Size([8, 1, 256, 256])
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    if target.dim() == 4:
        tgt = torch.squeeze(target, dim=1)  # torch.Size([8, 256, 256])
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=tgt.shape[1:], mode='bilinear',align_corners=True).clone()  # torch.Size([8, 1, 256, 256])

    return F.binary_cross_entropy_with_logits(input=input, target=target.float())
