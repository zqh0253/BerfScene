# python3.8
"""Contains some functions related to cameras, including rotation
transformation, projection, etc."""

import torch
import torch.nn.functional as F

def camera_9d_to_16d(d9):
    d6, translation = d9[..., :6], d9[..., 6:]
    rotation = rotation_6d_to_matrix(d6)
    RT = torch.eye(4).to(device=d9.device, dtype=d9.dtype).reshape(
        1, 4, 4).repeat(d6.size(0), 1, 1)
    RT[:, :3, :3] = rotation
    RT[:, :3, -1] = translation
    return RT.reshape(-1, 16)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035.
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)