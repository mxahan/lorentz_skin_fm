# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Pairwise Lorentzian inner product.

    Args:
        x: Spatial components, shape [B1, D].
        y: Spatial components, shape [B2, D].
        curv: Positive curvature magnitude.

    Returns:
        Lorentzian inner products, shape [B1, B2].
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    return x @ y.T - x_time @ y_time.T


def pairwise_dist(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Pairwise Lorentz geodesic distance.

    Args:
        x: Spatial components, shape [B1, D].
        y: Spatial components, shape [B2, D].
        curv: Positive curvature magnitude.
        eps: Numerical stability constant.

    Returns:
        Pairwise distances, shape [B1, B2].
    """
    c_xyl = -curv * pairwise_inner(x, y, curv)
    return torch.acosh(torch.clamp(c_xyl, min=1 + eps)) / curv**0.5


def pairwise_dist_sq(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 0.0) -> Tensor:
    """
    Pairwise squared Lorentzian distance.

    Args:
        x: Spatial components, shape [B1, D].
        y: Spatial components, shape [B2, D].
        curv: Positive curvature magnitude.
        eps: Minimum output value.

    Returns:
        Pairwise squared distances, shape [B1, B2].
    """
    xyl = pairwise_inner(x, y, curv)
    return torch.clamp(-2.0 / curv - 2.0 * xyl, min=eps)


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Exponential map from the tangent space at the Lorentz origin.

    Args:
        x: Tangent vectors, shape [..., D].
        curv: Positive curvature magnitude.
        eps: Numerical stability constant.

    Returns:
        Spatial components of hyperboloid points, shape [..., D].
    """
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))  # avoid sinh overflow
    return torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Logarithmic map to the tangent space at the Lorentz origin.

    Args:
        x: Spatial components of hyperboloid points, shape [..., D].
        curv: Positive curvature magnitude.
        eps: Numerical stability constant.

    Returns:
        Tangent vectors at the origin, shape [..., D].
    """
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    return distance0 * x / torch.clamp(rc_xnorm, min=eps)


def lorentz_sq_mean(x: Tensor, weights: Tensor | None = None, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Weighted Lorentz squared-distance mean.

    Args:
        x: Spatial components, shape [N, D] or [B, N, D].
        weights: Point weights, shape [N] or [B, N]. Default: uniform.
        curv: Positive curvature magnitude.
        eps: Numerical stability constant.

    Returns:
        Spatial components of the Lorentz mean, shape [D] or [B, D].
    """
    if x.ndim not in (2, 3):
        raise ValueError("x must have shape [N,D] or [B,N,D]")

    beta = 1.0 / curv
    x_time = torch.sqrt(beta + x.square().sum(dim=-1))  # [N] or [B,N]

    if weights is None:
        weights = torch.ones_like(x_time)
    else:
        weights = weights.to(device=x.device, dtype=x.dtype)
        if x.ndim == 3 and weights.ndim == 1:
            weights = weights.unsqueeze(0)

    s_time = (weights * x_time).sum(dim=-1)            # [] or [B]
    s_space = (weights.unsqueeze(-1) * x).sum(dim=-2)  # [D] or [B,D]

    # Normalize onto <mu, mu>_L = -beta
    norm = torch.sqrt(torch.clamp(s_time.square() - s_space.square().sum(dim=-1), min=eps))
    sqrt_beta = torch.sqrt(torch.as_tensor(beta, device=x.device, dtype=x.dtype))
    return sqrt_beta * s_space / norm.unsqueeze(-1)


def half_aperture(x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8) -> Tensor:
    """
    Half-aperture of the hyperbolic entailment cone.

    Args:
        x: Cone-apex spatial components, shape [B, D].
        curv: Positive curvature magnitude.
        min_radius: Minimum radius parameter K.
        eps: Numerical stability constant.

    Returns:
        Half-aperture angles in radians, shape [B].
    """
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    return torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Exterior angle at x in the hyperbolic triangle O-x-y.

    Args:
        x: Spatial components of cone apex, shape [B, D].
        y: Spatial components of target points, shape [B, D].
        curv: Positive curvature magnitude.
        eps: Numerical stability constant.

    Returns:
        OXY angles in radians, shape [B].
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


def oxy_angle_modified(x: Tensor, y: Tensor, labels: Tensor, curv: float | Tensor = 1.0,
                       min_radius: float = 0.1, eps: float = 1e-8):
    """
    OXY angle, distance, and cone aperture for spatial feature maps.

    Args:
        x: Image features, shape [B, H, W, D].
        y: Class prototypes, shape [C, D].
        labels: Class indices, shape [B, H, W].
        curv: Positive curvature magnitude.
        min_radius: Minimum cone radius K.
        eps: Numerical stability constant.

    Returns:
        angle: Label-selected OXY angles, shape [B, H, W].
        distance: Distances to all prototypes, shape [B, H, W, C].
        half_aperture: Label-selected cone half-apertures, shape [B, H, W].
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_space = torch.sum(y**2, dim=-1, keepdim=True)
    y_time = torch.sqrt(1 / curv + y_space)

    xyl = x @ y.T - x_time @ y_time.T
    c_xyl = curv * xyl
    distance = torch.acosh(torch.clamp(-c_xyl, min=1 / curv + eps)) / curv**0.5

    c_xyl = torch.gather(c_xyl, dim=3, index=labels.unsqueeze(-1)).squeeze(-1)
    y_space = torch.sqrt(y_space.squeeze())[labels]  # selected prototype ||y_space||
    y_time = y_time.squeeze()[labels]
    x_time = x_time.squeeze()

    acos_numer = x_time + c_xyl * y_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (y_space * acos_denom + eps)
    angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    asin_input = 2 * min_radius / (y_space * curv**0.5 + eps)
    half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return angle, distance, half_aperture


def oxy_angle_dist(x: Tensor, y: Tensor, labels: Tensor, curv: float | Tensor = 1.0,
                   min_radius: float = 0.1, eps: float = 1e-6):
    """
    OXY angle, distance, and cone aperture for labeled embeddings.

    Args:
        x: Sample embeddings, shape [B, D].
        y: Class prototypes, shape [C, D].
        labels: Class indices, shape [B].
        curv: Positive curvature magnitude.
        min_radius: Minimum cone radius K.
        eps: Numerical stability constant.

    Returns:
        angle: OXY angle to the labeled prototype, shape [B].
        distance: Distance to every prototype, shape [B, C].
        half_aperture: Half-aperture of the labeled prototype, shape [B].
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # [B,1]
    y_space = torch.sum(y**2, dim=-1, keepdim=True)                        # [C,1]
    y_time = torch.sqrt(1 / curv + y_space)                                # [C,1]

    c_xyl = curv * (x @ y.T - x_time @ y_time.T)                           # [B,C]
    distance = torch.acosh(torch.clamp(-c_xyl, min=1 + eps)) / curv**0.5   # [B,C]

    idx = torch.arange(x.shape[0], device=x.device)
    c_xyl = c_xyl[idx, labels]                                             # [B]
    y_space = torch.sqrt(y_space.squeeze(-1))[labels]                       # [B]
    y_time = y_time.squeeze(-1)[labels]                                    # [B]
    x_time = x_time.squeeze(-1)                                            # [B]

    # Clamp away from -1 so c_xyl² - 1 does not collapse
    c_xyl_safe = torch.clamp(c_xyl, max=-(1 + eps))
    acos_numer = x_time + c_xyl_safe * y_time
    acos_denom = torch.sqrt(c_xyl_safe**2 - 1)
    acos_input = acos_numer / (y_space * acos_denom + eps)
    angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    # Prototype must lie outside the minimum cone radius
    sqrt_c = torch.sqrt(torch.as_tensor(curv, device=y.device, dtype=y.dtype))
    min_space = 2 * min_radius / sqrt_c
    if torch.any(y_space <= min_space):
        raise ValueError("Prototype lies inside the minimum radius required for a valid entailment cone.")

    y_space_safe = torch.clamp(y_space, min=min_space + eps)
    asin_input = 2 * min_radius / (y_space_safe * sqrt_c)
    half_aperture = torch.asin(torch.clamp(asin_input, max=1 - eps))

    return angle, distance, half_aperture


def oxy_dist_angle_no_label(x: Tensor, y: Tensor, curv: float | Tensor = 1.0,
                            min_radius: float = 0.1, eps: float = 1e-6):
    """
    OXY angle, distance, and cone aperture for all sample-prototype pairs.

    Args:
        x: Sample embeddings, shape [B, D].
        y: Class prototypes, shape [C, D].
        curv: Positive curvature magnitude.
        min_radius: Minimum cone radius K.
        eps: Numerical stability constant.

    Returns:
        angle: OXY angles for all sample-prototype pairs, shape [B, C].
        distance: Distances for all sample-prototype pairs, shape [B, C].
        half_aperture: Prototype cone half-apertures, shape [C].
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # [B,1]
    y_space = torch.sum(y**2, dim=-1, keepdim=True)                        # [C,1]
    y_time = torch.sqrt(1 / curv + y_space)                                # [C,1]

    c_xyl = curv * (x @ y.T - x_time @ y_time.T)                           # [B,C]
    c_xyl_safe = torch.clamp(c_xyl, max=-(1 + eps))
    distance = torch.acosh(torch.clamp(-c_xyl, min=1 + eps)) / curv**0.5

    y_space = torch.sqrt(y_space.squeeze(-1))                              # [C]
    y_time = y_time.squeeze(-1)                                            # [C]
    x_time = x_time.squeeze(-1)                                            # [B]

    acos_numer = x_time.unsqueeze(1) + c_xyl_safe * y_time
    acos_denom = torch.sqrt(c_xyl_safe**2 - 1)
    acos_input = acos_numer / (y_space * acos_denom + eps)
    angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    y_space_safe = torch.clamp(y_space, min=eps**0.5)
    asin_input = 2 * min_radius / (y_space_safe * curv**0.5)
    half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return angle, distance, half_aperture


def oxy_angle_full(x: Tensor, y: Tensor, curv: float | Tensor = 1.0,
                   min_radius: float = 0.1, eps: float = 1e-8):
    """
    OXY angle between spatial features and all class prototypes.

    Args:
        x: Spatial features, shape [B, H, W, D].
        y: Class prototypes, shape [C, D].
        curv: Positive curvature magnitude.
        min_radius: Unused; retained for API compatibility.
        eps: Numerical stability constant.

    Returns:
        OXY angles, shape [B, H, W, C].
    """
    C = y.shape[0]
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_space = torch.sum(y**2, dim=-1, keepdim=True)
    y_time = torch.sqrt(1 / curv + y_space)

    c_xyl = curv * (x @ y.T - x_time @ y_time.T)
    y_time = y_time.view(1, 1, 1, C)
    y_space = torch.sqrt(y_space).view(1, 1, 1, C)

    acos_numer = x_time + c_xyl * y_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (y_space * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


# Can be used in lieu of oxy_angle_full; supports [B,N,D] and [B,H,W,D]
def oxy_angle_full_gnr(x: Tensor, y: Tensor, curv: float | Tensor = 1.0,
                       min_radius: float = 0.1, eps: float = 1e-6):
    """
    General OXY angle between features and all class prototypes.

    Args:
        x: Features, shape [B, N, D] or [B, H, W, D].
        y: Class prototypes, shape [C, D].
        curv: Positive curvature magnitude.
        min_radius: Unused; retained for API compatibility.
        eps: Numerical stability constant.

    Returns:
        OXY angles, shape [B, N, C] or [B, H, W, C].
    """
    C = y.shape[0]
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_space = torch.sum(y**2, dim=-1, keepdim=True)
    y_time = torch.sqrt(1 / curv + y_space)

    c_xyl = curv * (x @ y.T - x_time @ y_time.T)
    expand_shape = (1, 1, C) if x.ndim == 3 else (1, 1, 1, C)

    y_time = y_time.view(expand_shape)
    y_space = torch.sqrt(y_space).view(expand_shape)

    acos_numer = x_time + c_xyl * y_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (y_space * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


# Computes angle and distance corresponding to batched mask prototypes
def oxy_angle_mask(x: Tensor, y: Tensor, curv: float | Tensor = 1.0,
                   min_radius: float = 0.1, eps: float = 1e-6) -> Tensor:
    """
    OXY angle and distance between feature maps and batched mask prototypes.

    Args:
        x: Spatial feature maps, shape [B, H, W, D].
        y: Batched mask prototypes, shape [B, Q, D].
        curv: Positive curvature magnitude.
        min_radius: Unused; retained for API compatibility.
        eps: Numerical stability constant.

    Returns:
        angle: OXY angles, shape [B, H, W, Q].
        distance: Lorentz distances, shape [B, H, W, Q].
    """
    B, _, _, _ = x.shape
    Q = y.shape[1]

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # [B,H,W,1]
    y_space = torch.sum(y**2, dim=-1, keepdim=True)                        # [B,Q,1]
    y_time = torch.sqrt(1 / curv + y_space)                                # [B,Q,1]

    xyl_spatial = torch.einsum("bhwd,bqd->bhwq", x, y)                     # [B,H,W,Q]
    xyl_time = x_time * y_time.view(B, 1, 1, Q)
    c_xyl = curv * (xyl_spatial - xyl_time)

    distance = torch.acosh(torch.clamp(-c_xyl, min=1 + eps)) / curv**0.5

    y_time = y_time.view(B, 1, 1, Q)
    y_space = torch.sqrt(y_space).view(B, 1, 1, Q)

    acos_numer = x_time + c_xyl * y_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (y_space * acos_denom + eps)
    angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return angle, distance


# def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
#     """
#     Compute pairwise Lorentzian inner product between input vectors.
#     Args:
#         x: Tensor of shape `(B1, D)` giving a space components of a batch
#             of vectors on the hyperboloid.
#         y: Tensor of shape `(B2, D)` giving a space components of another
#             batch of points on the hyperboloid.
#         curv: Positive scalar denoting negative hyperboloid curvature.
#         eps: Small float number to avoid numerical instability.
#     Returns:
#         Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
#         between input vectors.
#     """

#     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
#     y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
#     xyl = x @ y.T - x_time @ y_time.T
#     return xyl


# def pairwise_dist(
#     x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
# ) -> Tensor:
#     """
#     Compute the pairwise geodesic distance between two batches of points on
#     the hyperboloid.
#     Args:
#         x: Tensor of shape `(B1, D)` giving a space components of a batch
#             of point on the hyperboloid.
#         y: Tensor of shape `(B2, D)` giving a space components of another
#             batch of points on the hyperboloid.
#         curv: Positive scalar denoting negative hyperboloid curvature.
#         eps: Small float number to avoid numerical instability.
#     Returns:
#         Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
#         connecting the input points.
#     """

#     # Ensure numerical stability in arc-cosh by clamping input.
#     c_xyl = -curv * pairwise_inner(x, y, curv)
#     _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
#     return _distance / curv**0.5


# def pairwise_dist_sq(
#     x: Tensor,
#     y: Tensor,
#     curv: float | Tensor = 1.0,
#     eps: float = 0.0,
# ) -> Tensor:
#     """
#     Pairwise squared Lorentzian distance.
#     Args:
#         x: Tensor of shape (B1, D), spatial components.
#         y: Tensor of shape (B2, D), spatial components.
#         curv: Positive curvature magnitude (1 / beta).
#         eps: Minimum distance value for numerical safety.

#     Returns:
#         Tensor of shape (B1, B2).
#     """
#     xyl = pairwise_inner(x, y, curv)
#     dist_sq = -2.0 / curv - 2.0 * xyl
#     return torch.clamp(dist_sq, min=eps)


# def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
#     """
#     Map points from the tangent space at the vertex of hyperboloid, on to the
#     hyperboloid. This mapping is done using the exponential map of Lorentz model.

#     Args:
#         x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
#             onto the hyperboloid. These vectors are interpreted as velocity
#             vectors in the tangent space at the hyperboloid vertex.
#         curv: Positive scalar denoting negative hyperboloid curvature.
#         eps: Small float number to avoid division by zero.

#     Returns:
#         Tensor of same shape as `x`, giving space components of the mapped
#         vectors on the hyperboloid.
#     """

#     rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

#     # Ensure numerical stability in sinh by clamping input.
#     sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
#     _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
#     return _output


# def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
#     """
#     Inverse of the exponential map: map points from the hyperboloid on to the
#     tangent space at the vertex, using the logarithmic map of Lorentz model.
#     Args:
#         x: Tensor of shape `(B, D)` giving space components of points
#             on the hyperboloid.
#         curv: Positive scalar denoting negative hyperboloid curvature.
#         eps: Small float number to avoid division by zero.
#     Returns:
#         Tensor of same shape as `x`, giving Euclidean vectors in the tangent
#         space of the hyperboloid vertex.
#     """

#     # Calculate distance of vectors to the hyperboloid vertex.
#     rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
#     _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

#     rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
#     _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
#     return _output


# def lorentz_sq_mean(
#     x: Tensor,
#     weights: Tensor | None = None,
#     curv: float | Tensor = 1.0,
#     eps: float = 1e-6,) -> Tensor:
#     """
#     Weighted Lorentz squared-distance mean.
#     Args:
#         x: Spatial components, shape [N, D] or [B, N, D].
#         weights: Shape [N] or [B, N]. Default: all ones.
#         curv: Positive curvature magnitude.
#     Returns: Spatial mean, shape [D] or [B, D].
#     """
#     if x.ndim not in (2, 3):
#         raise ValueError("x must have shape [N,D] or [B,N,D]")
#     beta = 1.0 / curv
#     # Lorentz time components
#     x_time = torch.sqrt(beta + x.square().sum(dim=-1))  # [N] or [B,N]

#     if weights is None:
#         weights = torch.ones_like(x_time)
#     else:
#         weights = weights.to(device=x.device, dtype=x.dtype)
#         if x.ndim == 3 and weights.ndim == 1:
#             weights = weights.unsqueeze(0)
#     # Weighted Lorentz sum
#     s_time = (weights * x_time).sum(dim=-1)                  # [] or [B]
#     s_space = (weights.unsqueeze(-1) * x).sum(dim=-2)        # [D] or [B,D]
#     # Normalize onto <mu,mu>_L = -beta
#     norm = torch.sqrt(
#         torch.clamp(
#             s_time.square() - s_space.square().sum(dim=-1),
#             min=eps))
#     return torch.sqrt(torch.as_tensor(beta, device=x.device, dtype=x.dtype)) * \
#            s_space / norm.unsqueeze(-1)


# def half_aperture(
#     x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
# ) -> Tensor:
#     """
#     Compute the half aperture angle of the entailment cone formed by vectors on
#     the hyperboloid. The given vector would meet the apex of this cone, and the
#     cone itself extends outwards to infinity.

#     Args:
#         x: Tensor of shape `(B, D)` giving a batch of space components of
#             vectors on the hyperboloid.
#         curv: Positive scalar denoting negative hyperboloid curvature.
#         min_radius: Radius of a small neighborhood around vertex of the hyperboloid
#             where cone aperture is left undefined. Input vectors lying inside this
#             neighborhood (having smaller norm) will be projected on the boundary.
#         eps: Small float number to avoid numerical instability.

#     Returns:
#         Tensor of shape `(B, )` giving the half-aperture of entailment cones
#         formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
#     """

#     # Ensure numerical stability in arc-sin by clamping input.
#     asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
#     _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

#     return _half_aperture


# def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
#     """
#     Given two vectors `x` and `y` on the hyperboloid, compute the exterior
#     angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
#     of the hyperboloid.

#     This expression is derived using the Hyperbolic law of cosines.

#     Args:
#         x: Tensor of shape `(B, D)` giving a batch of space components of
#             vectors on the hyperboloid.
#         y: Tensor of same shape as `x` giving another batch of vectors.
#         curv: Positive scalar denoting negative hyperboloid curvature.

#     Returns:
#         Tensor of shape `(B, )` giving the required angle. Values of this
#         tensor lie in `(0, pi)`.
#     """

#     # Calculate time components of inputs (multiplied with `sqrt(curv)`):
#     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
#     y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

#     # Calculate lorentzian inner product multiplied with curvature. We do not use
#     # the `pairwise_inner` implementation to save some operations (since we only
#     # need the diagonal elements).
#     c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

#     # Make the numerator and denominator for input to arc-cosh, shape: (B, )
#     acos_numer = y_time + c_xyl * x_time
#     acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

#     acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
#     _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

#     return _angle


# def oxy_angle_modified(x: Tensor, y: Tensor, labels: Tensor, 
#                        curv: float | Tensor = 1.0, 
#                        min_radius: float = 0.1,  # value of K
#                        eps: float = 1e-8):
#     """
#     Return both the OXY angle and the Hyperbolic distance
#     x: image features (bs, h, w, d) [reversed notation of the MERU paper]
#     y: text_features (class_nums, d)
#     labels: labels (bs, h, w) # integers
#      use for SS
#     """
#     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
#     y_space = torch.sum(y**2, dim=-1, keepdim=True)
#     y_time = torch.sqrt(1 / curv + y_space)
#     xyl = x @ y.T - x_time @ y_time.T
#     c_xyl = curv * xyl
#     _distance = torch.acosh(torch.clamp(-c_xyl, min= 1/curv + eps))/ curv**0.5
    
#     labels_exp = labels.unsqueeze(-1)
#     # Gather along the last dimension (dim=3)
#     c_xyl = torch.gather(c_xyl, dim=3, index=labels_exp).squeeze(-1)
#     y_space = torch.sqrt(y_space.squeeze())[labels] # ||y_space||
#     y_time = y_time.squeeze()[labels]
#     x_time = x_time.squeeze()
    
#     # Make the numerator and denominator for input to arc-cosh, shape: (B, )
#     # changed x y according to input 
#     acos_numer = x_time + c_xyl * y_time
#     acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
#     acos_input = acos_numer / (y_space * acos_denom + eps)
#     _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    
#     # Ensure numerical stability in arc-sin by clamping input.
#     asin_input = 2 * min_radius / (y_space * curv**0.5 + eps)
#     _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    
#     return _angle, _distance, _half_aperture

# def oxy_angle_dist(x: Tensor, y: Tensor, labels: Tensor,
#                        curv: float | Tensor = 1.0,
#                        min_radius: float = 0.1,
#                        eps: float = 1e-6):
#     """
#     x: image features (B, D)
#     y: text features (class, D)
#     returns:
#         _angle:          (B)
#         _distance:       (B, class)
#         _half_aperture:  (B)
#     # use for embedding
#     """    
#     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # (B, 1)
#     y_space = torch.sum(y**2, dim=-1, keepdim=True)                         # (class, 1)
#     y_time  = torch.sqrt(1 / curv + y_space)                                # (class, 1)

#     xyl   = x @ y.T - x_time @ y_time.T                                     # (B, class)
#     c_xyl = curv * xyl                                                       # (B, class), ≤ -1

#     _distance = torch.acosh(torch.clamp(-c_xyl, min=1 + eps)) / curv**0.5  # (B, class)

#     # c_xyl   = c_xyl[torch.arange(x.shape[0]), labels]                       # (B,)
#     idx = torch.arange(x.shape[0], device=x.device)
#     c_xyl = c_xyl[idx, labels]
    
#     y_space = torch.sqrt(y_space.squeeze(-1))[labels]                        # (B,)
#     y_time  = y_time.squeeze(-1)[labels]                                     # (B,)
#     x_time  = x_time.squeeze(-1)                                             # (B,)

#     # FIX 1: clamp c_xyl away from -1 before using it in the denom.
#     # Without this, c_xyl²-1 ≈ 0 → denom collapses → ratio explodes silently.
#     c_xyl_safe = torch.clamp(c_xyl, max=-(1 + eps))                         # strictly < -1

#     acos_numer = x_time + c_xyl_safe * y_time
#     acos_denom = torch.sqrt(c_xyl_safe**2 - 1)                              # safe: arg ≥ eps²
#     acos_input = acos_numer / (y_space * acos_denom + eps)
#     _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

#     # FIX 2: guard y_space with a meaningful floor, not just eps.
#     # eps alone leaves the denom as eps*sqrt(c), making the ratio ~2r/1e-8 → saturated asin.
#     sqrt_c = torch.sqrt(torch.as_tensor(curv, device=y.device, dtype=y.dtype))
#     min_space = 2 * min_radius / sqrt_c
#     if torch.any(y_space <= min_space):
#         raise ValueError("Prototype lies inside the minimum radius: required for a valid entailment cone.")
    
#     y_space_safe = torch.clamp( y_space, min=min_space + eps)
#     # y_space_safe = torch.clamp(y_space, min=eps**0.5)                       # ~1e-4 floor
#     asin_input   = 2 * min_radius / (y_space_safe * sqrt_c)
#     _half_aperture = torch.asin(torch.clamp(asin_input,  max=1 - eps))

#     return _angle, _distance, _half_aperture


# # def oxy_angle_dist(x: Tensor, y: Tensor, labels: Tensor,
# #                        curv: float | Tensor = 1.0,
# #                        min_radius: float = 0.1,
# #                        eps: float = 1e-5):
# #     """
# #     x: image features (B, D)
# #     y: text features (class, D)
# #     returns:
# #         _angle:          (B)
# #         _distance:       (B, class)
# #         _half_aperture:  (B)
# #     # use for embedding
# #     """    
# #     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # (B, 1)
# #     y_space = torch.sum(y**2, dim=-1, keepdim=True)                         # (class, 1)
# #     y_time  = torch.sqrt(1 / curv + y_space)                                # (class, 1)

# #     xyl   = x @ y.T - x_time @ y_time.T                                     # (B, class)
# #     c_xyl = curv * xyl                                                       # (B, class), ≤ -1

# #     _distance = torch.acosh(torch.clamp(-c_xyl, min=1 + eps)) / curv**0.5  # (B, class)

# #     c_xyl   = c_xyl[torch.arange(x.shape[0]), labels]                       # (B,)
# #     y_space = torch.sqrt(y_space.squeeze(-1))[labels]                        # (B,)
# #     y_time  = y_time.squeeze(-1)[labels]                                     # (B,)
# #     x_time  = x_time.squeeze(-1)                                             # (B,)

# #     # FIX 1: clamp c_xyl away from -1 before using it in the denom.
# #     # Without this, c_xyl²-1 ≈ 0 → denom collapses → ratio explodes silently.
# #     c_xyl_safe = torch.clamp(c_xyl, max=-(1 + eps))                         # strictly < -1

# #     acos_numer = x_time + c_xyl_safe * y_time
# #     acos_denom = torch.sqrt(c_xyl_safe**2 - 1)                              # safe: arg ≥ eps²
# #     acos_input = acos_numer / (y_space * acos_denom + eps)
# #     _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

# #     # FIX 2: guard y_space with a meaningful floor, not just eps.
# #     # eps alone leaves the denom as eps*sqrt(c), making the ratio ~2r/1e-8 → saturated asin.
# #     y_space_safe = torch.clamp(y_space, min=eps**0.5)                       # ~1e-4 floor
# #     asin_input   = 2 * min_radius / (y_space_safe * curv**0.5)
# #     _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

# #     return _angle, _distance, _half_aperture

# def oxy_dist_angle_no_label(x: Tensor, y: Tensor,
#                        curv: float | Tensor = 1.0,
#                        min_radius: float = 0.1,
#                        eps: float = 1e-5):
#     """
#     x: image features (B, D)
#     y: text features (class, D)
#     returns:
#         _angle:          (B, class)
#         _distance:       (B, class)
#         _half_aperture:  (class,)
#     """
#     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # (B, 1)
#     y_space = torch.sum(y**2, dim=-1, keepdim=True)                         # (class, 1)
#     y_time  = torch.sqrt(1 / curv + y_space)                                # (class, 1)

#     c_xyl      = curv * (x @ y.T - x_time @ y_time.T)                      # (B, class)
#     c_xyl_safe = torch.clamp(c_xyl, max=-(1 + eps))                         # (B, class)

#     _distance = torch.acosh(torch.clamp(-c_xyl, min=1 + eps)) / curv**0.5  # (B, class)

#     y_space = torch.sqrt(y_space.squeeze(-1))                                # (class,)
#     y_time  = y_time.squeeze(-1)                                             # (class,)
#     x_time  = x_time.squeeze(-1)                                             # (B,)

#     acos_numer     = x_time.unsqueeze(1) + c_xyl_safe * y_time              # (B, class)
#     acos_denom     = torch.sqrt(c_xyl_safe**2 - 1)                          # (B, class)
#     acos_input     = acos_numer / (y_space * acos_denom + eps)              # (B, class)
#     _angle         = torch.acos(torch.clamp(acos_input, -1+eps, 1-eps))     # (B, class)

#     y_space_safe   = torch.clamp(y_space, min=eps**0.5)                     # (class,)
#     asin_input     = 2 * min_radius / (y_space_safe * curv**0.5)            # (class,)
#     _half_aperture = torch.asin(torch.clamp(asin_input, -1+eps, 1-eps))     # (class,)

#     return _angle, _distance, _half_aperture


# def oxy_angle_full(x: Tensor, y: Tensor,
#                        curv: float | Tensor = 1.0, 
#                        min_radius: float = 0.1,  # value of K
#                        eps: float = 1e-8):
#     """
#     Return both the OXY angle and the Hyperbolic distance
#     x: image features (bs, h, w, d) [reversed notation of the MERU paper]
#     y: text_features (class_nums, d)
#     labels: labels (bs, h, w) # integers
#     """
#     class_nums = y.shape[0]
    
#     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
#     y_space = torch.sum(y**2, dim=-1, keepdim=True)
#     y_time = torch.sqrt(1 / curv + y_space)
#     xyl = x @ y.T - x_time @ y_time.T
#     c_xyl = curv * xyl
    
#     y_time_exp = y_time.view(1, 1, 1, class_nums)  # expand to broadcast
#     acos_numer = x_time + c_xyl * y_time_exp

#     # denominator part
#     acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))  # [4,512,512,151]

#     y_space_exp = torch.sqrt(y_space).view(1,1,1, class_nums)  # sqrt and broadcast
#     # acos input
#     denom = y_space_exp * acos_denom + eps

#     acos_input = acos_numer / denom

#     # safe acos
#     _angle = torch.acos(torch.clamp(acos_input, min=-1+eps, max=1-eps))
    
    
#     return _angle


# # the following can be used in lieu of oxy_angle_full without any issues
# def oxy_angle_full_gnr(x: Tensor, y: Tensor,
#                        curv: float | Tensor = 1.0, 
#                        min_radius: float = 0.1,  # value of K
#                        eps: float = 1e-6):
#     """
#     Computes OXY angle between feature embeddings x and class embeddings y in hyperbolic space.
    
#     Works for both:
#       - x: [B, N, D]   (e.g., query embeddings)
#       - x: [B, H, W, D] (e.g., spatial features)
#     y: [C, D] class embeddings
#     Returns:
#       angle: [B, N, C] or [B, H, W, C] matching input dimensionality
#     """
#     class_nums = y.shape[0]
    
#     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
#     y_space = torch.sum(y**2, dim=-1, keepdim=True)
#     y_time = torch.sqrt(1 / curv + y_space)
#     xyl = x @ y.T - x_time @ y_time.T
#     c_xyl = curv * xyl
    
    
#     if x.ndim == 3:  # [B, N, D]
#         expand_shape = (1, 1, class_nums)
#     else:  # [B, H, W, D]
#         expand_shape = (1, 1, 1, class_nums)
        
#     y_time_exp = y_time.view(expand_shape)  # expand to broadcast
#     y_space_exp = torch.sqrt(y_space).view(expand_shape)  # sqrt and broadcast
    
#     # Numerator and denominator for the OXY angle formula
#     acos_numer = x_time + c_xyl * y_time_exp
#     # denominator part
#     acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))  # [4,512,512,151]
#     denom = y_space_exp * acos_denom + eps
#     # acos input
#     acos_input = acos_numer / denom
#     # safe acos
#     _angle = torch.acos(torch.clamp(acos_input, min=-1+eps, max=1-eps))
#     return _angle


# # the following computes the angle correpsonding to the mask prototypes
# def oxy_angle_mask(x: Tensor, y: Tensor,
#                    curv: float | Tensor = 1.0, 
#                    min_radius: float = 0.1,
#                    eps: float = 1e-6) -> Tensor:
#     """
#     Compute OXY angle between batched feature maps and queries along channel axis.

#     Args:
#         x: Tensor of shape (B, H, W, C)
#         y: Tensor of shape (B, Q, C)
#         curv: curvature scalar or tensor
#         min_radius: minimum radius (unused, placeholder)
#         eps: numerical stability

#     Returns:
#         Tensor of shape (B, H, W, Q)
#     """

#     B, H, W, C = x.shape
#     _, Q, _ = y.shape

#     # Compute Lorentz time components
#     x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1, keepdim=True))  # (B,H,W,1)
#     y_space = torch.sum(y ** 2, dim=-1, keepdim=True)  # (B,Q,1)
#     y_time = torch.sqrt(1 / curv + y_space)  # (B,Q,1)

#     # Inner products (Lorentz spatial term)
#     # x: (B,H,W,C), y: (B,Q,C)
#     # => x @ y^T -> (B,H,W,Q)
#     # Inner products (Lorentz spatial term)
#     xyl_spatial = torch.einsum('bhwc,bqc->bhwq', x, y)  # (B,H,W,Q)

#     # Time product term (broadcast multiply)
#     xyl_time = x_time * y_time.view(B, 1, 1, Q)  # (B,H,W,Q)

#     xyl = xyl_spatial - xyl_time
    
#     c_xyl = curv * xyl
    
#     # distance operation
#     _distance = torch.acosh(torch.clamp(-c_xyl, min=1 + eps))/ curv**0.5
        
#     # Expand terms for broadcast
#     y_time_exp = y_time.view(B, 1, 1, Q)
#     acos_numer = x_time + c_xyl * y_time_exp  # (B,H,W,Q)

#     # denominator part
#     acos_denom = torch.sqrt(torch.clamp(c_xyl ** 2 - 1, min=eps))  # (B,H,W,Q)
#     y_space_exp = torch.sqrt(y_space).view(B, 1, 1, Q)

#     denom = y_space_exp * acos_denom + eps
#     acos_input = acos_numer / denom

#     # Safe acos
#     _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

#     return _angle, _distance
