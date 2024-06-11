# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Check PYTORCH3D_LICENCE before use

import functools
from typing import Optional
import torch
import torch.nn.functional as F

"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre-multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""

def quaternionToMatrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: Quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    twoS = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - twoS * (j * j + k * k),
            twoS * (i * j - k * r),
            twoS * (i * k + j * r),
            twoS * (i * j + k * r),
            1 - twoS * (i * i + k * k),
            twoS * (j * k - i * r),
            twoS * (i * k - j * r),
            twoS * (j * k + i * r),
            1 - twoS * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the corresponding element of a, with sign taken from the corresponding element of b.
    This is like the standard copysign floating-point operation but is not careful about negative 0 and NaN.

    Args:
        a: Source tensor.
        b: Tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signsDiffer = (a < 0) != (b < 0)
    return torch.where(signsDiffer, -a, a)

def sqrtPositivePart(x):
    """
    Returns torch.sqrt(torch.max(0, x)) but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positiveMask = x > 0
    ret[positiveMask] = torch.sqrt(x[positiveMask])
    return ret

def matrixToQuaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * sqrtPositivePart(1 + m00 + m11 + m22)
    x = 0.5 * sqrtPositivePart(1 + m00 - m11 - m22)
    y = 0.5 * sqrtPositivePart(1 - m00 + m11 - m22)
    z = 0.5 * sqrtPositivePart(1 - m00 - m11 + m22)
    o1 = copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def axisAngleRotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y" or "Z".
        angle: Any shape tensor of Euler angles in radians.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        rFlat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        rFlat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        rFlat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Invalid axis {axis}.")

    return torch.stack(rFlat, -1).reshape(angle.shape + (3, 3))

def eulerAnglesToMatrix(eulerAngles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        eulerAngles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if eulerAngles.dim() == 0 or eulerAngles.shape[-1] != 3:
        raise ValueError("Invalid input Euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(axisAngleRotation, convention, torch.unbind(eulerAngles, -1))
    return functools.reduce(torch.matmul, matrices)

def angleFromTan(axis: str, otherAxis: str, data, horizontal: bool, taitBryan: bool):
    """
    Extract the first or third Euler angle from the two members of the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y" or "Z" for the angle we are finding.
        otherAxis: Axis label "X" or "Y" or "Z" for the middle axis in the convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis, which means the relevant entries are in the same row of the rotation matrix. If not, they are in the same column.
        taitBryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler angles in radians for each matrix in data as a tensor of shape (...).
    """
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + otherAxis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if taitBryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def indexFromLetter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2

def matrixToEulerAngles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = indexFromLetter(convention[0])
    i2 = indexFromLetter(convention[2])
    taitBryan = i0 != i2
    if taitBryan:
        centralAngle = torch.asin(matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        centralAngle = torch.acos(matrix[..., i0, i0])

    o = (
        angleFromTan(convention[0], convention[1], matrix[..., i2], False, taitBryan),
        centralAngle,
        angleFromTan(convention[2], convention[1], matrix[..., i0, :], True, taitBryan),
    )
    return torch.stack(o, -1)

def randomQuaternions(n: int, dtype: Optional[torch.dtype] = None, device=None, requiresGrad=False):
    """
    Generate random quaternions representing rotations, i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default: uses the current device for the default tensor type.
        requiresGrad: Whether the resulting tensor should have the gradient flag set.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn((n, 4), dtype=dtype, device=device, requires_grad=requiresGrad)
    s = (o * o).sum(1)
    o = o / copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o

def randomRotations(n: int, dtype: Optional[torch.dtype] = None, device=None, requiresGrad=False):
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None, uses the current device for the default tensor type.
        requiresGrad: Whether the resulting tensor should have the gradient flag set.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = randomQuaternions(n, dtype=dtype, device=device, requires_grad=requiresGrad)
    return quaternionToMatrix(quaternions)

def randomRotation(dtype: Optional[torch.dtype] = None, device=None, requiresGrad=False):
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return.
        device: Device of returned tensor. Default: if None, uses the current device for the default tensor type.
        requiresGrad: Whether the resulting tensor should have the gradient flag set.

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return randomRotations(1, dtype, device, requiresGrad)[0]

def standardizeQuaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real part is non-negative.

    Args:
        quaternions: Quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternionRawMultiply(a, b):
    """
    Multiply two quaternions. Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternionMultiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternionRawMultiply(a, b)
    return standardizeQuaternion(ab)

def quaternionInvert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quaternionApply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    realParts = point.new_zeros(point.shape[:-1] + (1,))
    pointAsQuaternion = torch.cat((realParts, point), -1)
    out = quaternionRawMultiply(
        quaternionRawMultiply(quaternion, pointAsQuaternion),
        quaternionInvert(quaternion),
    )
    return out[..., 1:]

def axisAngleToMatrix(axisAngle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axisAngle: Rotations given as a vector in axis angle form, as a tensor of shape (..., 3), where the magnitude is the angle turned anticlockwise in radians around the vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternionToMatrix(axisAngleToQuaternion(axisAngle))

def matrixToAxisAngle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor of shape (..., 3), where the magnitude is the angle turned anticlockwise in radians around the vector's direction.
    """
    return quaternionToAxisAngle(matrixToQuaternion(matrix))

def axisAngleToQuaternion(axisAngle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axisAngle: Rotations given as a vector in axis angle form, as a tensor of shape (..., 3), where the magnitude is the angle turned anticlockwise in radians around the vector's direction.

    Returns:
        Quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axisAngle, p=2, dim=-1, keepdim=True)
    halfAngles = 0.5 * angles
    eps = 1e-6
    smallAngles = angles.abs() < eps
    sinHalfAnglesOverAngles = torch.empty_like(angles)
    sinHalfAnglesOverAngles[~smallAngles] = torch.sin(halfAngles[~smallAngles]) / angles[~smallAngles]
    sinHalfAnglesOverAngles[smallAngles] = 0.5 - (angles[smallAngles] * angles[smallAngles]) / 48
    quaternions = torch.cat(
        [torch.cos(halfAngles), axisAngle * sinHalfAnglesOverAngles], dim=-1
    )
    return quaternions

def quaternionToAxisAngle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: Quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor of shape (..., 3), where the magnitude is the angle turned anticlockwise in radians around the vector's direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    halfAngles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * halfAngles
    eps = 1e-6
    smallAngles = angles.abs() < eps
    sinHalfAnglesOverAngles = torch.empty_like(angles)
    sinHalfAnglesOverAngles[~smallAngles] = torch.sin(halfAngles[~smallAngles]) / angles[~smallAngles]
    sinHalfAnglesOverAngles[smallAngles] = 0.5 - (angles[smallAngles] * angles[smallAngles]) / 48
    return quaternions[..., 1:] / sinHalfAnglesOverAngles

def rotation6dToMatrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix using Gram--Schmidt orthogonalization per Section B of [1].
    
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        Batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrixToRotation6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1] by dropping the last row. Note that 6D representation is not unique.
    
    Args:
        matrix: Batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
