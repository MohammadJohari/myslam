# *****************************************************************
# This source code is only provided for the reviewing purpose of
# CVPR 2023. The source files should not be kept or used in any
# commercial or research products. Please delete all files after
# the reviewing period.
# *****************************************************************

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, quaternion_to_matrix


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    # weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    pdf = weights

    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

def get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i, device=device)], -1)
    dirs = dirs.unsqueeze(-2)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2ws[:, None, :3, :3], -1)
    rays_o = c2ws[:, None, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def select_uv(i, j, n, b, depths, colors, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n * b,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n * b)
    j = j[indices]  # (n * b)

    indices = indices.reshape(b, -1)
    i = i.reshape(b, -1)
    j = j.reshape(b, -1)

    depths = depths.reshape(b, -1)
    colors = colors.reshape(b, -1, 3)

    depths = torch.gather(depths, 1, indices)  # (b, n)
    colors = torch.gather(colors, 1, indices.unsqueeze(-1).expand(-1, -1, 3))  # (b, n, 3)

    return i, j, depths, colors

def get_sample_uv(H0, H1, W0, W1, n, b, depths, colors, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    depths = depths[:, H0:H1, W0:W1]
    colors = colors[:, H0:H1, W0:W1]

    i, j = torch.meshgrid(torch.linspace(W0, W1 - 1, W1 - W0, device=device), torch.linspace(H0, H1 - 1, H1 - H0, device=device))

    i = i.t()  # transpose
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, b, depths, colors, device=device)

    return i, j, depth, color

def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2ws, depths, colors, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    b = c2ws.shape[0]
    i, j, sample_depth, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, b, depths, colors, device=device)

    rays_o, rays_d = get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), sample_depth.reshape(-1), sample_color.reshape(-1, 3)

def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor

def axis_angle_to_matrix(data):
    batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3, device=data.device).expand(*batch_dims,3,3)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

def matrix_to_axis_angle(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

def matrix_to_pose6d(batch_matrices):
    return torch.cat([matrix_to_quaternion(batch_matrices[:,:3,:3]), batch_matrices[:,:3,3]], dim=-1)

def pose6d_to_matrix(batch_poses):
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = quaternion_to_matrix(batch_poses[:,:4])
    c2w[:,:3,3] = batch_poses[:,4:]
    return c2w

def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p
