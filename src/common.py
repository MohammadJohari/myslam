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

    # u = u.contiguous()
    # try:
    #     # this should work fine with the provided environment.yaml
    #     inds = torch.searchsorted(cdf, u, right=True)
    # except:
    #     # for lower version torch that does not have torch.searchsorted,
    #     # you need to manually install from
    #     # https://github.com/aliutkus/torchsearchsorted
    #     from torchsearchsorted import searchsorted
    #     inds = searchsorted(cdf, u, side='right')
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


# def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
#     """
#     Get corresponding rays from input uv.
#
#     """
#     if isinstance(c2w, np.ndarray):
#         c2w = torch.from_numpy(c2w).to(device)
#
#     dirs = torch.stack(
#         [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
#     dirs = dirs.reshape(-1, 1, 3)
#     # Rotate ray directions from camera frame to the world frame
#     # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     rays_d = torch.sum(dirs * c2w[:3, :3], -1)
#     rays_o = c2w[:3, -1].expand(rays_d.shape)
#     return rays_o, rays_d

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

# def select_uv(i, j, n, depth, color, device='cuda:0'):
#     """
#     Select n uv from dense uv.
#
#     """
#     i = i.reshape(-1)
#     j = j.reshape(-1)
#     indices = torch.randint(i.shape[0], (n,), device=device)
#     indices = indices.clamp(0, i.shape[0])
#     i = i[indices]  # (n)
#     j = j[indices]  # (n)
#     depth = depth.reshape(-1)
#     color = color.reshape(-1, 3)
#     depth = depth[indices]  # (n)
#     color = color[indices]  # (n,3)
#     return i, j, depth, color

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

# def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
#     """
#     Sample n uv coordinates from an image region H0..H1, W0..W1
#
#     """
#     depth = depth[H0:H1, W0:W1]
#     color = color[H0:H1, W0:W1]
#
#     i, j = torch.meshgrid(torch.linspace(
#     W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
#
#     i = i.t()  # transpose
#     j = j.t()
#     i, j, depth, color = select_uv(i, j, n, depth, color, device=device)
#
#     return i, j, depth, color

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

# def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device):
#     """
#     Get n rays from the image region H0..H1, W0..W1.
#     c2w is its camera pose and depth/color is the corresponding image tensor.
#
#     """
#     i, j, sample_depth, sample_color = get_sample_uv(
#         H0, H1, W0, W1, n, depth, color, device=device)
#     rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
#     return rays_o, rays_d, sample_depth, sample_color

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

# def get_samples(loader, H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device):
#     """
#     Get n rays from the image region H0..H1, W0..W1.
#     c2w is its camera pose and depth/color is the corresponding image tensor.
#
#     """
#     # u = torch.randint(low=W0, high=W1, size=[n]).to(device)
#     # v = torch.randint(low=H0, high=H1, size=[n]).to(device)
#     #
#     # dirs = torch.stack([(u-cx)/fx, -(v-cy)/fy, -torch.ones_like(u)], -1)
#     # dirs = dirs.reshape(-1, 1, 3)
#
#     u, v, dirs = next(loader)
#
#     rays_d = torch.sum(dirs * c2w[:3, :3], -1)
#     rays_o = c2w[:3, -1].expand(rays_d.shape)
#
#     sample_depth = depth[v, u]
#     sample_color = color[v, u]
#
#     return rays_o, rays_d, sample_depth, sample_color

def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


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

# def pose6d_to_matrix(batch_poses):
#     c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
#     c2w[:,:3,:3] = axis_angle_to_matrix(batch_poses[:,:,0])
#     c2w[:,:3,3] = batch_poses[:,:,1]
#     return c2w
#
# def matrix_to_pose6d(batch_matrices):
#     return torch.cat([matrix_to_axis_angle(batch_matrices[:,:3,:3]).unsqueeze(-1),
#                       batch_matrices[:,:3,3:]], dim=-1)

def matrix_to_pose6d(batch_matrices):
    return torch.cat([matrix_to_quaternion(batch_matrices[:,:3,:3]), batch_matrices[:,:3,3]], dim=-1)

def pose6d_to_matrix(batch_poses):
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = quaternion_to_matrix(batch_poses[:,:4])
    c2w[:,:3,3] = batch_poses[:,4:]
    return c2w

def raw2outputs_nerf_color(raw, sharpness, z_vals, rays_d, truncation, occupancy=False, device='cuda:0'):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    def raw2alpha(raw, dists, act_fn=F.relu):
        # return 1. - torch.exp(-act_fn(raw)*dists)
        # return 1. - torch.exp(-act_fn(10 * raw))
        return 1. - torch.exp(-F.softplus(20 * raw))
        # return 1. - torch.exp(-F.softplus(raw) * dists * 10)

    def sdf2weights(sdf, truncation, z_vals, sharpness):
        # weights = torch.sigmoid(sdf / truncation) * torch.sigmoid(-sdf / truncation)
        weights = torch.sigmoid(sharpness * sdf) * torch.sigmoid(-sharpness * sdf)

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + truncation, torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)

    def sdf2alpha(sdf, sharpness):
        # return torch.sigmoid(-sdf * sharpness)
        return 1. - torch.exp(-sharpness * torch.sigmoid(-sdf * sharpness))

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
        device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]
    sdf_mode = True
    if not sdf_mode:
        if occupancy:
            raw[..., 3] = torch.sigmoid(10*raw[..., -1])
            alpha = raw[..., -1]
        else:
            # original nerf, volume density
            alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

        weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
            device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    else:
        weights = sdf2weights(raw[..., -1], truncation, z_vals, sharpness)  # (N_rays, N_samples)
        # alpha = sdf2alpha(raw[..., -1], sharpness)
        # weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        #     device).float(), (1. - alpha + 1e-10).float()], -1).float(), -1)[:, :-1]

    prob = weights / weights.sum(-1, keepdim=True)
    entr = (- torch.log(prob + 1e-12) * prob).sum(-1)
    
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)
    return depth_map, depth_var, rgb_map, weights, entr


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
