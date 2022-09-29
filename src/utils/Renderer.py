import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf
from tqdm import tqdm


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=10000):
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_importance = cfg['rendering']['N_importance']

        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        self.nice = slam.nice
        self.bound = slam.bound.to(cfg['mapping']['device'], non_blocking=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    # def eval_points(self, p, p_mask, planes_xy, planes_xz, planes_yz, decoders, c=None, c_mask=None, stage='color', device='cuda:0'):
    #     """
    #     Evaluates the occupancy and/or color value for the points.
    #
    #     Args:
    #         p (tensor, N*3): Point coordinates.
    #         decoders (nn.module decoders): Decoders.
    #         c (dicts, optional): Feature grids. Defaults to None.
    #         stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
    #         device (str, optional): CUDA device. Defaults to 'cuda:0'.
    #
    #     Returns:
    #         ret (tensor): occupancy (and color) value of input points.
    #     """
    #
    #     p_split = torch.split(p, self.points_batch_size)
    #     p_mask_split = torch.split(p_mask, self.points_batch_size)
    #     bound = self.bound
    #     rets = []
    #     for pi, pi_mask in zip(p_split, p_mask_split):
    #         # mask for points out of bound
    #         mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
    #         mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
    #         mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
    #         mask = mask_x & mask_y & mask_z
    #
    #         pi = pi.unsqueeze(0)
    #         pi_mask = pi_mask.unsqueeze(0)
    #         if self.nice:
    #             ret, _ = decoders(pi, pi_mask, c_grid=c, c_mask=c_mask, planes_xy=planes_xy, planes_xz=planes_xz, planes_yz=planes_yz, stage=stage)
    #         else:
    #             ret = decoders(pi, c_grid=None)
    #         ret = ret.squeeze(0)
    #         if len(ret.shape) == 1 and ret.shape[0] == 4:
    #             ret = ret.unsqueeze(0)
    #
    #         # ret[~mask, 3] = 100
    #         ret[~mask, 3] = -1
    #         rets.append(ret)
    #
    #     ret = torch.cat(rets, dim=0)
    #     return ret

    def eval_points(self, p, p_mask, all_planes, decoders, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """
        pi = p.unsqueeze(0)
        ret, sharpness = decoders(pi, all_planes=all_planes, stage=stage)
        ret = ret.squeeze(0)

        return ret, sharpness

    def perturbation(self, z_vals):
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def no_render_batch_ray(self, all_planes, decoders, rays_d, rays_o, device, stage, truncation, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """
        N_samples = self.N_samples
        N_surface = self.N_surface
        N_rays = rays_o.shape[0]
        z_vals = torch.empty([N_rays, N_samples + N_surface], device=device)
        near = 0.0

        if gt_depth is None:
            gt_mask = torch.zeros(N_rays, N_samples, device=device)
        else:
            gt_depth = gt_depth.reshape(-1, 1)
            gt_mask = (gt_depth > 0).squeeze()

            t_vals_surface = torch.linspace(0., 1., steps=N_surface, device=device)
            t_vals_free = torch.linspace(0., 1., steps=N_samples, device=device)

            gt_none_zero = gt_depth[gt_mask]

            gt_depth_surface = gt_none_zero.expand(-1, N_surface)
            z_vals_surface = gt_depth_surface - truncation  + 2 * truncation * t_vals_surface
            if self.perturb > 0.:
                z_vals_surface = self.perturbation(z_vals_surface)

            gt_depth_free = gt_none_zero.expand(-1, N_samples)
            z_vals_free = near + (gt_depth_free - truncation) * t_vals_free
            if self.perturb > 0.:
                z_vals_free = self.perturbation(z_vals_free)

            z_vals[gt_mask] = torch.cat([z_vals_free, z_vals_surface], dim=-1)

        with torch.no_grad():
            det_rays_o = rays_o[~gt_mask].detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d[~gt_mask].detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        t_vals_uni = torch.linspace(0., 1., steps=N_samples + N_surface, device=device)
        z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
        if self.perturb > 0.:
            z_vals_uni = self.perturbation(z_vals_uni)
        z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        pointsf = pts.reshape(-1, 3)

        raw, sharpness= self.eval_points(pointsf, torch.zeros_like(pointsf), all_planes, decoders, stage, device)
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)

        depth, uncertainty, color, weights, entr = raw2outputs_nerf_color(raw, sharpness, z_vals, rays_d, truncation, occupancy=self.occupancy, device=device)

        sdf = raw[..., -1]

        # if N_importance > 0:
        #     z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        #     z_samples = sample_pdf(
        #         z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
        #     z_samples = z_samples.detach()
        #     z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        #     pts = rays_o[..., None, :] + \
        #         rays_d[..., None, :] * z_vals[..., :, None]
        #     pts = pts.reshape(-1, 3)
        #     raw = self.eval_points(pts, decoders, c, stage, device)
        #     raw = raw.reshape(N_rays, N_samples+N_importance+N_surface, -1)

        #     depth, uncertainty, color, weights = raw2outputs_nerf_color(
        #         raw, z_vals, rays_d, truncation, occupancy=self.occupancy, device=device)
        #     return depth, uncertainty, color

        return depth, uncertainty, color, _, sdf, z_vals

    def render_img(self, all_planes, decoders, c2w, truncation, device, stage, gt_depth=None):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    # ret = self.render_batch_ray(
                    #     c, planes_xy, planes_xz, planes_yz, decoders, rays_d_batch, rays_o_batch, device, stage, truncation, gt_depth=None)
                    ret = self.no_render_batch_ray(all_planes, decoders, rays_d_batch,
                                                            rays_o_batch, device, stage, truncation,
                                                            gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    # ret = self.render_batch_ray(
                    #     c, planes_xy, planes_xz, planes_yz, decoders, rays_d_batch, rays_o_batch, device, stage, truncation, gt_depth=gt_depth_batch)
                    ret = self.no_render_batch_ray(all_planes, decoders, rays_d_batch,
                                                            rays_o_batch, device, stage, truncation,
                                                            gt_depth=gt_depth_batch)

                depth, uncertainty, color, _, _, _ = ret
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, uncertainty, color