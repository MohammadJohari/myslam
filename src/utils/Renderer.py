import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf, normalize_3d_coordinate
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
        N_importance = self.N_importance
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
            # z_vals_surface = gt_depth_surface - truncation  + 2 * truncation * t_vals_surface
            # if self.perturb > 0.:
            #     z_vals_surface = self.perturbation(z_vals_surface)
            z_vals_surface = gt_depth_surface - (1.5 * truncation)  + (3 * truncation * t_vals_surface)

            gt_depth_free = gt_none_zero.expand(-1, N_samples)
            # z_vals_free = near + (gt_depth_free - truncation) * t_vals_free
            # if self.perturb > 0.:
            #     z_vals_free = self.perturbation(z_vals_free)
            z_vals_free = near + 1.2 * gt_depth_free * t_vals_free

            # z_vals[gt_mask] = torch.cat([z_vals_free, z_vals_surface], dim=-1)
            z_vals[gt_mask], _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)

        with torch.no_grad():
            det_rays_o = rays_o[~gt_mask].detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d[~gt_mask].detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        t_vals_uni = torch.linspace(0., 1., steps=N_samples + N_surface, device=device)
        z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
        # if self.perturb > 0.:
        #     z_vals_uni = self.perturbation(z_vals_uni)
        z_vals[~gt_mask] = z_vals_uni

        if self.perturb > 0.:
            z_vals = self.perturbation(z_vals)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]

        # rendered_depth, rendered_rgb, sdf = decoders(pts, z_vals, all_planes)
        # return rendered_depth, rendered_rgb, sdf, z_vals

        pts_nor = normalize_3d_coordinate(pts.clone(), self.bound)
        sdf = decoders.get_raw_sdf2(pts_nor, all_planes)
        sdf = sdf.reshape(*pts.shape[0:2])
        weights = self.sdf2weights(sdf, z_vals, truncation)
        # alpha = self.sdf2alpha(raw[..., -1])
        # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
        #                                         , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        rgb = torch.zeros([pts_nor.shape[0], 3], device=device)
        nonzero_mask = weights.reshape(-1) > 0.0
        rgb_nonzero = decoders.get_raw_rgb2(pts_nor[nonzero_mask], all_planes)
        rgb[nonzero_mask] = rgb_nonzero
        rgb = rgb.reshape(*pts.shape[0:2], 3)
        raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)

        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1].detach(), N_importance, det=False, device=device)
            pts_importance = rays_o[..., None, :] + rays_d[..., None, :] * \
                  z_samples[..., :, None]
            raw_importance = decoders.get_only_raw(pts_importance, all_planes)

            z_vals, ind = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            raw = torch.cat([raw, raw_importance], 1)
            raw = torch.gather(raw, dim=1, index=ind.unsqueeze(-1).expand(-1, -1, 4))

            weights = self.sdf2weights(raw[..., -1], z_vals, truncation)
            # alpha = self.sdf2alpha(raw[..., -1])
            # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
            #                                     , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)

        return rendered_depth, rendered_rgb, raw[..., -1], z_vals

    def sdf2alpha(self, sdf, sharpness=20.0):
        return 1. - torch.exp(-sharpness * torch.sigmoid(-sdf * sharpness))
        # return 1. - torch.exp(- 20 * torch.sigmoid(-sdf * 10))
    def sdf2weights(self, sdf, z_vals, truncation, sharpness=10.0):
        weights = torch.sigmoid(sharpness * sdf) * torch.sigmoid(-sharpness * sdf)

        with torch.no_grad():
            # signs = sdf[:, 1:] * sdf[:, :-1]
            mask = torch.where(sdf < -0.01, torch.ones_like(sdf), torch.zeros_like(sdf))
            mask[:, -1] = 1
            # mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
            # mask = torch.cat([mask, torch.ones([mask.shape[0], 1], device=mask.device)], dim=1)
            inds = torch.argmax(mask, dim=1)
            inds = inds[..., None]
            z_min = torch.gather(z_vals, 1, inds) # The first surface
            mask = torch.where(z_vals < z_min + truncation, torch.ones_like(z_vals), torch.zeros_like(z_vals))
            # mask = mask * (weights > 0.0001)

        weights = weights * mask
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)

        return weights

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

                depth, color, _, _ = ret
                depth_list.append(depth.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, color