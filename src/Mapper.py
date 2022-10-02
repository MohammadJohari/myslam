import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_samples, random_select, matrix_to_pose6d, pose6d_to_matrix)
from src.utils.datasets import get_dataset, SeqSampler
from src.utils.Visualizer import Visualizer
from src.tools.full_cull_mesh import cull_mesh

import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.optim import Adam


from tqdm import tqdm

class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam, coarse_mapper=False, aux_mapper=False
                 ):

        self.stage = None
        self.cfg = cfg
        self.args = args
        self.coarse_mapper = coarse_mapper
        self.aux_mapper = aux_mapper

        self.idx = slam.idx
        self.nice = slam.nice
        self.truncation = slam.truncation
        self.bound = slam.bound
        self.logger = slam.logger
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders

        self.planes_xy = slam.shared_planes_xy
        self.planes_xz = slam.shared_planes_xz
        self.planes_yz = slam.shared_planes_yz

        self.c_planes_xy = slam.shared_c_planes_xy
        self.c_planes_xz = slam.shared_c_planes_xz
        self.c_planes_yz = slam.shared_c_planes_yz

        self.estimate_c2w_list = slam.estimate_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.device = cfg['mapping']['device']
        self.fix_fine = cfg['mapping']['fix_fine']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.BA = False  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.fix_color = cfg['mapping']['fix_color']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.num_joint_iters = cfg['mapping']['iters']
        self.clean_mesh = cfg['meshing']['clean_mesh']
        self.every_frame = cfg['mapping']['every_frame']
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio']
        self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio']
        self.mesh_coarse_level = cfg['meshing']['mesh_coarse_level']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        if self.nice:
            if coarse_mapper:
                self.keyframe_selection_method = 'global'

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=False, prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        ################################################
        # ray_dataset = get_ray_dataset(cfg, args, self.scale, device=self.device)
        # from tqdm import tqdm
        # loader = DataLoader(ray_dataset, batch_size=1, num_workers=1, pin_memory=False, prefetch_factor=2)
        # xx = iter(loader)
        # for i in tqdm(range(10000000)):
        #     _, gt_color, gt_depth, gt_c2w = self.frame_reader[i]
        #     # xxx = next(xx)
        #     # time.sleep(0.1)
        #
        # # _ = next(xx)

        ################################################

        if 'Demo' not in self.output:  # disable this visualization in demo
            self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                         vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                         truncation=self.truncation, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    # def sdf_loss(self, sdf, z_vals, gt_depth):
    #     front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
    #     back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
    #     sdf_mask = (~front_mask) * (~back_mask)

    #     fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
    #     # fs_loss = torch.mean(torch.abs(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        
    #     # side = 1 + 2 * (z_vals < gt_depth[:, None]).float()
    #     side = 1
    #     sdf_loss = torch.mean(torch.square((z_vals + sdf * side * self.truncation)[sdf_mask] - gt_depth[:, None].expand(z_vals.shape)[sdf_mask]))
    #     # sdf_loss = torch.mean(torch.abs((z_vals + sdf * self.truncation)[sdf_mask] - gt_depth[:, None].expand(z_vals.shape)[sdf_mask]))

    #     return 10 * fs_loss + 100 * sdf_loss

    def sdf_loss(self, sdf, z_vals, gt_depth):
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()        
        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                        (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        
        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square((z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square((z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))
       
        # back_loss = torch.mean(torch.square(sdf[back_mask] + torch.ones_like(sdf[back_mask])))
        # back_loss = torch.mean((sdf[:, :-1] - sdf[:, 1:])[back_mask[:, :-1]] ** 2)
        # new_loss = torch.mean(torch.exp(-10 * sdf[back_mask].abs()))

        # sdf_back = sdf[back_mask]
        # back_loss = torch.mean(torch.minimum(torch.square(sdf_back + torch.ones_like(sdf_back)), torch.square(sdf_back - torch.ones_like(sdf_back))))

        # return 5 * fs_loss + 200 * center_loss + 10 * tail_loss + 0.01 * back_loss + 0.001 * new_loss
        # return 5 * fs_loss + 200 * center_loss + 10 * tail_loss + 0.001 * back_loss
        return 5 * fs_loss + 200 * center_loss + 10 * tail_loss

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        scale = 1
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2] // scale),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1] // scale),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0] // scale))

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        if key == 'grid_coarse':
            mask = np.ones(val_shape[::-1]).astype(np.bool)
            return mask
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = 0
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak-ray_o
        dist = torch.sum(dist*dist, axis=1)
        mask2 = dist < 0.5*0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]

        mask = mask.reshape(val_shape[2] // scale, val_shape[1] // scale, val_shape[0] // scale)
        mask = F.interpolate(torch.from_numpy(mask).float()[None][None], size=[val_shape[2], val_shape[1], val_shape[0]],
                mode='nearest', align_corners=None, antialias=False)[0, 0].numpy() > 0

        return mask

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=16, pixels=100):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w, wandb_q):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device
        bottom = torch.tensor([0, 0, 0, 1.], device=device).reshape(1, 4)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                num = self.mapping_window_size-2
                optimize_frame = random_select(len(self.keyframe_dict)-1, num)
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
            optimize_frame = sorted(optimize_frame)
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                else:
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        decoders_para_list = []
        decoders_para_list += list(self.decoders.parameters())

        planes_para = []
        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = Variable(plane, requires_grad=True)
                planes_para.append(plane)
                planes[i] = plane

        c_planes_para = []
        for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
            for i, c_plane in enumerate(c_planes):
                c_plane = Variable(c_plane, requires_grad=True)
                c_planes_para.append(c_plane)
                c_planes[i] = c_plane

        gt_depths = []
        gt_colors = []
        c2ws = []
        gt_c2ws = []
        for frame in optimize_frame:
            # the oldest frame should be fixed to avoid drifting
            if frame != oldest_frame:
                if frame != -1:
                    gt_depths.append(keyframe_dict[frame]['depth'])
                    gt_colors.append(keyframe_dict[frame]['color'])
                    c2ws.append(keyframe_dict[frame]['est_c2w'])
                    gt_c2ws.append(keyframe_dict[frame]['gt_c2w'])
                else:
                    gt_depths.append(cur_gt_depth)
                    gt_colors.append(cur_gt_color)
                    c2ws.append(cur_c2w)
                    gt_c2ws.append(gt_cur_c2w)
        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        gt_c2ws = torch.stack(gt_c2ws, dim=0)

        if self.BA:
            pose6ds = Variable(matrix_to_pose6d(c2ws[-1:]), requires_grad=True)
            # gt_pose6ds = matrix_to_pose6d(gt_c2ws)

            # The corresponding lr will be set according to which stage the optimization is in
            optimizer = Adam([{'params': decoders_para_list, 'lr': 0},
                              {'params': planes_para, 'lr': 0},
                              {'params': c_planes_para, 'lr': 0},
                              {'params': [pose6ds], 'lr': 0, 'betas':(0.5, 0.999)}])
        else:
            optimizer = Adam([{'params': decoders_para_list, 'lr': 0},
                              {'params': planes_para, 'lr': 0},
                              {'params': c_planes_para, 'lr': 0}])

        for joint_iter in range(num_joint_iters):
            self.stage = 'color'

            optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr']*lr_factor
            optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['fine_lr']*lr_factor
            optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['color_lr']*lr_factor
            if self.BA:
                if self.stage == 'color':
                    # optimizer.param_groups[2]['lr'] = self.BA_cam_lr
                    optimizer.param_groups[3]['lr'] = 0.001
                    # if joint_iter >= 9:
                    #     optimizer.param_groups[2]['lr'] = 0.001

            if (not (idx == 0 and self.no_vis_on_first_frame)) and ('Demo' not in self.output) and (not self.aux_mapper):
                self.visualizer.vis(
                    idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes, self.decoders, wandb_q)

            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.BA:
                # c2ws[1:] = pose6d_to_matrix(pose6ds)
                c2ws_ = torch.cat([c2ws[0:-1], pose6d_to_matrix(pose6ds)], dim=0)
                # c2ws = pose6d_to_matrix(pose6ds)
                # c2ws[:-1] = c2ws[:-1].detach()
            else:
                c2ws_ = c2ws

            # c2ws = []
            # camera_tensor_id = 0
            # for frame in optimize_frame:
            #     if frame != -1:
            #         if self.BA and frame != oldest_frame:
            #             pose6d = pose6ds[camera_tensor_id]
            #             camera_tensor_id += 1
            #             c2w = get_camera_from_tensor(camera_tensor)
            #         else:
            #             c2w = keyframe_dict[frame]['est_c2w']
            #
            #     else:
            #         if self.BA:
            #             camera_tensor = camera_tensor_list[camera_tensor_id]
            #             c2w = get_camera_from_tensor(camera_tensor)
            #         else:
            #             c2w = cur_c2w
            #
            #     c2ws.append(c2w)
            # c2ws = torch.stack(c2ws, dim=0)

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, self.device)



            # ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
            #                                      batch_rays_o, device, self.stage, self.truncation,
            #                                      gt_depth=None if self.coarse_mapper else batch_gt_depth)
            # depth, uncertainty, color, entr, sdf, z_vals = ret


            ret = self.renderer.no_render_batch_ray(all_planes, self.decoders, batch_rays_d,
                                                    batch_rays_o, device, self.stage, self.truncation,
                                                    gt_depth=None if self.coarse_mapper else batch_gt_depth)
            depth, color, sdf, z_vals = ret

            depth_mask = (batch_gt_depth > 0)
            loss = self.sdf_loss(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

            ## Sepehr Color loss
            if ((not self.nice) or (self.stage == 'color')):
                color_loss = torch.square(batch_gt_color - color).mean()
                weighted_color_loss = self.w_color_loss * color_loss
                loss += weighted_color_loss

            # regulation_loss = entr[~depth_mask].mean()
            # regulation_loss = entr.mean()
            # # print('loss: ', loss.item(), 'reg loss: ', regulation_loss.item())
            # loss += 0.5 * regulation_loss


            # if idx == self.n_img - 1:
            #     print('stage: ', self.stage)
            #     for key, val in c.items():
            #         if (self.coarse_mapper and 'coarse' in key) or \
            #                 ((not self.coarse_mapper) and ('coarse' not in key) and ('color' not in key)):
            #             print(key, val.shape)
            #             tv1 = torch.pow((val[..., 1:] - val[...,:-1]),2).sum()
            #             tv2 = torch.pow((val[..., 1:, :] - val[...,:-1, :]),2).sum()
            #             tv3 = torch.pow((val[..., 1:, :, :] - val[...,:-1, :, :]),2).sum()
            #             tv_loss = tv1 + tv2 + tv3
            #             print('tv loss: ', tv_loss.item())

            # regulation_loss = entr.mean()
            # print('loss: ', loss.item(), 'reg loss: ', regulation_loss.item())
            # loss += 5 * regulation_loss

            # for key, val in c.items():
            #     if (self.coarse_mapper and 'coarse' in key) or \
            #             ((not self.coarse_mapper) and ('coarse' not in key) and ('color' not in key)):
            #         # print(key, val.shape)
            #         tv1 = torch.pow((val[..., 1:] - val[...,:-1]),2).mean()
            #         tv2 = torch.pow((val[..., 1:, :] - val[...,:-1, :]),2).mean()
            #         tv3 = torch.pow((val[..., 1:, :, :] - val[...,:-1, :, :]),2).mean()
            #         tv_loss = tv1 + tv2 + tv3
            #         # print('loss: ', loss, 'tv loss: ', tv_loss.item())
            #         loss += 0.01 * tv_loss

            # for key in ['grid_middle', 'grid_fine']:
            #     tv1 = (torch.pow((c[key][..., 1:] - c[key][...,:-1]),2) * self.c_mask[key][...,:-1]).sum() / (self.c_mask[key][...,:-1].sum() + 1e-6)
            #     tv2 = (torch.pow((c[key][..., 1:, :] - c[key][...,:-1, :]),2) * self.c_mask[key][...,:-1, :]).sum() / (self.c_mask[key][...,:-1, :].sum() + 1e-6)
            #     tv3 = (torch.pow((c[key][..., 1:, :, :] - c[key][...,:-1, :, :]),2) * self.c_mask[key][...,:-1, :, :]).sum() / (self.c_mask[key][...,:-1, :, :].sum() + 1e-6)
            #     tv_loss = tv1 + tv2 + tv3
            #     # print('loss: ', loss, 'tv loss: ', tv_loss.item())
            #     loss += 0.01 * tv_loss

            # pooling = {'grid_middle': torch.nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            #            'grid_fine': torch.nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False)
            #            }
            # for key in ['grid_middle', 'grid_fine']:
            #     # for key in ['grid_middle']:
            #     tv_loss = (torch.pow(c[key] - pooling[key](c[key].detach()), 2) * self.c_mask[key]).sum() / (self.c_mask[key].sum() + 1e-6)
            #
            #     # p3d = (1, 1, 1, 1, 1, 1)
            #     # c_padded = F.pad(c[key].detach(), p3d, "constant", -1)
            #     # # mask_padded = F.pad(self.c_mask[key], p3d, "constant", 0)
            #     # tv_loss = (torch.pow(c[key] - pooling[key](c_padded), 2) * self.c_mask[key]).sum() / (self.c_mask[key].sum() + 1e-6)
            #
            #     loss += 0.5 * tv_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

        if self.BA:
            # put the updated camera poses back
            optimized_c2ws = pose6d_to_matrix(pose6ds.detach())

            camera_tensor_id = 0
            for frame in optimize_frame[-1:]:
                if frame != -1:
                    keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]

            return cur_c2w
        else:
            return None

    def run(self, wandb_q):
        cfg = self.cfg
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        data_iter = iter(self.frame_loader)

        self.estimate_c2w_list[0] = gt_c2w

        init = True
        prev_idx = -1
        while (1):
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1:
                    break
                
                if self.sync_method == 'strict':
                    if idx % self.every_frame == 0 and idx != prev_idx:
                        break

                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx+self.every_frame//2:
                        break
                elif self.sync_method == 'free':
                    break
                
                time.sleep(0.001)
            prev_idx = idx

            start_time = time.time()

            if self.verbose:
                print(Fore.GREEN)
                prefix = 'Coarse ' if self.coarse_mapper else ''
                prefix = 'Auxiliary ' if self.aux_mapper else ''
                print(prefix+"Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            # _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx]
            _, gt_color, gt_depth, gt_c2w = next(data_iter)
            gt_color, gt_depth, gt_c2w = gt_color.squeeze(0), gt_depth.squeeze(0), gt_c2w.squeeze(0)
            if not init:
                lr_factor = cfg['mapping']['lr_factor']
                num_joint_iters = cfg['mapping']['iters']

                # here provides a color refinement postprocess
                if idx == self.n_img-1 and self.color_refine and not self.coarse_mapper:
                    outer_joint_iters = 10
                    self.mapping_window_size *= 4
                    self.middle_iter_ratio = 0.0
                    self.fine_iter_ratio = 0.0
                    num_joint_iters *= 5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else:
                    if self.nice:
                        outer_joint_iters = 1
                    else:
                        outer_joint_iters = 3

            else:
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = cfg['mapping']['iters_first']

            cur_c2w = self.estimate_c2w_list[idx]
            num_joint_iters = num_joint_iters//outer_joint_iters
            for outer_joint_iter in range(outer_joint_iters):

                self.BA = (len(self.keyframe_list) > 4) and cfg['mapping']['BA'] and (
                    not self.coarse_mapper) and (not self.aux_mapper)

                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w, wandb_q=wandb_q)

                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters-1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) \
                            and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        # self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                        # ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})
                        self.keyframe_dict.append({'gt_c2w': gt_c2w, 'idx': idx, 'color': gt_color,
                        'depth': gt_depth, 'est_c2w': cur_c2w.clone()})

            if self.verbose:
                prefix = 'Auxiliary' if self.aux_mapper else ''
                print("---%s Mapping Time: %s seconds ---" % (prefix, (time.time() - start_time)))

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            init = False

            if (not self.coarse_mapper) and (not self.aux_mapper):
                # mapping of first frame is done, can begin tracking
                self.mapping_first_frame[0] = 1

                if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) \
                        or idx == self.n_img-1:
                    self.logger.log(idx, self.keyframe_dict, self.keyframe_list,
                                    selected_keyframes=self.selected_keyframes
                                    if self.save_selected_keyframes_info else None)

                self.mapping_idx[0] = idx
                self.mapping_cnt[0] += 1

                if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)

                if idx == self.n_img-1:
                    # mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                    # self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                    #                      idx,  self.device, show_forecast=self.mesh_coarse_level,
                    #                      clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)
                    # os.system(
                    #     f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                    
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, self.device, show_forecast=False,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True)

                        cull_mesh(mesh_out_file, self.cfg, self.args, self.device)

                        # from zipfile import ZipFile
                        # import zipfile
                        # zip_file = f'{self.output}/mesh/final_mesh_eval_rec.zip'
                        # with ZipFile(zip_file, 'w') as zipf:
                        #     zipf.write(mesh_out_file, compression=zipfile.ZIP_DEFLATED)

                    break

            if idx == self.n_img-1:
                break
