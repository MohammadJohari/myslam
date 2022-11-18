# *****************************************************************
# This source code is only provided for the reviewing purpose of
# CVPR 2023. The source files should not be kept or used in any
# commercial or research products. Please delete all files after
# the reviewing period.
# *****************************************************************

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
from src.tools.cull_mesh import cull_mesh

import torch.nn.functional as F
from torch.utils.data import DataLoader

class Mapper(object):
    """
    Mapper thread.

    """

    def __init__(self, cfg, args, slam):

        self.cfg = cfg
        self.args = args

        self.idx = slam.idx
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

        self.device = cfg['mapping']['device']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.BA = False  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.num_joint_iters = cfg['mapping']['iters']
        self.every_frame = cfg['mapping']['every_frame']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}


        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=True, prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                     truncation=self.truncation, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy


    def sdf_loss(self, sdf, z_vals, gt_depth):
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()        
        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                        (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square((z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square((z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))
       
        return 5 * fs_loss + 200 * center_loss + 10 * tail_loss

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, k, N_samples=8, pixels=50):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
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
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w.unsqueeze(0), gt_depth.unsqueeze(0), gt_color.unsqueeze(0), self.device)

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
        pts = pts.reshape(1, -1, 3)

        w2cs = torch.inverse(self.rough_key_c2ws[:-2])
        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

        # selected_keyframes = torch.argsort(percent_inside, descending=True)[:k]

        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)
        rnd_inds = torch.randperm(selected_keyframes.shape[0])
        selected_keyframes = selected_keyframes[rnd_inds[:k]]

        selected_keyframes = list(selected_keyframes.cpu().numpy())

        return selected_keyframes

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w):
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
                num = self.mapping_window_size-3
                optimize_frame = random_select(len(self.keyframe_dict)-2, num)
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-1
                optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, num)

        # add the last two keyframes and the current frame(use -1 to denote)
        if len(keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(keyframe_list)-1] + [len(keyframe_list)-2]
            optimize_frame = sorted(optimize_frame)
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

        if self.BA:
            pose6ds = Variable(matrix_to_pose6d(c2ws[1:]), requires_grad=True)

            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                              {'params': planes_para, 'lr': 0},
                              {'params': c_planes_para, 'lr': 0},
                              {'params': [pose6ds], 'lr': 0}])

        else:
            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                          {'params': planes_para, 'lr': 0},
                                          {'params': c_planes_para, 'lr': 0}])

        for joint_iter in range(num_joint_iters):
            optimizer.param_groups[0]['lr'] = cfg['mapping']['lr']['decoders_lr'] * lr_factor
            optimizer.param_groups[1]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor
            optimizer.param_groups[2]['lr'] = cfg['mapping']['lr']['c_planes_lr'] * lr_factor

            if self.BA:
                optimizer.param_groups[3]['lr'] = self.BA_cam_lr

            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.vis(
                    idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes, self.decoders)

            if self.BA:
                c2ws_ = torch.cat([c2ws[0:1], pose6d_to_matrix(pose6ds)], dim=0)
            else:
                c2ws_ = c2ws

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, self.device)


            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(
                    device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

            ret = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d,
                                                 batch_rays_o, device, self.truncation,
                                                 gt_depth=batch_gt_depth)
            depth, color, sdf, z_vals = ret

            depth_mask = (batch_gt_depth > 0)
            loss = self.sdf_loss(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

            ## Color loss
            color_loss = torch.square(batch_gt_color - color).mean()
            weighted_color_loss = self.w_color_loss * color_loss
            loss += weighted_color_loss

            ### Depth loss
            loss = loss + 0.1 * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

        if self.BA:
            # put the updated camera poses back
            optimized_c2ws = pose6d_to_matrix(pose6ds.detach())

            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]

            return cur_c2w
        else:
            return None

    def run(self):
        cfg = self.cfg
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        data_iter = iter(self.frame_loader)

        self.estimate_c2w_list[0] = gt_c2w
        self.rough_key_c2ws = None

        init = True
        prev_idx = -1
        while (1):
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1:
                    break

                if idx % self.every_frame == 0 and idx != prev_idx:
                    break

                time.sleep(0.001)
            prev_idx = idx

            start_time = time.time()

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = next(data_iter)
            gt_color, gt_depth, gt_c2w = gt_color.squeeze(0).to(self.device, non_blocking=True), gt_depth.squeeze(0).to(self.device, non_blocking=True), gt_c2w.squeeze(0).to(self.device, non_blocking=True)
            if not init:
                lr_factor = cfg['mapping']['lr_factor']
                num_joint_iters = cfg['mapping']['iters']
                outer_joint_iters = 1

            else:
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = cfg['mapping']['iters_first']

            cur_c2w = self.estimate_c2w_list[idx]
            num_joint_iters = num_joint_iters//outer_joint_iters
            for outer_joint_iter in range(outer_joint_iters):

                self.BA = (len(self.keyframe_list) > 4) and cfg['mapping']['BA']

                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)

                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters-1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) \
                            and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        self.keyframe_dict.append({'gt_c2w': gt_c2w, 'idx': idx, 'color': gt_color,
                                                   'depth': gt_depth, 'est_c2w': cur_c2w.clone()})

                        if self.rough_key_c2ws is None:
                            self.rough_key_c2ws = cur_c2w.unsqueeze(0).clone()
                        else:
                            self.rough_key_c2ws = torch.cat([self.rough_key_c2ws, cur_c2w.unsqueeze(0).clone()], dim=0)

            if self.verbose:
                print("---Mapping Time: %s seconds ---" % (time.time() - start_time))

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            init = False
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
                self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device)

            if idx == self.n_img-1:
                if self.eval_rec:
                    mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                    self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device)

                    cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)

                break

            if idx == self.n_img-1:
                break