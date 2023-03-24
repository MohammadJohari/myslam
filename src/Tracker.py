# ESLAM is a A NeRF-based SLAM system.
# It utilizes Neural Radiance Fields (NeRF) to perform Simultaneous
# Localization and Mapping (SLAM) in real-time. This system uses neural
# rendering techniques to create a 3D map of an environment from a
# sequence of images and estimates the camera pose simultaneously.
#
# Apache License 2.0
#
# Copyright (c) 2023 ams-OSRAM AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (matrix_to_pose6d, pose6d_to_matrix, get_samples)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']

        self.idx = slam.idx
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.truncation = slam.truncation

        self.shared_planes_xy = slam.shared_planes_xy
        self.shared_planes_xz = slam.shared_planes_xz
        self.shared_planes_yz = slam.shared_planes_yz

        self.shared_c_planes_xy = slam.shared_c_planes_xy
        self.shared_c_planes_xz = slam.shared_c_planes_xz
        self.shared_c_planes_yz = slam.shared_c_planes_yz

        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=2)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, truncation=self.truncation, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

        self.decoders = copy.deepcopy(self.shared_decoders)

        self.planes_xy = copy.deepcopy(self.shared_planes_xy)
        self.planes_xz = copy.deepcopy(self.shared_planes_xz)
        self.planes_yz = copy.deepcopy(self.shared_planes_yz)

        self.c_planes_xy = copy.deepcopy(self.shared_c_planes_xy)
        self.c_planes_xz = copy.deepcopy(self.shared_c_planes_xz)
        self.c_planes_yz = copy.deepcopy(self.shared_c_planes_yz)

        self.depth_map_err = 0

        for p in self.decoders.parameters():
            p.requires_grad_(False)

    def sdf_loss(self, sdf, z_vals, gt_depth):
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()        
        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) * 
                        (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square((z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square((z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))
       
        return 10 * fs_loss + 200 * center_loss + 50 * tail_loss

    def optimize_cam_in_batch(self, pose6d, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = pose6d_to_matrix(pose6d)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(
                device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth
            inside_mask = inside_mask & (batch_gt_depth > 0)

        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]

        ret = self.renderer.render_batch_ray(all_planes,
                                             self.decoders, batch_rays_d,
                                             batch_rays_o, self.device, self.truncation,
                                             gt_depth=batch_gt_depth)
        depth, color, sdf, z_vals = ret

        depth_diff = (batch_gt_depth - depth).abs()

        self.depth_map_err = 0.9 * self.depth_map_err + 0.1 * depth_diff[depth_diff < 0.5].detach().mean()

        diff_median = depth_diff.median()
        good_mask = (depth_diff < 10 * diff_median)

        loss = self.sdf_loss(sdf[good_mask], z_vals[good_mask], batch_gt_depth[good_mask])

        color_loss = torch.square(batch_gt_color - color)[good_mask].mean()
        loss += self.w_color_loss * color_loss

        ### Depth loss
        loss = loss + 1 * torch.square(batch_gt_depth[good_mask] - depth[good_mask]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')

            self.decoders.load_state_dict(self.shared_decoders.state_dict())

            for planes, self_planes in zip(
                    [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
                    [self.planes_xy, self.planes_xz, self.planes_yz]):
                for i, plane in enumerate(planes):
                    self_planes[i] = plane.detach()

            for c_planes, self_c_planes in zip(
                    [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
                    [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]):
                for i, c_plane in enumerate(c_planes):
                    self_c_planes[i] = c_plane.detach()

            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        device = self.device
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            gt_color, gt_depth, gt_c2w = gt_color.to(device, non_blocking=True), gt_depth.to(device, non_blocking=True), gt_c2w.to(device, non_blocking=True)
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            # initiate mapping every self.every_frame frames
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx-1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx-1].unsqueeze(0).to(device)
                        
            start_time = time.time()
            
            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)

            else:
                # gt_pose6d = matrix_to_pose6d(gt_c2w)

                if self.const_speed_assumption and idx-2 >= 0:
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_pose6d(pre_poses)
                    pose6d = 2 * pre_poses[1:] - pre_poses[0:1]
                else:
                    pose6d = matrix_to_pose6d(pre_c2w)

                # T = pose6d[:, -3:]
                # R = pose6d[:,:4]
                # T = Variable(pose6d[:, -3:].clone(), requires_grad=True)
                # R = Variable(pose6d[:,:4].clone(), requires_grad=True)
                T = torch.nn.Parameter(pose6d[:, -3:].clone())
                R = torch.nn.Parameter(pose6d[:,:4].clone())
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T},
                                                     {'params': cam_para_list_R, 'lr': self.cam_lr_R}])
                # breakpoint()
                # pose6d = Variable(pose6d, requires_grad=True)
                # cam_para_list = [pose6d]
                # optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr_T)

                candidate_cam_pose6d = None
                current_min_loss = 10000000000.
                for cam_iter in range(self.num_cam_iters):
                    pose6d = torch.cat([R, T], -1)

                    self.visualizer.vis(idx, cam_iter, gt_depth, gt_color, pose6d, all_planes, self.decoders)

                    loss = self.optimize_cam_in_batch(pose6d, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose6d = pose6d.clone().detach()

                c2w = pose6d_to_matrix(candidate_cam_pose6d)

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            self.idx[0] = idx

            if self.verbose:
                print("---Tracking Time: %s seconds ---" % (time.time() - start_time))