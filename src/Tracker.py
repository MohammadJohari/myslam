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

import wandb

class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
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

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
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

        for p in self.decoders.parameters():
            p.requires_grad_(False)

        self.optimizer_state = None

        # self.noise = torch.empty([1, 7]).normal_(mean=0, std=0.001).to(self.device)

    # def sdf_loss(self, sdf, z_vals, gt_depth):
    #     front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
    #     back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
    #     sdf_mask = (~front_mask) * (~back_mask)

    #     fs_loss = torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask]))
    #     sdf_loss = torch.square((z_vals + sdf * self.truncation)[sdf_mask] - gt_depth[:, None].expand(z_vals.shape)[sdf_mask])

    #     # tmp1 = fs_loss < 10 * fs_loss.median()
    #     # tmp2 = sdf_loss < 10 * sdf_loss.median()

    #     # loss = 10 * fs_loss[tmp1].mean() + 200 * sdf_loss[tmp2].mean()
        
    #     # loss = 10 * fs_loss.mean() + 200 * sdf_loss.mean()
    #     loss = 10 * fs_loss.mean() + 200 * sdf_loss.mean()

    #     return loss

    def sdf_loss(self, sdf, z_vals, gt_depth):
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()        
        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) * 
                        (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)), torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square((z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square((z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))
       
        # return 10 * fs_loss + 200 * center_loss + 1 * tail_loss
        return 10 * fs_loss + 200 * center_loss + 10 * tail_loss

    def optimize_cam_in_batch(self, iter, pose6d, gt_color, gt_depth, batch_size, optimizer, pre_pose6d=None, noise=0):
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

        # if not pre_pose6d is None:
        #     std = 0.02 * torch.norm(pose6d - pre_pose6d).item()
        #     noise = torch.empty_like(pose6d).normal_(mean=0, std=std)
        # else:
        #     noise = 0

        c2w = pose6d_to_matrix(pose6d + noise)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(
                    device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            # if inside_mask.float().mean() < (1.0 - 1e-6):
            #     breakpoint()
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        ret = self.renderer.no_render_batch_ray(all_planes,
                                                self.decoders, batch_rays_d,
                                                batch_rays_o, self.device, 'color', self.truncation,
                                                gt_depth=batch_gt_depth)
        depth, color, sdf, z_vals = ret

        # good_mask = torch.abs(batch_gt_depth - depth) < 0.10
        depth_mask = (batch_gt_depth > 0)
        depth_diff = (batch_gt_depth - depth).abs()
        diff_median = depth_diff[depth_mask].median()
        good_mask = (depth_diff < 10 * diff_median) & depth_mask

        if iter > -1:
            loss = self.sdf_loss(sdf[good_mask], z_vals[good_mask], batch_gt_depth[good_mask])

            if self.use_color_in_tracking:
                color_loss = torch.square(
                    batch_gt_color - color)[good_mask].mean()
                loss += self.w_color_loss * color_loss
        else:
            loss = 0

        ### Depth loss
        loss = loss + 0.1 * torch.square(batch_gt_depth[good_mask] - depth[good_mask]).mean()

        ## Super Stupid
        # if not pre_pose6d is None:
        #     loss = loss + 0.01 * torch.abs(pose6d - pre_pose6d).mean()

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

    def run(self, wandb_q):
        device = self.device
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

        wandb_dir = self.cfg['data']['output']
        wandb_name = wandb_dir.split('/')[-1]
        wandb.init(config=self.cfg, project='slam', name=wandb_name, dir=wandb_dir)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.001)
                    pre_c2w = self.estimate_c2w_list[idx-1].unsqueeze(0).to(device)
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.001)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass
                        
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
                        idx, 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders, wandb_q)

            else:
                gt_pose6d = matrix_to_pose6d(gt_c2w)
                # if self.const_speed_assumption and idx-2 >= 0:
                #     delta = pre_c2w.squeeze(0) @ self.estimate_c2w_list[idx-2].inverse()
                #     estimated_new_cam_c2w = delta.unsqueeze(0) @ pre_c2w
                # else:
                #     estimated_new_cam_c2w = pre_c2w
                # pose6d = matrix_to_pose6d(estimated_new_cam_c2w)

                if self.const_speed_assumption and idx-2 >= 0:
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_pose6d(pre_poses)
                    pose6d = 2 * pre_poses[1:] - pre_poses[0:1]
                    pre_pose6d = pre_poses[1:].clone()
                else:
                    pose6d = matrix_to_pose6d(pre_c2w)
                    pre_pose6d = None

                # if self.const_speed_assumption and idx-3 >= 0:
                #     pre_poses = torch.stack([self.estimate_c2w_list[idx - 3], self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                #     pre_poses = matrix_to_pose6d(pre_poses)
                #     pose6d = 3 * pre_poses[2:] - 3 * pre_poses[1:2] + pre_poses[0:1]
                #     pre_pose6d = None
                # else:
                #     pose6d = matrix_to_pose6d(pre_c2w)
                #     pre_pose6d = None

                if self.seperate_LR:
                    T = pose6d[:, -3:]
                    quad = pose6d[:,:4]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    pose6d = torch.cat([quad, T], -1)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr, 'betas':(0.5, 0.999)},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr * 0.2, 'betas':(0.5, 0.999)}])
                else:
                    pose6d = Variable(pose6d, requires_grad=True)
                    # noise = Variable(self.noise, requires_grad=True)
                    # noise = Variable(torch.empty([1, 7], device=self.device).normal_(mean=0, std=0.00001), requires_grad=True)
                    noise_sigma = Variable(0.00001 * torch.ones_like(pose6d, device=self.device), requires_grad=True)
                    cam_para_list = [pose6d] + [noise_sigma]
                    optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr, betas=(0.5, 0.999))

                # if not self.optimizer_state is None:
                #     self.optimizer_state['state'][0]['exp_avg'] *= 0
                #     optimizer_camera.load_state_dict(self.optimizer_state)

                initial_loss_camera_tensor = torch.abs(gt_pose6d.to(device)-pose6d).mean().item()
                candidate_cam_pose6d = None
                current_min_loss = 10000000000.
                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        pose6d = torch.cat([quad, T], -1)

                    self.visualizer.vis(idx, cam_iter, gt_depth, gt_color, pose6d, all_planes, self.decoders, wandb_q)

                    # noise = noise_sigma * torch.randn_like(pose6d, device=self.device) * torch.randint(2, [1], device=self.device)
                    noise = 0
                    loss = self.optimize_cam_in_batch(cam_iter, pose6d, gt_color, gt_depth, self.tracking_pixels,
                                                      optimizer_camera, pre_pose6d, noise)

                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(gt_pose6d.to(device)-pose6d).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')


                            # wandb_q.put(({"Noise Sigma": noise_sigma.detach().abs().mean().item()}, idx))

                            wandb_q.put(({"Tracking Loss (Before)": initial_loss, "Tracking Loss (After)": loss}, idx))
                            wandb_q.put(({
                                             "Tracking Error (Before)": initial_loss_camera_tensor,
                                             "Tracking Error (After)": loss_camera_tensor,
                                             "Tracking Error (Diff)": initial_loss_camera_tensor - loss_camera_tensor
                                         }, idx))

                    # candidate_cam_pose6d = pose6d.detach()
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose6d = pose6d.detach()

                c2w = pose6d_to_matrix(candidate_cam_pose6d)
                self.optimizer_state = optimizer_camera.state_dict()

                # print('noise: ', self.noise.detach().abs().mean().item())
                # self.noise = noise.detach()

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            self.idx[0] = idx

            if self.verbose:
                print("---Tracking Time: %s seconds ---" % (time.time() - start_time))

            while not wandb_q.empty():
                wandb_val, wandb_idx = wandb_q.get()
                wandb.log(wandb_val, wandb_idx)

            if self.low_gpu_mem:
                torch.cuda.empty_cache()
        wandb.finish()
