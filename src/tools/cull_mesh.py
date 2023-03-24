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

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

import os
import sys
sys.path.append('.')
from src.utils.datasets import get_dataset
from src import config

def cull_mesh(mesh_file, cfg, args, device, estimate_c2w_list=None):
    frame_reader = get_dataset(cfg, args, 1, device=device)

    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
    eval_rec = cfg['meshing']['eval_rec']

    if estimate_c2w_list is not None:
        n_imgs = len(estimate_c2w_list)
    else:
        n_imgs = len(frame_reader)

    mesh = trimesh.load(mesh_file, process=False)
    pc = mesh.vertices

    whole_mask = np.ones(pc.shape[0]).astype(np.bool)
    for i in tqdm(range(0, n_imgs, 1)):
        _, _, depth, c2w = frame_reader[i]
        depth, c2w = depth.to(device), c2w.to(device)

        if not estimate_c2w_list is None:
            c2w = estimate_c2w_list[i].to(device)

        points = pc.copy()
        points = torch.from_numpy(points).to(device)

        w2c = torch.inverse(c2w)
        K = torch.from_numpy(
            np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).to(device)
        ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
        homo_points = torch.cat(
            [points, ones], dim=1).reshape(-1, 4, 1).to(device).float()
        cam_cord_homo = w2c@homo_points
        cam_cord = cam_cord_homo[:, :3]

        cam_cord[:, 0] *= -1
        uv = K.float()@cam_cord.float()
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.squeeze(-1)

        grid = uv[None, None].clone()
        grid[..., 0] = grid[..., 0] / W
        grid[..., 1] = grid[..., 1] / H
        grid = 2 * grid - 1
        depth_samples = F.grid_sample(depth[None, None], grid, padding_mode='zeros', align_corners=True).squeeze()

        edge = 0
        if eval_rec:
            eps = 0.06
            mask = (depth_samples + eps >= -z[:, 0, 0]) & (0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
        else:
            mask = (0 <= -z[:, 0, 0]) & (uv[:, 0] < W -edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)

        mask = mask.cpu().numpy()

        whole_mask &= ~mask

    face_mask = whole_mask[mesh.faces].all(axis=1)
    mesh.update_faces(~face_mask)
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=False)

    mesh_ext = mesh_file.split('.')[-1]
    output_file = mesh_file[:-len(mesh_ext) - 1] + '_culled.' + mesh_ext

    mesh.export(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to cull the mesh.'
    )

    parser.add_argument('config', type=str,  help='path to the config file')
    parser.add_argument('--input_mesh', type=str, help='path to the mesh to be culled')

    args = parser.parse_args()
    args.input_folder = None

    cfg = config.load_config(args.config, 'configs/ESLAM.yaml')

    cull_mesh(args.input_mesh, cfg, args, 'cuda')