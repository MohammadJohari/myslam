import argparse

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from pathlib import Path

import sys
sys.path.append('.')
from src.utils.datasets import get_dataset
from src import config

def cull_mesh(mesh_file, cfg, args, device):
    frame_reader = get_dataset(cfg, args, 1, device=device)

    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
    scale = cfg['cam']['png_depth_scale']

    n_imgs = len(frame_reader)

    mesh = trimesh.load(mesh_file, process=False)
    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.015, max_iter=10)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.remove_unreferenced_vertices()

    pc = mesh.vertices

    whole_mask = np.ones(pc.shape[0]).astype(np.bool)
    for i in tqdm(range(0, n_imgs, 1)):
        idx, color, depth, c2w = frame_reader[i]

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
        eps = 0.10
        # mask = (0 <= -z[:, 0, 0]) & (uv[:, 0] < W -edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
        mask = (depth_samples + eps >= -z[:, 0, 0]) & (0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
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

    parser.add_argument('--input_mesh', type=str, help='path to the mesh to be culled')
    parser.add_argument('--config', type=str,  help='path to the config file')

    args = parser.parse_args()
    args.input_folder = None

    cfg = config.load_config(args.config, 'configs/nice_slam.yaml')

    cull_mesh(args.input_mesh, cfg, args, 'cuda')
