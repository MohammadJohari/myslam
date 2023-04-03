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

import numpy as np
import torch

import argparse
import os
import time
import glob
import trimesh
from tqdm import tqdm

from src import config
from src.tools.visualizer_util import SLAMFrontend

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments to visualize the SLAM process.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `save_imgs.mp4` in output folder ')
    parser.add_argument('--no_gt_traj', action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/ESLAM.yaml')
    scale = cfg['scale']
    mesh_resolution = cfg['meshing']['resolution']
    if mesh_resolution <= 0.01:
        wait_time = 0.25
    else:
        wait_time = 0.1
    output = cfg['data']['output'] if args.output is None else args.output
    ckptsdir = f'{output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            N = ckpt['idx']
            estimate_c2w_list[:, :3, 3] /= scale
            gt_c2w_list[:, :3, 3] /= scale
            estimate_c2w_list = estimate_c2w_list.cpu().numpy()
            gt_c2w_list = gt_c2w_list.cpu().numpy()

            ## Setting view point ##
            # get the latest .ply file in the "mesh" folder and use it to set the view point
            meshfile = sorted(glob.glob(f'{output}/mesh/*.ply'))[-1]
            if os.path.isfile(meshfile):
                mesh = trimesh.load(meshfile, process=False)
                to_origin, _ = trimesh.bounds.oriented_bounds(mesh, ordered=False)
                init_pose = np.eye(4)
                init_pose = np.linalg.inv(to_origin) @ init_pose

                frontend = SLAMFrontend(output, init_pose=init_pose, cam_scale=0.25,
                                        save_rendering=args.save_rendering, near=0,
                                        estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list)
                frontend.start()

                ## Visualize the trajectory ##
                for i in tqdm(range(0, N+1)):
                    meshfile = f'{output}/mesh/{i:05d}_mesh_culled.ply'
                    if os.path.isfile(meshfile):
                        frontend.update_mesh(meshfile)
                    frontend.update_pose(1, estimate_c2w_list[i], gt=False)
                    if not args.no_gt_traj:
                        frontend.update_pose(1, gt_c2w_list[i], gt=True)
                    if i % 10 == 0:
                        frontend.update_cam_trajectory(i, gt=False)
                        if not args.no_gt_traj:
                            frontend.update_cam_trajectory(i, gt=True)
                    time.sleep(wait_time)

                time.sleep(1)
                frontend.terminate()

                if args.save_rendering:
                    time.sleep(1)
                    os.system(f"/usr/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")
