import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

import torch.nn.functional as F

from torch.optim import Adam

class Gridder(object):
    """
    Gridder thread.

    """

    def __init__(self, cfg, args, slam):

        self.cfg = cfg
        self.args = args

        self.idx = slam.idx
        self.truncation = slam.truncation

        self.scale = cfg['scale']
        self.device = cfg['mapping']['device']

        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def sampling(self):
        N = 100000
        grid_feature = self.c['grid_middle'].clone()
        grid_feature = grid_feature[:, :, :7, :7, :7]
        grid_feature = grid_feature.expand(N, -1, -1, -1, -1).clone()
        vgrid = torch.rand([N, 200, 1, 1, 3]).to(self.device)

        for _ in range(1):
            samples = F.grid_sample(grid_feature, vgrid, padding_mode='zeros',
                            align_corners=True, mode='bilinear') #.squeeze(-1).squeeze(-1)
        
        print(samples.shape)

        return samples

    def forward(self, p, c_grid, **kwargs):
        c = self.sample_grid_feature(
            p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)


    def run(self, wandb_q):
        cfg = self.cfg
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]

        while (1):
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1:
                    break
                
                # if self.sync_method == 'strict':
                #     break
                time.sleep(0.1)
                break
                
            
            start_time = time.time()

            # w = 0
            # for i in range(10000000):
            #     w = w + i + 2
            
            self.sampling()
            

            print("---Gridding Time: %s seconds ---" % (time.time() - start_time))

            if idx == self.n_img-1:
                break
