# *****************************************************************
# This source code is only provided for the reviewing purpose of
# CVPR 2023. The source files should not be kept or used in any
# commercial or research products. Please delete all files after
# the reviewing period.
# *****************************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate

class Decoders(nn.Module):
    def __init__(self, c_dim=32, hidden_size=16, truncation=0.08, n_blocks=2):
        super().__init__()

        self.c_dim = c_dim
        self.truncation = truncation
        self.n_blocks = n_blocks

        self.linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        self.c_linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size)  for i in range(n_blocks - 1)])

        self.output_linear = nn.Linear(hidden_size, 1)
        self.c_output_linear = nn.Linear(hidden_size, 3)

        # self.sharpness = nn.Parameter(10 * torch.ones(1))
        self.sharpness = 10

        self.drop_mode = False

    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz, act=True):
        vgrid = p_nor[None, :, None]

        feat = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            if act:
                # feat.append(F.relu(xy + xz + yz, inplace=True))
                feat.append(xy + xz + yz)
            else:
                feat.append(xy + xz + yz)
        feat = torch.cat(feat, dim=-1)

        return feat

    def get_raw_sdf(self, p_nor, all_planes):
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz, act=True)
        h = feat

        if self.drop_mode:
            h = self.drop(h)

        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        sdf = torch.tanh(self.output_linear(h)).squeeze()

        return sdf

    def get_raw_rgb(self, p_nor, all_planes):
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        c_feat = self.sample_plane_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz, act=True)
        h = c_feat

        if self.drop_mode:
            h = self.drop(h)

        for i, l in enumerate(self.c_linears):
            h = self.c_linears[i](h)
            h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(self.c_output_linear(h))

        return rgb

    def get_raw(self, p, all_planes):
        p_shape = p.shape

        p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        sdf = self.get_raw_sdf(p_nor, all_planes)
        rgb = self.get_raw_rgb(p_nor, all_planes)

        raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)
        raw = raw.reshape(*p_shape[:-1], -1)

        return raw

    def set_drop_mode(self, drop_mode):
        self.drop_mode = drop_mode

    def drop(self, h):
        dropped = h
        # mask = (torch.rand(h.shape, device=h.device) > 0.6).float()
        # dropped = mask * torch.cat([2 * h[:, :self.c_dim], 0 * h[:, self.c_dim:]], dim=-1) + (1 - mask) * h

        # if torch.rand([1]) > 0.6:
        #     dropped = torch.cat([2 * h[:,:self.c_dim], 0 * h[:,self.c_dim:]], dim=-1)
        # else:
        #     dropped = h

        return dropped