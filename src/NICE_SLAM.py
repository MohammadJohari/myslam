import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.Gridder import Gridder
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')

class NICE_SLAM():
    """
    NICE_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args
        self.nice = args.nice

        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        model = config.get_model(cfg,  nice=self.nice)
        self.shared_decoders = model

        self.scale = cfg['scale']

        self.load_bound(cfg)
        if self.nice:
            # self.load_pretrain(cfg)
            self.grid_init(cfg)
        else:
            self.shared_c = {}

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4), device=self.cfg['mapping']['device'])
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()

        for shared_planes in [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz]:
            for i, plane in enumerate(shared_planes):
                plane = plane.to(self.cfg['mapping']['device'])
                plane.share_memory_()
                shared_planes[i] = plane

        for shared_c_planes in [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz]:
            for i, plane in enumerate(shared_c_planes):
                plane = plane.to(self.cfg['mapping']['device'])
                plane.share_memory_()
                shared_c_planes[i] = plane

        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        
        ## New params
        self.truncation = 0.08

        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False, aux_mapper=False)
        # self.aux_mapper = Mapper(cfg, args, self, coarse_mapper=False, aux_mapper=True)
        if self.coarse:
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True)
        self.tracker = Tracker(cfg, args, self)
        self.gridder = Gridder(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float()
        bound_divisable = cfg['grid_len']['bound_divisable']
        # enlarge the bound a bit to allow it divisable by bound_divisable
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisable).int()+1)*bound_divisable+self.bound[:, 0]
        if self.nice:
            self.shared_decoders.bound = self.bound
            # self.shared_decoders.middle_decoder.bound = self.bound
            # self.shared_decoders.fine_decoder.bound = self.bound
            # self.shared_decoders.color_decoder.bound = self.bound
            # self.shared_decoders.sdf_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound*self.coarse_bound_enlarge

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        # if self.coarse:
        #     ckpt = torch.load(cfg['pretrained_decoders']['coarse'],
        #                       map_location=cfg['mapping']['device'])
        #     coarse_dict = {}
        #     for key, val in ckpt['model'].items():
        #         if ('decoder' in key) and ('encoder' not in key):
        #             key = key[8:]
        #             coarse_dict[key] = val
        #     self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict)

        # ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
        #                   map_location=cfg['mapping']['device'])
        # middle_dict = {}
        # fine_dict = {}
        # for key, val in ckpt['model'].items():
        #     if ('decoder' in key) and ('encoder' not in key):
        #         if 'coarse' in key:
        #             key = key[8+7:]
        #             middle_dict[key] = val
        #         elif 'fine' in key:
        #             key = key[8+5:]
        #             fine_dict[key] = val
        # self.shared_decoders.middle_decoder.load_state_dict(middle_dict)
        # self.shared_decoders.fine_decoder.load_state_dict(fine_dict)

    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids.

        Args:
            cfg (dict): parsed config dict.
        """
        if self.coarse:
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        c_dim = cfg['model']['c_dim']
        o_dim = 8
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        ####### Initializing Planes ############
        planes_xy, planes_xz, planes_yz = [], [], []
        c_planes_xy, c_planes_xz, c_planes_yz = [], [], []
        planes_res = [0.24, 0.06]
        # planes_res = [0.64, 0.32]
        planes_dim = c_dim
        for grid_len in planes_res:
            grid_shape = list(map(int, (xyz_len / grid_len).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

            c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        ########################################

        self.shared_planes_xy = planes_xy
        self.shared_planes_xz = planes_xz
        self.shared_planes_yz = planes_yz

        self.shared_c_planes_xy = c_planes_xy
        self.shared_c_planes_xz = c_planes_xz
        self.shared_c_planes_yz = c_planes_yz

    def tracking(self, rank, wandb_q):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while (1):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run(wandb_q)

    def mapping(self, rank, wandb_q):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run(wandb_q)

    def coarse_mapping(self, rank, wandb_q):
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run(wandb_q)

    def aux_mapping(self, rank, wandb_q):
        """
        Auxiliary mapping Thread.

        Args:
            rank (int): Thread ID.
        """

        self.aux_mapper.run(wandb_q)

    def gridding(self, rank, wandb_q):
        """
        gridding Thread.

        Args:
            rank (int): Thread ID.
        """

        self.gridder.run(wandb_q)

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        wandb_q = mp.Queue()
        for rank in range(-1, 3):
            if rank == -1:
                continue
                # p = mp.Process(target=self.aux_mapping, args=(rank, wandb_q, ))
                p = mp.Process(target=self.gridding, args=(rank, wandb_q, ))
            elif rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, wandb_q, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, wandb_q, ))
            elif rank == 2:
                if self.coarse:
                    p = mp.Process(target=self.coarse_mapping, args=(rank, wandb_q, ))
                else:
                    continue            
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
