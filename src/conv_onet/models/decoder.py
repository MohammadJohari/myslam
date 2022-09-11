import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate

from colorama import Fore, Style


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(0)
        return x


class MLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid=None):
        if self.c_dim != 0:
            c = self.sample_grid_feature(
                p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

            if self.concat_feature:
                # only happen to fine decoder, get feature from middle level and concat to the current feature
                with torch.no_grad():
                    c_middle = self.sample_grid_feature(
                        p, c_grid['grid_middle']).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_middle], dim=1)

        p = p.float()

        embedded_pts = self.embedder(p)
        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                h = h + self.fc_c[i](c)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out

class ColorMLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(c_dim, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") for i in range(2)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid=None):
        c = self.sample_grid_feature(
            p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

        h = c
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        out = self.output_linear(h)
        return out

class MyMLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips


        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'

        # if self.name == 'fine':
        #     from tqdm import tqdm
        #     print(vgrid.shape)
        #     vgrid = vgrid.expand(-1, -1, 8, -1, -1).clone()
        #     for _ in tqdm(range(1000000)):
        #         ccc = F.grid_sample(c, vgrid, padding_mode='zeros', align_corners=False,
        #                   mode='nearest').squeeze(-1).squeeze(-1)

        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid=None):
        c = self.sample_grid_feature(
            p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

        return c.squeeze(-1)

class SDF_Decoder(nn.Module):

    def __init__(self, name='', dim=3, c_dim=16,
                 hidden_size=32, n_blocks=2, skips=[], pos_embedding_method='nerf'):
        super().__init__()
        self.name = name
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.linears = nn.ModuleList(
            [DenseLayer(embedding_size + c_dim, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

    def forward(self, p, feat):
        # print(p.min(), p.max(), p.abs().min(), p.abs().max())
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        embedded_pts = self.embedder(p_nor.float())
        # print(Fore.BLUE, embedded_pts.shape, feat.shape)
        # print(Fore.WHITE)

        h = torch.cat([embedded_pts, feat], dim=-1)
        # h = feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)

        out = torch.tanh(self.output_linear(h))

        return out.squeeze(-1)


class SDF_Decoder_Sep(nn.Module):

    def __init__(self, name='', dim=3, c_dim=8,
                 hidden_size=32, n_blocks=2, skips=[], pos_embedding_method='nerf'):
        super().__init__()
        self.name = name
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.linears_mid = nn.ModuleList(
            [DenseLayer(c_dim, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.linears_fine = nn.ModuleList(
            [DenseLayer(2 * c_dim, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])


        self.output_linear_mid = DenseLayer(
                hidden_size, 1, activation="linear", bias=False)
        
        self.output_linear_fine = DenseLayer(
                hidden_size, 1, activation="linear", bias=False)

    def forward(self, p, feat_mid, feat_fine, stage):
        # p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        # embedded_pts = self.embedder(p_nor.float())
    
        if stage == 'middle':
            h = feat_mid
            for i, l in enumerate(self.linears_mid):
                h = self.linears_mid[i](h)
                h = F.relu(h)
            # out = torch.tanh(self.output_linear_mid(h) - 0.10)
            out = torch.tanh(self.output_linear_mid(h) - 0.10)
            # out = self.output_linear_mid(h) - 0.10
            # out = torch.sin(0.5 * torch.pi * torch.tanh(self.output_linear_mid(h) - 0.10))
            # out = torch.sin(0.5 * torch.pi * self.output_linear_mid(h) - 0.10)
            # out = torch.asin(torch.tanh(self.output_linear_mid(h) - 0.10)) * 2 / torch.pi * 1.1
        else:
            # h = torch.cat([feat_mid.detach(), feat_fine], dim=-1)
            h = torch.cat([feat_mid, feat_fine], dim=-1)
            for i, l in enumerate(self.linears_fine):
                h = self.linears_fine[i](h)
                h = F.relu(h)
            # out = torch.tanh(self.output_linear_fine(h) - 0.10)
            out = torch.tanh(self.output_linear_fine(h) - 0.10)
            # out = self.output_linear_fine(h) - 0.10
            # out = torch.sin(0.5 * torch.pi * torch.tanh(self.output_linear_fine(h) - 0.10))
            # out = torch.sin(0.5 * torch.pi * self.output_linear_fine(h) - 0.10)
            # out = torch.asin(torch.tanh(self.output_linear_fine(h) - 0.10)) * 2 / torch.pi * 1.1


        return out.squeeze(-1)

class MyNICE(nn.Module):
    """    
    Neural Implicit Scalable Encoding.

    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        coarse_grid_len (float): voxel length in coarse grid.
        middle_grid_len (float): voxel length in middle grid.
        fine_grid_len (float): voxel length in fine grid.
        color_grid_len (float): voxel length in color grid.
        hidden_size (int): hidden size of decoder network
        coarse (bool): whether or not to use coarse level.
        pos_embedding_method (str): positional embedding method.
    """

    def __init__(self, dim=3, c_dim=32,
                 coarse_grid_len=2.0,  middle_grid_len=0.16, fine_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=32, coarse=False, pos_embedding_method='fourier'):
        super().__init__()

        self.middle_grid_len = middle_grid_len

        self.middle_decoder = MyMLP(name='middle', dim=dim, c_dim=c_dim, color=False,
                                  skips=[2], n_blocks=2, hidden_size=hidden_size,
                                  grid_len=middle_grid_len, pos_embedding_method=pos_embedding_method)
        self.fine_decoder = MyMLP(name='fine', dim=dim, c_dim=c_dim*2, color=False,
                                skips=[2], n_blocks=2, hidden_size=hidden_size,
                                grid_len=fine_grid_len, concat_feature=True, pos_embedding_method=pos_embedding_method)
        self.color_decoder = ColorMLP(name='color', dim=dim, c_dim=c_dim, color=True,
                                 skips=[2], n_blocks=5, hidden_size=hidden_size,
                                 grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)

        self.sdf_decoder = SDF_Decoder_Sep()

    def get_neighbors(self, p, grid_len, grid, unit_len, device):
        local_p = p / grid_len
        local_floor = local_p.floor()
        local_frac = local_p.frac()
        
        configs = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]).to(device)

        print(configs.shape, p.shape, grid.shape)

        # for i in range(8):
        #     grid_read = 


    def forward(self, p, c_grid, stage='middle', **kwargs):
        """
            Output occupancy/color in different stage.
        """
        device = f'cuda:{p.get_device()}'
        if stage == 'coarse':
            occ = self.coarse_decoder(p, c_grid)
            occ = occ.squeeze(0)
            raw = torch.zeros(occ.shape[0], 4).to(device).float()
            raw[..., -1] = occ
            return raw
        # elif stage == 'middle':
        #     middle_occ = self.middle_decoder(p, c_grid)
        #     raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
        #     raw[..., -1] = middle_occ
        #     return raw
        elif stage == 'middle':
            # self.get_neighbors(p, self.middle_grid_len, c_grid['grid_middle'], self.middle_grid_len, device)
            middle_occ = self.middle_decoder(p, c_grid)            
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
            # raw[..., -1] = torch.tanh(fine_occ+middle_occ)
            # raw[..., -1] = self.sdf_decoder(p, fine_occ+middle_occ)
            # raw[..., -1] = self.sdf_decoder(p, torch.cat([fine_occ, middle_occ], dim=-1))
            raw[..., -1] = self.sdf_decoder(p, middle_occ, fine_occ, stage)
            # raw[..., -1] = self.sdf_decoder(p, middle_occ)
            return raw, middle_occ
        elif stage == 'fine':
            middle_occ = self.middle_decoder(p, c_grid)
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
            # raw[..., -1] = torch.tanh(fine_occ+middle_occ)
            # raw[..., -1] = self.sdf_decoder(p, fine_occ+middle_occ)
            # raw[..., -1] = self.sdf_decoder(p, torch.cat([fine_occ, middle_occ], dim=-1))
            raw[..., -1] = self.sdf_decoder(p, middle_occ, fine_occ, stage)
            # raw[..., -1] = self.sdf_decoder(p, middle_occ)
            return raw, middle_occ
        elif stage == 'color':
            middle_occ = self.middle_decoder(p, c_grid)
            fine_occ = self.fine_decoder(p, c_grid)
            # raw = self.color_decoder(p, c_grid)
            raw = torch.zeros([p.shape[1], 4]).to(p.device)
            # raw[..., -1] = torch.tanh(fine_occ+middle_occ)
            # raw[..., -1] = self.sdf_decoder(p, fine_occ+middle_occ)
            # raw[..., -1] = self.sdf_decoder(p, torch.cat([fine_occ, middle_occ], dim=-1))
            raw[..., -1] = self.sdf_decoder(p, middle_occ, fine_occ, stage)
            # raw[..., -1] = self.sdf_decoder(p, middle_occ)
            return raw, middle_occ
