import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate
from src.utils.Perceiver import Perceiver

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
        self.max_freq_log2 = multires - 1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0.,
                                              self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2. ** 0., 2. ** self.max_freq, steps=self.N_freqs)
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
            embedding_size = multires * 6 + 3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in
             range(n_blocks - 1)])

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
            embedding_size = multires * 6 + 3
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
            embedding_size = multires * 6 + 3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in
             range(n_blocks - 1)])

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
            embedding_size = multires * 6 + 3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.linears = nn.ModuleList(
            [DenseLayer(embedding_size + c_dim, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in
             range(n_blocks - 1)])

        self.output_linear = DenseLayer(
            hidden_size, 1, activation="linear")

    def forward(self, p, feat):
        # print(p.min(), p.max(), p.abs().min(), p.abs().max())
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        embedded_pts = self.embedder(p_nor.float())
        # print(Fore.BLUE, embedded_pts.shape, feat.shape)
        # print(Fore.WHITE)

        h = torch.cat([embedded_pts * 0, feat], dim=-1)
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
            embedding_size = multires * 6 + 3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.linears_mid = nn.ModuleList(
            [nn.Linear(c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) if i not in self.skips
             else nn.Linear(hidden_size + embedding_size, hidden_size) for i in range(n_blocks - 1)])

        self.linears_fine = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) if i not in self.skips
             else nn.Linear(hidden_size + embedding_size, hidden_size) for i in range(n_blocks - 1)])

        self.output_linear_mid = nn.Linear(hidden_size, 1, bias=False)

        self.output_linear_fine = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, p, feat_mid, feat_fine, stage):
        # p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        # embedded_pts = self.embedder(p_nor.float())

        if stage == 'middle':
            h = feat_mid
            for i, l in enumerate(self.linears_mid):
                h = self.linears_mid[i](h)
                h = F.gelu(h)
            out = torch.tanh(self.output_linear_mid(h) - 0.10)
            # out = F.softsign(self.output_linear_mid(h) - 0.10)
            # out = torch.atan(self.output_linear_mid(h) - 0.10) * 2 / torch.pi
        else:
            # h = torch.cat([feat_mid.detach(), feat_fine], dim=-1)
            h = torch.cat([feat_mid, feat_fine], dim=-1)
            for i, l in enumerate(self.linears_fine):
                h = self.linears_fine[i](h)
                h = F.gelu(h)
            out = torch.tanh(self.output_linear_fine(h) - 0.10)
            # out = F.softsign(self.output_linear_fine(h) - 0.10)
            # out = torch.atan(self.output_linear_fine(h) - 0.10) * 2 / torch.pi

        return out.squeeze(-1)


# class MyNICE(nn.Module):
#     """
#     Neural Implicit Scalable Encoding.
#
#     Args:
#         dim (int): input dimension.
#         c_dim (int): feature dimension.
#         coarse_grid_len (float): voxel length in coarse grid.
#         middle_grid_len (float): voxel length in middle grid.
#         fine_grid_len (float): voxel length in fine grid.
#         color_grid_len (float): voxel length in color grid.
#         hidden_size (int): hidden size of decoder network
#         coarse (bool): whether or not to use coarse level.
#         pos_embedding_method (str): positional embedding method.
#     """
#
#     def __init__(self, dim=3, c_dim=32,
#                  coarse_grid_len=2.0, middle_grid_len=0.16, fine_grid_len=0.16,
#                  color_grid_len=0.16, hidden_size=32, coarse=False, pos_embedding_method='fourier'):
#         super().__init__()
#
#         self.middle_grid_len = middle_grid_len
#         self.fine_grid_len = fine_grid_len
#
#         self.middle_decoder = MyMLP(name='middle', dim=dim, c_dim=c_dim, color=False,
#                                     skips=[2], n_blocks=2, hidden_size=hidden_size,
#                                     grid_len=middle_grid_len, pos_embedding_method=pos_embedding_method)
#         self.fine_decoder = MyMLP(name='fine', dim=dim, c_dim=c_dim * 2, color=False,
#                                   skips=[2], n_blocks=2, hidden_size=hidden_size,
#                                   grid_len=fine_grid_len, concat_feature=True,
#                                   pos_embedding_method=pos_embedding_method)
#         self.color_decoder = ColorMLP(name='color', dim=dim, c_dim=c_dim, color=True,
#                                       skips=[2], n_blocks=5, hidden_size=hidden_size,
#                                       grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)
#
#         self.sdf_decoder = SDF_Decoder_Sep()
#         # self.sdf_decoder = SDF_Decoder()
#         # self.sdf_decoder = Perceiver(
#         #     input_channels=8,  # number of channels for each token of the input
#         #     input_axis=1,  # number of axis for input data (2 for images, 3 for video)
#         #     num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
#         #     max_freq=10.,  # maximum frequency, hyperparameter depending on how fine the data is
#         #     depth=2,  # depth of net. The shape of the final attention mechanism will be:
#         #     #   depth * (cross attention -> self_per_cross_attn * self attention)
#         #     num_latents=4,
#         #     # number of latents, or induced set points, or centroids. different papers giving it different names
#         #     latent_dim=8,  # latent dimension
#         #     cross_heads=1,  # number of heads for cross attention. paper said 1
#         #     latent_heads=1,  # number of heads for latent self attention, 8
#         #     cross_dim_head=8,  # number of dimensions per cross attention head
#         #     latent_dim_head=8,  # number of dimensions per latent self attention head
#         #     num_classes=1,  # output number of classes
#         #     attn_dropout=0.,
#         #     ff_dropout=0.,
#         #     weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
#         #     fourier_encode_data=False,
#         #     # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
#         #     self_per_cross_attn=2  # number of self attention blocks per cross attention
#         # )
#         #
#         # multires = 4
#         # self.embedder = Nerf_positional_embedding(
#         #     multires, log_sampling=False)
#
#     def sample_grid_feature(self, p, c):
#         p_nor = normalize_3d_coordinate(p.clone(), self.bound)
#         p_nor = p_nor.unsqueeze(0)
#         vgrid = p_nor[:, :, None, None].float()
#         c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1).squeeze(-1)
#
#         return c.transpose(1, 2).squeeze(0)
#
#     def get_corners(self, p, grid_len, grid, unit_len, device):
#         local_p = p.clone()
#         local_p[..., 0] = local_p[..., 0] - self.bound[0, 0]
#         local_p[..., 1] = local_p[..., 1] - self.bound[1, 0]
#         local_p[..., 2] = local_p[..., 2] - self.bound[2, 0]
#
#         local_p = local_p / grid_len
#         local_floor = local_p.floor()
#
#         configs = torch.tensor([
#             [0, 0, 0],
#             [0, 0, 1],
#             [0, 1, 0],
#             [0, 1, 1],
#             [1, 0, 0],
#             [1, 0, 1],
#             [1, 1, 0],
#             [1, 1, 1],
#         ]).to(device)
#
#         # print(configs.shape, p.shape, grid.shape)
#         corners = local_floor[:, :, None] + configs[:, :]
#         corner_vecs = (corners - local_p[:, :, None]) * grid_len / unit_len
#         corner_vecs_shape = corner_vecs.shape
#         corner_vecs = corner_vecs.reshape(-1, 3)
#         corner_embeds = self.embedder(corner_vecs)
#         corner_embeds = corner_embeds.reshape(*corner_vecs_shape[:-1], -1)
#
#         pp = corners * grid_len
#         pp[..., 0] = pp[..., 0] + self.bound[0, 0]
#         pp[..., 1] = pp[..., 1] + self.bound[1, 0]
#         pp[..., 2] = pp[..., 2] + self.bound[2, 0]
#
#         corner_samples = self.sample_grid_feature(pp, grid)
#         corner_samples = corner_samples.reshape(local_floor.shape[1], 8, -1)
#
#         return torch.cat([corner_samples, corner_embeds.squeeze(0)], dim=-1)
#
#     # def forward(self, p, c_grid, stage='middle', **kwargs):
#     #     """
#     #         Output occupancy/color in different stage.
#     #     """
#     #     device = f'cuda:{p.get_device()}'
#     #     if stage == 'middle':
#     #         raw = torch.zeros(p.shape[1], 4).to(device).float()
#     #
#     #         ## Perceiver mode
#     #         p_float = p.float()
#     #         corner_samples_mid = self.get_corners(p_float, self.middle_grid_len, c_grid['grid_middle'],
#     #                                               self.middle_grid_len, device)
#     #         raw[..., -1] = torch.tanh(self.sdf_decoder(corner_samples_mid).squeeze(-1) - 0.1)
#     #         # raw[..., -1] = self.sdf_decoder(corner_samples_mid).squeeze(-1)
#     #
#     #         return raw, None
#     #
#     #     elif stage == 'fine' or stage == 'color':
#     #         raw = torch.zeros(p.shape[1], 4).to(device).float()
#     #
#     #         p_float = p.float()
#     #         corner_samples_mid = self.get_corners(p_float, self.middle_grid_len, c_grid['grid_middle'],
#     #                                               self.middle_grid_len, device)
#     #         corner_samples_fine = self.get_corners(p_float, self.fine_grid_len, c_grid['grid_fine'],
#     #                                                self.middle_grid_len, device)
#     #         corner_samples = torch.cat([corner_samples_mid, corner_samples_fine], dim=1)
#     #         # corner_samples = corner_samples_mid
#     #
#     #         raw[..., -1] = torch.tanh(self.sdf_decoder(corner_samples).squeeze(-1) - 0.1)
#     #         # raw[..., -1] = self.sdf_decoder(corner_samples).squeeze(-1)
#     #
#     #         return raw, None
#
#     def update_masks(self, p, grid_len, grid_mask, device):
#         local_p = p.clone()
#         local_p[..., 0] = local_p[..., 0] - self.bound[0, 0]
#         local_p[..., 1] = local_p[..., 1] - self.bound[1, 0]
#         local_p[..., 2] = local_p[..., 2] - self.bound[2, 0]
#
#         local_p = local_p / grid_len
#         local_floor = local_p.floor()
#
#         configs = torch.tensor([
#             [0, 0, 0],
#             [0, 0, 1],
#             [0, 1, 0],
#             [0, 1, 1],
#             [1, 0, 0],
#             [1, 0, 1],
#             [1, 1, 0],
#             [1, 1, 1],
#         ]).to(device)
#
#         grid_shape = grid_mask.shape
#         grid_mask = grid_mask.reshape(-1)
#
#         corners = (local_floor[:, :, None] + configs[:, :])
#         corners = corners.reshape(-1, 3)
#         corners[:, 0] = torch.clamp(corners[:, 0], 0, grid_shape[-1] - 1)
#         corners[:, 1] = torch.clamp(corners[:, 1], 0, grid_shape[-2] - 1)
#         corners[:, 2] = torch.clamp(corners[:, 2], 0, grid_shape[-3] - 1)
#
#         indices = (grid_shape[-2] * grid_shape[-1] * corners[:, 2]) + (grid_shape[-1] * corners[:, 1]) + corners[:, 0]
#         unique_indices = indices.unique().long()
#
#         grid_mask[unique_indices] = 0
#         grid_mask = grid_mask.reshape(grid_shape)
#
#         return grid_mask
#
#     def forward(self, p, p_mask, c_grid, c_mask=None, planes_xy=None, planes_xz=None, planes_yz=None, stage='middle', **kwargs):
#         """
#             Output occupancy/color in different stage.
#         """
#         device = f'cuda:{p.get_device()}'
#         if stage == 'coarse':
#             occ = self.coarse_decoder(p, c_grid)
#             occ = occ.squeeze(0)
#             raw = torch.zeros(occ.shape[0], 4).to(device).float()
#             raw[..., -1] = occ
#             return raw
#         elif stage == 'middle':
#             raw = torch.zeros(p.shape[1], 4).to(device).float()
#             middle_occ = self.middle_decoder(p, c_grid)
#             fine_occ = self.fine_decoder(p, c_grid)
#             # raw[..., -1] = torch.tanh(fine_occ+middle_occ)
#             # raw[..., -1] = self.sdf_decoder(p, fine_occ+middle_occ)
#             # raw[..., -1] = self.sdf_decoder(p, torch.cat([fine_occ, middle_occ], dim=-1))
#             raw[..., -1] = self.sdf_decoder(p, middle_occ, fine_occ, stage)
#             return raw, fine_occ
#         elif stage == 'fine':
#             middle_occ = self.middle_decoder(p, c_grid)
#             fine_occ = self.fine_decoder(p, c_grid)
#             raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
#             # raw[..., -1] = torch.tanh(fine_occ+middle_occ)
#             # raw[..., -1] = fine_occ+middle_occ
#             # raw[..., -1] = self.sdf_decoder(p, fine_occ+middle_occ)
#             # raw[..., -1] = self.sdf_decoder(p, torch.cat([fine_occ, middle_occ], dim=-1))
#             raw[..., -1] = self.sdf_decoder(p, middle_occ, fine_occ, stage)
#             return raw, fine_occ
#         elif stage == 'color':
#             if c_mask is not None:
#                 p_float = p[p_mask].float().unsqueeze(0)
#                 c_mask['grid_middle'] = self.update_masks(p_float, self.middle_grid_len, c_mask['grid_middle'], device)
#                 c_mask['grid_fine'] = self.update_masks(p_float, self.fine_grid_len, c_mask['grid_fine'], device)
#
#             middle_occ = self.middle_decoder(p, c_grid)
#             fine_occ = self.fine_decoder(p, c_grid)
#             # raw = self.color_decoder(p, c_grid)
#             raw = torch.zeros([p.shape[1], 4]).to(p.device)
#             # raw[..., -1] = torch.tanh(fine_occ+middle_occ)
#             # raw[..., -1] = fine_occ+middle_occ
#             # raw[..., -1] = self.sdf_decoder(p, fine_occ+middle_occ)
#             # raw[..., -1] = self.sdf_decoder(p, torch.cat([fine_occ, middle_occ], dim=-1))
#             raw[..., -1] = self.sdf_decoder(p, middle_occ, fine_occ, stage)
#             return raw, fine_occ

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

    def __init__(self, dim=3, c_dim=16,
                 coarse_grid_len=2.0, middle_grid_len=0.16, fine_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=16, coarse=False, pos_embedding_method='fourier', n_blocks=2, skips=[]):
        super().__init__()

        self.middle_grid_len = middle_grid_len
        self.fine_grid_len = fine_grid_len

        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        # if pos_embedding_method == 'fourier':
        #     embedding_size = 93
        #     self.embedder = GaussianFourierFeatureTransform(
        #         dim, mapping_size=embedding_size, scale=25)
        # elif pos_embedding_method == 'same':
        #     embedding_size = 3
        #     self.embedder = Same()
        # elif pos_embedding_method == 'nerf':
        #
        #     multires = 5
        #     self.embedder = Nerf_positional_embedding(
        #         multires, log_sampling=False)
        #     embedding_size = multires * 6 + 3
        # elif pos_embedding_method == 'fc_relu':
        #     embedding_size = 93
        #     self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) if i not in self.skips
             else nn.Linear(hidden_size + embedding_size, hidden_size) for i in range(n_blocks - 1)])

        self.output_linear = nn.Linear(hidden_size, 1, bias=False)

        # self.regularizer = Regularizer(c_dim)

    def sample_plane_feature(self, p, planes_xy, planes_xz, planes_yz):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        vgrid = p_nor[None, :, None]

        feat = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            # feat.append(xy + xz + yz)
            # feat.append(xy * xz * yz)
            feat.append(F.relu(xy + xz + yz, inplace=True))
            # feat.append(F.relu(xy * xz * yz, inplace=True))
            # feat.append(F.relu(xy, inplace=True) + F.relu(xz, inplace=True) + F.relu(yz, inplace=True))
            # feat.append(F.relu(xy, inplace=True) * F.relu(xz, inplace=True) * F.relu(yz, inplace=True))
        feat = torch.cat(feat, dim=-1)

        # embedded_pts = self.embedder(p_nor)
        # feat = torch.cat([feat, embedded_pts], dim=-1)

        return feat

    def forward(self, p, p_mask, c_grid, c_mask=None, planes_xy=None, planes_xz=None, planes_yz=None, stage='middle', **kwargs):
        device = f'cuda:{p.get_device()}'

        raw = torch.zeros([p.shape[1], 4], device=device)

        # regulated_xy ,regulated_xz, regulated_yz = [], [], []
        # for i in range(len(planes_xy)):
        #     regulated_xy.append(self.regularizer(planes_xy[i]))
        #     regulated_xz.append(self.regularizer(planes_xz[i]))
        #     regulated_yz.append(self.regularizer(planes_yz[i]))
        # feat = self.sample_plane_feature(p, regulated_xy, regulated_xz, regulated_yz)

        feat = self.sample_plane_feature(p, planes_xy, planes_xz, planes_yz)
        h = feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        raw[..., -1] = torch.tanh(self.output_linear(h)).squeeze()

        return raw, None

class Regularizer(nn.Module):
    def __init__(self, dim):
        super(Regularizer, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(2 * dim),
            nn.MaxPool2d(2),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(4 * dim),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(4 * dim),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, 2 * dim, 4, stride=2, padding=1),
            # nn.InstanceNorm2d(4 * dim),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, stride=2, padding=1),
            # nn.InstanceNorm2d(2 * dim),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, dim, 4, stride=2, padding=1),
            # nn.InstanceNorm2d(dim),
            nn.ELU(alpha=1.0, inplace=True),
        )
        # Output
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(dim),
            nn.ELU(alpha=1.0, inplace=True),
        )

    def forward(self, x):
        pad_h = (8 - x.shape[2] % 8) % 8
        pad_w = (8 - x.shape[3] % 8) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        input = x

        x = self.conv1(x)
        conv1_out = x
        x = self.conv2(x)
        conv2_out = x
        x = self.conv3(x)

        x = self.t_conv1(x)
        x = self.t_conv2(torch.cat([x, conv2_out], dim=1))
        x = self.t_conv3(torch.cat([x, conv1_out], dim=1))

        x = self.conv_out(torch.cat([x, input], dim=1))

        return x
