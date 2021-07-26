from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
# from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork

from model.nerf_helpers import cumprod_exclusive, sample_pdf_2, compute_weights, depth2pts_outside, get_minibatches


HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


class SDFNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=0.5,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)  # ここのbiasが単位球の半径
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.forward(x)[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        # x = self.tanh(x)
        x = self.sigmoid(x)
        return x


class NeuSNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.sdf_network = SDFNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        # self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.training_scale_log = nn.Parameter(torch.ones(1) * conf.get_float('training_scale_log'))
        self.fixed_scale = conf.get_float('fixed_scale')

        # self.training_scale_log = nn.Parameter(torch.ones(1) * 3.0)
        # self.fixed_scale = 64.0
        # self.scale = nn.Parameter(torch.ones(1) * 10.0)

        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

        self.max_freq_log2 = 10
        self.max_freq_log2_viewdirs = 4
        self.netdepth = 8
        self.netwidth = 256
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=self.max_freq_log2 - 1,
                                             N_freqs=self.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=self.max_freq_log2_viewdirs - 1,
                                            N_freqs=self.max_freq_log2_viewdirs)
        self.bg_net = NeRF_BG_MLPNet(D=self.netdepth, W=self.netwidth,
                                     input_ch=self.bg_embedder_position.out_dim,
                                     input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                                     use_viewdirs=False)

    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)
        near = 0.01
        num_coarse = 64
        # num_coarse = 128
        num_fine = 64
        num_bg = 32
        # is_perturbed = True
        is_perturbed = False

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape
        # cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1)

        ro = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).view((-1, 3))
        rd = ray_dirs.view((-1, 3))
        sphere_intersections, _ = rend_util.get_sphere_intersection(
            cam_loc, ray_dirs, r=self.object_bounding_sphere)
        # near_depth, far_depth shape: [10000, 1]
        near_depth = sphere_intersections[..., 0].squeeze().unsqueeze(1)
        far_depth = sphere_intersections[..., 1].squeeze().unsqueeze(1)

        rays = torch.cat((ro, rd, near_depth, far_depth), dim=-1)

        ray_batches = get_minibatches(rays, chunksize=1024)
        rgb_values = []
        fg_rgb_values = []
        bg_rgb_values = []
        points_all = []

        for ray_batch in ray_batches:
            ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
            bounds = ray_batch[..., 6:8].view((-1, 1, 2))
            near, far = bounds[..., 0], bounds[..., 1]

            uniform_t_vals = torch.linspace(0.0, 1.0, num_coarse, dtype=ro.dtype, device=ro.device)

            uniform_z_vals = near * (1.0 - uniform_t_vals) + far * uniform_t_vals

            # perturb
            if is_perturbed:
                # Get intervals between samples.
                mids = 0.5 * (uniform_z_vals[..., 1:] + uniform_z_vals[..., :-1])
                upper = torch.cat((mids, uniform_z_vals[..., -1:]), dim=-1)
                lower = torch.cat((uniform_z_vals[..., :1], mids), dim=-1)
                # Stratified samples in those intervals.
                t_rand = torch.rand(uniform_z_vals.shape, dtype=ro.dtype, device=ro.device)
                uniform_z_vals = lower + (upper - lower) * t_rand

            uniform_sampled_pts = ro[..., None, :] + rd[..., None, :] * uniform_z_vals[..., :, None]

            coarse_weights, fine_weights = compute_weights(
                uniform_sampled_pts, self.sdf_network, self.training_scale_log, fixed_scale=self.fixed_scale)

            coarse_weights = coarse_weights.squeeze()
            fine_weights = fine_weights.squeeze()

            uniform_z_vals_mid = 0.5 * (uniform_z_vals[..., 1:] + uniform_z_vals[..., :-1])
            coarse_z_samples = sample_pdf_2(uniform_z_vals_mid, coarse_weights[..., 1:], num_coarse)
            fine_z_samples = sample_pdf_2(uniform_z_vals_mid, fine_weights[..., 1:], num_fine)

            coarse_z_samples = coarse_z_samples.detach()
            fine_z_samples = fine_z_samples.detach()

            z_vals, _ = torch.sort(torch.cat((coarse_z_samples, fine_z_samples), dim=-1), dim=-1)

            # z_vals = uniform_z_vals

            alpha_pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

            # color points are mid points of alpha points
            color_pts = 0.5 * (alpha_pts[..., :-1, :] + alpha_pts[..., 1:, :])

            flattened_alpha_pts = alpha_pts.view(-1, 3)
            flattened_color_pts = color_pts.view(-1, 3)

            flattened_pts = torch.cat([flattened_alpha_pts, flattened_color_pts], dim=0)

            flattened_sdf_output = self.sdf_network(flattened_pts)
            # alpha part
            flattened_alpha_sdf = flattened_sdf_output[:flattened_alpha_pts.shape[0], 0:1]

            # color part
            flattened_color_feature = flattened_sdf_output[flattened_alpha_pts.shape[0]:, 1:]
            flattened_color_normals = self.sdf_network.gradient(flattened_color_pts)[:, 0, :]

            flattened_color_dirs = rd.unsqueeze(1).repeat(1, color_pts.shape[1], 1).view(-1, 3) * -1.0

            flattened_color = self.rendering_network(
                flattened_color_pts, flattened_color_normals, flattened_color_dirs, flattened_color_feature)

            alpha_sdf = flattened_alpha_sdf.view(color_pts.shape[0], -1, 1)
            color = flattened_color.view(color_pts.shape[0], -1, 3)

            fg_rgb_map, bg_lambda = self.fg_volume_rendering(alpha_sdf, color)

            # background
            bg_z_vals = torch.linspace(0.001, 0.999, num_bg).unsqueeze(0).repeat(
                ro.shape[0], 1).to(ro.device)  # [num_pixels, num_bg]

            bg_ro = ro.unsqueeze(-2).repeat(1, num_bg, 1)
            bg_rd = rd.unsqueeze(-2).repeat(1, num_bg, 1)
            bg_viewdirs = -bg_rd
            bg_pts, _ = depth2pts_outside(bg_ro, bg_rd, bg_z_vals)  # [..., N_samples, 4]

            bg_input = torch.cat((self.bg_embedder_position(bg_pts),
                                  self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
            bg_input = torch.flip(bg_input, dims=[-2, ])
            bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])
            bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
            # [..., N_samples]
            bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)
            bg_raw = self.bg_net(bg_input)
            bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
            T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
            T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
            bg_weights = bg_alpha * T  # [..., N_samples]
            bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
            # bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

            rgb_map = fg_rgb_map + bg_lambda.unsqueeze(-1) * bg_rgb_map

            rgb_values.append(rgb_map)
            fg_rgb_values.append(fg_rgb_map)
            bg_rgb_values.append(bg_rgb_map)
            points_all.append(flattened_pts)

        rgb_values = torch.cat(rgb_values, dim=0)
        fg_rgb_values = torch.cat(fg_rgb_values, dim=0)
        bg_rgb_values = torch.cat(bg_rgb_values, dim=0)
        points_all = torch.cat(points_all, dim=0)

        if self.training:
            g = self.sdf_network.gradient(points_all)
            grad_theta = g[:, 0, :]
        else:
            grad_theta = None

        # Sample points for the eikonal loss
        # eik_bounding_box = self.object_bounding_sphere
        # n_eik_points = batch_size * num_pixels // 2
        # eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
        # eikonal_pixel_points = points.clone()
        # eikonal_pixel_points = eikonal_pixel_points.detach()
        # eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

        output = {
            'rgb_values': rgb_values,
            'fg_rgb_values': fg_rgb_values,
            'bg_rgb_values': bg_rgb_values,
            'grad_theta': grad_theta,
        }

        # breakpoint()

        return output

    def fg_volume_rendering(
        self,
        sdf,
        color,
        radiance_field_noise_std=0.0,
        white_background=False,
    ):
        # TESTED
        # one_e_10 = torch.tensor(
        #     [1e10], dtype=ray_directions.dtype, device=ray_directions.device
        # )
        # dists = torch.cat(
        #     (
        #         depth_values[..., 1:] - depth_values[..., :-1],
        #         one_e_10.expand(depth_values[..., :1].shape),
        #     ),
        #     dim=-1,
        # )
        # dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

        training_scale = torch.exp(self.training_scale_log)
        sdf_sigmoid = torch.sigmoid(sdf * training_scale)
        alpha = (sdf_sigmoid[..., :-1, 0:1] -
                 sdf_sigmoid[..., 1:, 0:1] + 1e-5) / (sdf_sigmoid[..., :-1, 0:1] + 1e-5)  # y = max(x, 0)は結局ReLUに等しい
        alpha = torch.clamp(alpha, 0.0, 1.0)  # 0.0から1.0の値に(1.0は実質必要ないが誤差考慮)

        # rgb = torch.sigmoid(ld[..., :3])
        rgb = color
        # noise = 0.0
        # if radiance_field_noise_std > 0.0:
        #     noise = (
        #         torch.randn(
        #             radiance_field[..., 3].shape,
        #             dtype=radiance_field.dtype,
        #             device=radiance_field.device,
        #         ) *
        #         radiance_field_noise_std
        #     )
        #     # noise = noise.to(radiance_field)
        # sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
        # alpha = 1.0 - torch.exp(-sigma_a * dists)

        alpha = alpha.squeeze()
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)
        weights = T * alpha
        weights = weights.unsqueeze(-1)

        rgb_map = weights * rgb
        rgb_map = rgb_map.sum(dim=-2)

        # depth_map = weights * depth_values
        # depth_map = depth_map.sum(dim=-1)
        # depth_map = (weights * depth_values).sum(dim=-1)
        # acc_map = weights.sum(dim=-1)
        # disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

        # if white_background:
        #     rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, bg_lambda  # , disp_map, acc_map, weights, depth_map


class NeRF_BG_MLPNet(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_viewdirs=3,
                 skips=[4], use_viewdirs=False):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs
        self.skips = skips

        self.base_layers = []
        dim = self.input_ch
        for i in range(D):
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, W), nn.ReLU())
            )
            dim = W
            if i in self.skips and i != (D - 1):      # skip connection after i^th layer
                dim += input_ch
        self.base_layers = nn.ModuleList(self.base_layers)
        # self.base_layers.apply(weights_init)        # xavier init

        sigma_layers = [nn.Linear(dim, 1), ]       # sigma must be positive
        self.sigma_layers = nn.Sequential(*sigma_layers)
        # self.sigma_layers.apply(weights_init)      # xavier init

        # rgb color
        rgb_layers = []
        base_remap_layers = [nn.Linear(dim, 256), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)
        # self.base_remap_layers.apply(weights_init)

        dim = 256 + self.input_ch_viewdirs
        for i in range(1):
            rgb_layers.append(nn.Linear(dim, W // 2))
            rgb_layers.append(nn.ReLU())
            dim = W // 2
        rgb_layers.append(nn.Linear(dim, 3))
        rgb_layers.append(nn.Sigmoid())     # rgb values are normalized to [0, 1]
        self.rgb_layers = nn.Sequential(*rgb_layers)
        # self.rgb_layers.apply(weights_init)

    def forward(self, input):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        input_pts = input[..., :self.input_ch]

        base = self.base_layers[0](input_pts)
        for i in range(len(self.base_layers) - 1):
            if i in self.skips:
                base = torch.cat((input_pts, base), dim=-1)
            base = self.base_layers[i + 1](base)

        sigma = self.sigma_layers(base)
        sigma = torch.nn.ReLU()(sigma)
        # sigma = torch.abs(sigma)

        base_remap = self.base_remap_layers(base)
        input_viewdirs = input[..., -self.input_ch_viewdirs:]
        rgb = self.rgb_layers(torch.cat((base_remap, input_viewdirs), dim=-1))

        ret = OrderedDict([('rgb', rgb),
                           ('sigma', sigma.squeeze(-1))])
        return ret


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out
