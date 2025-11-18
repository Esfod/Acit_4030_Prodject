import math
import torch
from torch import nn
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points


class HarmonicEmbedding(nn.Module):
    """
    Standard NeRF-style harmonic (Fourier) positional encoding.

    Given x [..., dim], returns [..., dim * 2 * n_harmonic_functions]
    with sin and cos of exponentially increasing frequencies.
    """

    def __init__(self, n_harmonic_functions: int = 60, omega0: float = 0.1):
        super().__init__()
        self.register_buffer(
            "frequencies",
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        # [..., dim * n_harm]
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class IntegratedPositionalEncoding(nn.Module):
    """
    Integrated positional encoding (IPE) for mip-NeRF.

    Approximates E[sin(ω x)] and E[cos(ω x)] when x ~ N(μ, Σ),
    with diagonal covariance Σ = diag(σ^2) for each point.

    For each dimension i and frequency ω:
      E[sin(ω x_i)] = exp(-0.5 * ω^2 σ_i^2) * sin(ω μ_i)
      E[cos(ω x_i)] = exp(-0.5 * ω^2 σ_i^2) * cos(ω μ_i)
    """

    def __init__(self, n_frequencies: int = 10, base_omega: float = math.pi):
        super().__init__()
        self.register_buffer(
            "frequencies",
            base_omega * (2.0 ** torch.arange(n_frequencies)),
        )

    def forward(
        self,
        mean: torch.Tensor,
        cov_diag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mean:     [..., 3]
            cov_diag: [..., 3]  (σ^2 per dimension)

        Returns:
            embedding: [..., 3 * n_frequencies * 2]
        """
        if mean.shape != cov_diag.shape:
            raise ValueError(
                f"mean and cov_diag must have same shape, "
                f"got {mean.shape} and {cov_diag.shape}"
            )

        freqs = self.frequencies  # [L]

        # scaled_mean, scaled_var: [..., L, 3]
        scaled_mean = mean[..., None, :] * freqs[None, :, None]
        scaled_var = cov_diag[..., None, :] * (freqs[None, :, None] ** 2)

        # Damping term exp(-0.5 * ω^2 σ^2)
        damping = torch.exp(-0.5 * scaled_var)

        sin_part = damping * torch.sin(scaled_mean)
        cos_part = damping * torch.cos(scaled_mean)

        # [..., L, 3 * 2]
        embedding = torch.cat((sin_part, cos_part), dim=-1)

        # Flatten frequency and channel dimensions
        return embedding.view(*mean.shape[:-1], -1)


class MipNeuralRadianceField(nn.Module):
    """
    A mip-NeRF–style neural radiance field.

    Differences from the baseline NeRF:
      - Uses integrated positional encoding (IPE) for 3D positions,
        based on conical frustum approximations along each ray.
      - Retains a standard harmonic embedding for view directions.

    The API is compatible with the original NeuralRadianceField:
      - forward(ray_bundle) -> (densities, colors)
      - batched_forward(ray_bundle, n_batches=16)
    """

    def __init__(
        self,
        n_frequencies_pos: int = 10,
        n_harmonic_dir: int = 24,
        n_hidden_neurons: int = 256,
        cone_angle: float = 0.01,
    ):
        super().__init__()

        # Integrated positional encoding for 3D positions.
        self.pos_encoding = IntegratedPositionalEncoding(
            n_frequencies=n_frequencies_pos
        )
        pos_embedding_dim = n_frequencies_pos * 2 * 3  # sin/cos * xyz

        # Harmonic embedding for view directions.
        self.dir_encoding = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_dir
        )
        dir_embedding_dim = n_harmonic_dir * 2 * 3

        self.cone_angle = cone_angle

        # MLP: position embedding -> features
        self.mlp = nn.Sequential(
            nn.Linear(pos_embedding_dim, n_hidden_neurons),
            nn.Softplus(beta=10.0),
            nn.Linear(n_hidden_neurons, n_hidden_neurons),
            nn.Softplus(beta=10.0),
        )

        # Density branch
        self.density_layer = nn.Sequential(
            nn.Linear(n_hidden_neurons, 1),
            nn.Softplus(beta=10.0),
        )
        # Initialize bias to encourage near-zero density at start
        self.density_layer[0].bias.data[0] = -1.5

        # Color branch: features + encoded view direction -> RGB
        self.color_layer = nn.Sequential(
            nn.Linear(n_hidden_neurons + dir_embedding_dim, n_hidden_neurons),
            nn.Softplus(beta=10.0),
            nn.Linear(n_hidden_neurons, 3),
            nn.Sigmoid(),  # colors in [0, 1]
        )

    # ------------------------ internal helpers ------------------------ #

    def _compute_gaussian_cov_diag(
        self,
        ray_bundle: RayBundle,
        rays_points_world: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate per-point covariance along each ray as isotropic,
        induced by a fixed cone angle.

        For a sample at depth t along a ray:
            σ ≈ cone_angle * t
            Σ ≈ σ^2 I
        """
        lengths = ray_bundle.lengths
        if lengths is None:
            raise ValueError("ray_bundle.lengths must be set for mip-NeRF.")

        sigma = self.cone_angle * lengths     # [B, ..., N]
        var = sigma ** 2                      # [B, ..., N]

        # Expand to xyz channels to match rays_points_world [..., N, 3]
        cov_diag = var[..., None].expand_as(rays_points_world)
        return cov_diag

    def _get_densities(self, features: torch.Tensor) -> torch.Tensor:
        """
        Map features to densities in [0, 1].
        """
        raw_densities = self.density_layer(features)
        # 1 - exp(-raw) as in the reference NeRF code
        return 1.0 - torch.exp(-raw_densities)

    def _get_colors(
        self,
        features: torch.Tensor,
        rays_directions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict RGB colors given per-point features and ray directions.
        """
        spatial_size = features.shape[:-1]

        # Normalize directions
        rays_directions_normed = nn.functional.normalize(
            rays_directions, dim=-1
        )
        # Encode directions: [..., dir_embedding_dim]
        rays_embedding = self.dir_encoding(rays_directions_normed)

        # Expand to match per-point features (add samples dimension)
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        color_input = torch.cat((features, rays_embedding_expand), dim=-1)
        return self.color_layer(color_input)

    # ------------------------ main interface ------------------------ #

    def forward(
        self,
        ray_bundle: RayBundle,
        **kwargs,
    ):
        """
        Args:
            ray_bundle: RayBundle with
                origins:     [B, ..., 3]
                directions:  [B, ..., 3]
                lengths:     [B, ..., N] (sample depths per ray)

        Returns:
            rays_densities: [B, ..., N, 1]
            rays_colors:    [B, ..., N, 3]
        """
        # World coordinates of samples
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world: [B, ..., N, 3]

        # Gaussian covariance along rays
        cov_diag = self._compute_gaussian_cov_diag(ray_bundle, rays_points_world)

        # Integrated positional encoding
        embeds_pos = self.pos_encoding(rays_points_world, cov_diag)

        # Features
        features = self.mlp(embeds_pos)

        # Density and color
        rays_densities = self._get_densities(features)
        rays_colors = self._get_colors(features, ray_bundle.directions)

        return rays_densities, rays_colors

    def batched_forward(
        self,
        ray_bundle: RayBundle,
        n_batches: int = 16,
        **kwargs,
    ):
        """
        Memory-efficient forward pass by splitting rays into batches.
        Same pattern as in NeuralRadianceField.batched_forward.
        """
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Flatten all rays and split into chunks of indices
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(
            torch.arange(tot_samples, device=ray_bundle.origins.device),
            n_batches,
        )

        batch_outputs = []
        for batch_idx in batches:
            sub_bundle = RayBundle(
                origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                xys=None,
            )
            batch_outputs.append(self.forward(sub_bundle))

        rays_densities, rays_colors = [
            torch.cat(
                [bo[i] for bo in batch_outputs], dim=0
            ).view(*spatial_size, -1)
            for i in (0, 1)
        ]

        return rays_densities, rays_colors
