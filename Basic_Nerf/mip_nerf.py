import math
import torch
from torch import nn
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions: int = 60, omega0: float = 0.1):
        super().__init__()
        self.register_buffer(
            "frequencies",
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
        def forward(self, x:torch.Tensor) -> torch.Tensor:
            embed = (x[...,None] * self.frequencies).view(*x.shape[:-1], -1)
            return torch.cad((embed.sin(),embed.cos()), dim=-1)
        
class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, n_frequencies: int = 10, base_omega: float = math.pi):
        super().__init__()
        self.register_buffer(
            "frequencies",
            base_omega * (2.0 ** torch.arange(n_frequencies)),
        )
        def forward(self,mean: torch.Tensor, cov_diag: torch.Tensor,) ->torch.Tensor:
            if mean.shape != cov_diag.shape:
                raise ValueError(
                f"mean and cov_diag must have same shape, "
                f"got {mean.shape} and {cov_diag.shape}"
            )
            freqs = self.frequencies
            scaled_mean = mean[..., None, :] * freqs[None,:, None]
            scaled_var = cov_diag[..., None, :] * (freqs[None,:, None] ** 2)
            
            damping = torch.exp(-0.5 * scaled_var)
            
            sin_part =  damping * torch.sin(scaled_mean)
            cos_part = damping * torch.cos(scaled_mean)
            
            embedding = torch.cat((sin_part, cos_part), dim=-1)
            
            return embedding.view(*mean.shape[:-1], -1)

class MipNeuralRadianceField(nn.Module):
    def __init__(
        self, 
        n_frequencies_pos: int = 10,
        n_harmonic_dir: int = 24,
        n_hidden_neurons: int = 256,
        cone_angle: float= 0.01,
        ):
        super().__init__()
        self.pos_encoding = IntegratedPositionalEncoding(
            n_frequencies=n_frequencies_pos
        )
        pos_embedding_dim = n_frequencies_pos * 2 * 3
        
        self.dir_enconding = HarmonicEmbedding(n_harmonic_functions=n_harmonic_dir)
        dir_embedding_dim = n_harmonic_dir * 2 * 3
        
        self.cone_angle = cone_angle
        
        self.mlp = nn.Sequential(
            nn.Linear(pos_embedding_dim, n_hidden_neurons),
            nn.Softplus(beta = 10.0),
            nn.Linear(n_hidden_neurons, n_hidden_neurons),
            nn.Softplus(beta = 10.0),            
        )
        
        self.density_layer = nn.Sequential(
            nn.Linear(n_hidden_neurons, 1),
            nn.Softplus(beta = 10.0),
        )
        
        self.density_layer[0].bias.data[0] = -1.5

        self.color_layer = nn.Sequential(
            nn.Linear(n_hidden_neurons + dir_embedding_dim, n_hidden_neurons),
            nn.Softplus(beta = 10.0),
            nn.Linear(n_hidden_neurons, 3),
            nn.Sigmoid(),
        )
        
    def _compute_gaussian_cov_diag(
        self,
        ray_bundle: RayBundle,
        rays_points_world: torch.Tensor,
    ) -> torch.Tensor:
        lengths = ray_bundle.lengths
        if lengths is None:
            raise ValueError("ray_bundle.lengths must be set for mip-NeRF.")
        
        sigma = self.cone_angle * lengths
        var = sigma ** 2
        
        cov_diag = var[...,None].expand_as(rays_points_world)
        return cov_diag

def _get_densities(self, features: torch.Tensor) -> torch.Tensor:
    raw_densities = self.density_layer(features)
    return 1.0 - torch.exp(-raw_densities)

def _get_colors(
    self,
    features: torch.Tensor,
    rays_directions: torch.Tensor,
) -> torch.Tensor:
    spatial_size = features.shape[:-1]
    
    rays_directions_normed = nn.functional.normalize(
        rays_directions, dim=-1
    )
    
    rays_embedding = self.dir_encoding(rays_directions_normed)
    
    rays_embedding_expand = rays_embedding[..., None, :].expand(
        *spatial_size, rays_embedding.shape[-1]
    )
    
    color_input = torch.cat((features, rays_embedding_expand), dim=-1)
    return self.color_layer(color_input)

def forward(
    self,
    ray_bundle: RayBundle,
    **kwargs,
):
    rays_points_world = ray_bundle_to_ray_points(ray_bundle)
    
    cov_diag = self._compute_gaussian_cov_diag(ray_bundle,rays_points_world)
    
    embeds_pos = self.pos_encoding(rays_points_world, cov_diag)
    
    features = self.mlp(embeds_pos)
    
    ray_densities = self._get_densities(features)
    
    rays_colors = self._get_colors(features, ray_bundle.directions)
    
    return ray_densities, rays_colors

def bached_forward(
    self,
    ray_bundle: RayBundle,
    n_batches: int = 16,
    **kwargs,
):
    n_pts_per_ray = ray_bundle.lengths.shape[-1]
    spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]
    
    tot_samples = ray_bundle.origins.shape[:-1].numel()
    batches = torch.chunk(torch.arange(tot_samples, device=ray_bundle.origins.device), n_batches)
    
    batch_outputs = []
    for batch_idx in batches:
        sub_bundle = RayBundle(
            oringins = ray_bundle.origins.view(-1,3)[batch_idx],
            directions = ray_bundle.directions.view(-1,3)[batch_idx],
            lengths = ray_bundle.lengths.view(-1,n_pts_per_ray)[batch_idx],
            xys = None,
        )
        batch_outputs.append(self.forward(sub_bundle))
    
    rays_densities, rays_colors = [
        torch.cat(
            [bo[i] for bo in batch_outputs], dim=0
        ).view(*spatial_size, -1)
        for i in (0,1)
    ]
    
    return rays_densities, rays_colors