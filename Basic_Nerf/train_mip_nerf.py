import torch
import matplotlib.pyplot as plt
import os

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)

from utils.generate_cow_renders import generate_cow_renders
from utils.plot_image_grid import plot_image_grid
from utils.helper_functions import (
    generate_rotating_nerf,
    huber,
    show_full_render,
    sample_images_at_mc_locs,
)

from mip_nerf import MipNeuralRadianceField

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
    print ("Using device: ", device)
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print ("Using device: ", device)

mesh_dir = "rocket_mesh"
target_cameras, target_images, target_silhouettes = generate_cow_renders(
    num_views = 100,
    file_name = mesh_dir,
    azimuth_range = 180,
)
print(f'Generated {len(target_images)} images/silhouettes/cameras.')


render_size = target_images.shape[1] * 2
volume_extend_world = 3.0

ray_sampler_mc = MonteCarloRaysampler(
    min_x= -1.0,
    max_x= 1.0,
    min_y= -1.0,
    max_y= 1.0,
    n_rays_per_image= 8192,
    n_pts_per_ray= 64,
    min_depth= 0.1,
    max_depth= volume_extend_world,
)

raymarcher = EmissionAbsorptionRaymarcher()
render_mc = ImplicitRenderer(raysampler=ray_sampler_mc, raymarcher=raymarcher)

raysampler_grid = NDCMultinomialRaysampler(
    image_height= render_size,
    image_width= render_size,
    n_pts_per_ray= 128,
    min_depth= 0.1,
    max_depth= volume_extend_world,
)

raymarcher_grid = EmissionAbsorptionRaymarcher()
renderer_grid = ImplicitRenderer(
    raysampler= raysampler_grid,
    raymarcher= raymarcher_grid,
)


neural_radiance_field = MipNeuralRadianceField(
    n_frequencies_pos=10,
    n_harmonic_dir=24,
    n_hidden_neurons=256,
    cone_angle=0.01,
)

torch.manual_seed(1)
renderer_mc = render_mc.to(device)
renderer_grid = renderer_grid.to(device)
target_cameras = target_cameras.to(device)
target_images = target_images.to(device)
target_silhouettes = target_silhouettes.to(device)
neural_radiance_field = neural_radiance_field.to(device)

lr = 1e-3
batch_size = 4
n_iter = 2000

optimizer = torch.optim.AdamW(neural_radiance_field.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_iter,
    eta_min=5e-6,
    )
save_dir = f"output/{mesh_dir}_mip_nerf"
os.makedirs(save_dir, exist_ok=True)

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=(device.type == "cuda"))
loss_history_color, loss_history_sil = [], []

for iteration in range(n_iter):
    batch_idx = torch.randperm(len(target_images))[:batch_size]
    bartch_cameras = FoVPerspectiveCameras(
        R= target_cameras.R[batch_idx],
        T= target_cameras.T[batch_idx],
        znear= target_cameras.znear[batch_idx],
        zfar= target_cameras.zfar[batch_idx],
        aspect_ratio= target_cameras.aspect_ratio[batch_idx],
        fov= target_cameras.fov[batch_idx],
        device= device,
    )
    
    optimizer.zero_grad(set_to_none=True)
    
    sil_W = 1.0 if iteration < 3000 else 0.1
    