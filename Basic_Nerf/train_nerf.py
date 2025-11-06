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
from utils.plot_image_grid import image_grid
from utils.generate_cow_renders import generate_cow_renders
from utils.helper_functions import (generate_rotating_nerf,
                                    huber,
                                    show_full_render,
                                    sample_images_at_mc_locs)
from nerf_model import NeuralRadianceField

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
    print(device)
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

mesh_dir = "rocket_mesh"
target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40, file_name=mesh_dir, azimuth_range=180)
print(f'Generated {len(target_images)} images/silhouettes/cameras.')

render_size = target_images.shape[1] * 2

volume_extent_world = 3.0

raysampler_mc = MonteCarloRaysampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=8192,
    n_pts_per_ray=64,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

raymarcher = EmissionAbsorptionRaymarcher()
renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)

render_size = target_images.shape[1] * 2
volume_extent_world = 3.0
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size,
    image_width=render_size,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world)

raymarcher = EmissionAbsorptionRaymarcher()
renderer_grid = ImplicitRenderer(
    raysampler=raysampler_grid, raymarcher=raymarcher)

neural_radiance_field = NeuralRadianceField()

torch.manual_seed(1)
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)
target_cameras = target_cameras.to(device)
target_images = target_images.to(device)
target_silhouettes = target_silhouettes.to(device)
neural_radiance_field = neural_radiance_field.to(device)

lr = 1e-3
batch_size = 4
n_iter = 10000
optimizer = torch.optim.AdamW(neural_radiance_field.parameters(), lr=lr) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=n_iter,eta_min=5e-6)

save_dir = f'output/{mesh_dir}'
os.makedirs(save_dir, exist_ok=True)

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=(device.type == "cuda"))

loss_history_color, loss_history_sil = [], []

for iteration in range(n_iter):

    # ---- sample a batch FIRST ----
    batch_idx = torch.randperm(len(target_cameras))[:batch_size]
    batch_cameras = FoVPerspectiveCameras(
        R=target_cameras.R[batch_idx],
        T=target_cameras.T[batch_idx],
        znear=target_cameras.znear[batch_idx],
        zfar=target_cameras.zfar[batch_idx],
        aspect_ratio=target_cameras.aspect_ratio[batch_idx],
        fov=target_cameras.fov[batch_idx],
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)

    # Optionally anneal the silhouette weight to speed color convergence
    sil_w = 1.0 if iteration < 3000 else 0.1

    with autocast(enabled=(device.type == "cuda")):
        rendered_images_silhouettes, sampled_rays = renderer_mc(
            cameras=batch_cameras, volumetric_function=neural_radiance_field
        )
        rendered_images, rendered_silhouettes = rendered_images_silhouettes.split([3, 1], dim=-1)

        silhouettes_at_rays = sample_images_at_mc_locs(
            target_silhouettes[batch_idx, ..., None], sampled_rays.xys
        )
        colors_at_rays = sample_images_at_mc_locs(
            target_images[batch_idx], sampled_rays.xys
        )

        sil_err   = huber(rendered_silhouettes, silhouettes_at_rays).abs().mean()
        color_err = huber(rendered_images, colors_at_rays).abs().mean()
        loss = color_err + sil_w * sil_err

    # ---- single AMP backward/step ----
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    loss_history_color.append(float(color_err))
    loss_history_sil.append(float(sil_err))

    # ---- visualization (unchanged) ----
    if iteration % 1000 == 0:
        show_idx = torch.randperm(len(target_cameras))[:1]
        fig = show_full_render(
            neural_radiance_field,
            FoVPerspectiveCameras(
                R=target_cameras.R[show_idx],
                T=target_cameras.T[show_idx],
                znear=target_cameras.znear[show_idx],
                zfar=target_cameras.zfar[show_idx],
                aspect_ratio=target_cameras.aspect_ratio[show_idx],
                fov=target_cameras.fov[show_idx],
                device=device,
            ),
            target_images[show_idx][0],
            target_silhouettes[show_idx][0],
            renderer_grid,
            loss_history_color,
            loss_history_sil,
        )
        fig.savefig(f'{save_dir}/intermediate_{iteration}.png')

#with torch.no_grad():
    #rotating_nerf_frames = generate_rotating_nerf(neural_radiance_field, target_cameras, renderer_grid, n_frames=3*5, device=device) 

#image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), rows=3, cols=5, rgb=True, fill=True)
#plt.show()