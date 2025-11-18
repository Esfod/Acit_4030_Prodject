import math
import torch
from pytorch3d.transforms import so3_exp_map
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)


def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def sample_images_at_mc_locs(target_images, sampled_rays_xy):
    """
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this method samples the tensor `target_images` at the
    respective 2D locations.
    
    This function is used in order to extract the colors from
    ground truth images that correspond to the colors
    rendered using `MonteCarloRaysampler`.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    # In order to sample target_images, we utilize
    # the grid_sample function which implements a
    # bilinear image sampler.
    # Note that we have to invert the sign of the 
    # sampled ray positions to convert the NDC xy locations
    # of the MonteCarloRaysampler to the coordinate
    # convention of grid_sample.
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2), 
        -sampled_rays_xy.view(ba, -1, 1, 2),  # note the sign inversion
        align_corners=True
    )
    return images_sampled.permute(0, 2, 3, 1).view(
        ba, *spatial_size, dim
    )


def show_full_render(
    neural_radiance_field,
    camera,
    target_image,
    target_silhouette,
    renderer_grid,
    loss_history_color,
    loss_history_sil,
):
    """
    Helper for visualizing intermediate learning results.
    Uses batched_forward to stay within memory limits.
    """
    # Prevent gradient caching.
    with torch.no_grad():
        # Render using the grid renderer and the
        # batched_forward function of neural_radiance_field.
        rendered_image_silhouette, _ = renderer_grid(
            cameras=camera, 
            volumetric_function=neural_radiance_field.batched_forward
        )
        # Split the rendering result to a silhouette render
        # and the image render.
        rendered_image, rendered_silhouette = (
            rendered_image_silhouette[0].split([3, 1], dim=-1)
        )
        
    # Generate plots.
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.ravel()
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    ax[0].plot(list(range(len(loss_history_color))), loss_history_color, linewidth=1)
    ax[1].imshow(clamp_and_detach(rendered_image))
    ax[2].imshow(clamp_and_detach(rendered_silhouette[..., 0]))
    ax[3].plot(list(range(len(loss_history_sil))), loss_history_sil, linewidth=1)
    ax[4].imshow(clamp_and_detach(target_image))
    ax[5].imshow(clamp_and_detach(target_silhouette))
    for ax_, title_ in zip(
        ax,
        (
            "loss color", "rendered image", "rendered silhouette",
            "loss silhouette", "target image",  "target silhouette",
        )
    ):
        if not title_.startswith("loss"):
            ax_.grid("off")
            ax_.axis("off")
        ax_.set_title(title_)
    fig.canvas.draw()
    return fig


def generate_rotating_nerf(
    neural_radiance_field,
    target_cameras,
    renderer_grid,
    n_frames: int = 36,  # 36 frames → 10 degrees per frame over 360°
    device: torch.device = torch.device("cpu"),
):
    """
    Render a 360-degree rotation of the learned NeRF efficiently.

    - Uses eval() + no_grad() to disable gradients.
    - Uses optional AMP on GPU for faster inference.
    - Moves frames to CPU as they are generated to free GPU memory.

    Args:
        neural_radiance_field: trained NeRF model.
        target_cameras: reference cameras from training (for znear, zfar, fov, etc.).
        renderer_grid: ImplicitRenderer used for final rendering.
                       For speed, you can pass a lighter renderer (lower res, fewer pts).
        n_frames: number of frames for the full 360° turn.
        device: torch.device.

    Returns:
        frames_stacked: (n_frames, H, W, 3) tensor on CPU.
    """
    from torch.cuda.amp import autocast

    neural_radiance_field.eval()

    # Angles from 0 to 2π (360°), but avoid duplicating the first angle at the end
    thetas = torch.linspace(0.0, 2.0 * math.pi, n_frames + 1, device=device)[:-1]

    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = thetas
    Rs = so3_exp_map(logRs)

    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7  # same distance as before

    frames = []
    print("Rendering rotating NeRF (360°) ...")

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for R, T in tqdm(zip(Rs, Ts), total=n_frames):
            camera = FoVPerspectiveCameras(
                R=R[None], 
                T=T[None], 
                znear=target_cameras.znear[0],
                zfar=target_cameras.zfar[0],
                aspect_ratio=target_cameras.aspect_ratio[0],
                fov=target_cameras.fov[0],
                device=device,
            )

            # Mixed precision on GPU to speed up rendering
            with autocast(enabled=use_amp):
                rendered = renderer_grid(
                    cameras=camera, 
                    volumetric_function=neural_radiance_field.batched_forward,
                )[0][..., :3]

            # Move each frame to CPU to free GPU memory as we go
            frames.append(rendered.cpu())

    if len(frames) == 0:
        raise RuntimeError("No frames were rendered - check the renderer and camera setup.")

    frames_stacked = torch.stack(frames, dim=0)
    print(f"Rendered frames shape: {frames_stacked.shape}")
    return frames_stacked
