import os
import argparse
import torch
import imageio.v2 as imageio

from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)

from utils.generate_cow_renders import generate_cow_renders
from utils.helper_functions import generate_rotating_nerf
from Models.basic_nerf_model import NeuralRadianceField
from Models.mip_nerf import MipNeuralRadianceField


def build_renderer_grid(
    render_size: int,
    volume_extent_world: float,
    n_pts_per_ray: int,
    device: torch.device,
) -> ImplicitRenderer:
    """
    Create a full-image renderer for evaluation / video.

    For faster videos:
        - keep render_size modest (e.g. training resolution),
        - use a smaller n_pts_per_ray (e.g. 32–64).
    """
    raysampler_grid = NDCMultinomialRaysampler(
        image_height=render_size,
        image_width=render_size,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )
    raymarcher = EmissionAbsorptionRaymarcher()
    renderer_grid = ImplicitRenderer(
        raysampler=raysampler_grid,
        raymarcher=raymarcher,
    ).to(device)
    return renderer_grid


def infer_mesh_and_type(ckpt_name: str):
    """
    Infer mesh name (e.g. 'cow_mesh') and model type ('mip' or 'basic')
    from a checkpoint filename like 'cow_mesh_mip_nerf_best.pt'.
    """
    base = os.path.splitext(os.path.basename(ckpt_name))[0]
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse mesh / model type from name: {base}")

    # Example: 'cow_mesh', 'rocket_mesh', 'sheep_mesh', ...
    mesh_name = "_".join(parts[:2])

    if "mip" in parts:
        model_type = "mip"
    else:
        model_type = "basic"

    return mesh_name, model_type


def load_model(model_type: str, checkpoint_path: str, device: torch.device):
    """
    Load either basic NeRF or mip-NeRF from a checkpoint.
    Accepts both:
        - raw state_dict
        - dict with 'model_state_dict' or 'state_dict' keys.
    """
    if model_type == "mip":
        model = MipNeuralRadianceField()
    else:
        model = NeuralRadianceField()

    state = torch.load(checkpoint_path, map_location=device)

    # Accept both raw state_dict and dict-wrapped
    if isinstance(state, dict) and any(k.endswith("state_dict") for k in state.keys()):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                state = state[key]
                break

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main(args):
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Make paths absolute relative to current working dir
    ckpt_root_abs = os.path.abspath(args.ckpt_root)
    out_root_abs = os.path.abspath(args.out_root)

    print(f"Searching for checkpoints under: {ckpt_root_abs}")
    print(f"Videos will be saved under:      {out_root_abs}")

    if not os.path.isdir(ckpt_root_abs):
        print(f"ERROR: checkpoint root does not exist: {ckpt_root_abs}")
        return

    os.makedirs(out_root_abs, exist_ok=True)

    # Cache cameras & renderer per mesh so we do not recompute them
    mesh_cache = {}
    n_ckpts = 0

    # ------------------------------------------------------------------
    # Walk all .pt checkpoints under ckpt_root
    # ------------------------------------------------------------------
    for root, _, files in os.walk(ckpt_root_abs):
        for fname in files:
            if not fname.endswith(".pt"):
                continue

            n_ckpts += 1
            ckpt_path = os.path.join(root, fname)
            mesh_name, model_type = infer_mesh_and_type(fname)

            print(f"\nFound checkpoint: {ckpt_path}")
            print(f"  mesh      : {mesh_name}")
            print(f"  model type: {model_type}")

            # Prepare cameras & renderer for this mesh (cache per mesh_name)
            if mesh_name not in mesh_cache:
                print(f"Generating reference renders for mesh '{mesh_name}'...")
                cams, images, sils = generate_cow_renders(
                    num_views=40,
                    file_name=mesh_name,
                    azimuth_range=180,
                )
                cams = cams.to(device)
                images = images.to(device)

                # For video we can use training resolution for speed
                # (if you want sharper videos, you can multiply by a scale factor)
                base_size = images.shape[1]
                render_size = int(base_size * args.render_scale)

                print(
                    f"  Using render_size={render_size}, "
                    f"n_pts_per_ray={args.n_pts_per_ray}"
                )

                renderer_grid = build_renderer_grid(
                    render_size=render_size,
                    volume_extent_world=args.volume_extent,
                    n_pts_per_ray=args.n_pts_per_ray,
                    device=device,
                )
                mesh_cache[mesh_name] = (cams, renderer_grid)

            cameras, renderer_grid = mesh_cache[mesh_name]

            # Load the model
            model = load_model(model_type, ckpt_path, device)

            # ------------------------------------------------------------------
            # Render full 360° rotation
            # (generate_rotating_nerf already uses eval + no_grad + AMP)
            # ------------------------------------------------------------------
            with torch.no_grad():
                frames = generate_rotating_nerf(
                    neural_radiance_field=model,
                    target_cameras=cameras,
                    renderer_grid=renderer_grid,
                    n_frames=args.n_frames,
                    device=device,
                )

            # --- Save output as GIF instead of MP4 (no ffmpeg needed) ---

    frames = frames.clamp(0.0, 1.0).cpu().numpy()
    frames_uint8 = (frames * 255).astype("uint8")

    rel = os.path.relpath(ckpt_path, ckpt_root_abs)
    rel_no_ext = os.path.splitext(rel)[0]
    safe_name = rel_no_ext.replace(os.sep, "_")

    gif_path = os.path.join(out_root_abs, safe_name + ".gif")
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    print(f"Saving GIF to: {gif_path}")

    # duration = seconds per frame = 1 / fps
    imageio.mimsave(gif_path, frames_uint8, duration=1.0 / args.fps)

    print(f"Finished writing GIF: {os.path.exists(gif_path)}")

    print(f"\nTotal checkpoints found: {n_ckpts}")
    if n_ckpts == 0:
        print("WARNING: No .pt files found – nothing was rendered.")
    else:
        print("All videos done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render 360-degree NeRF rotation videos for all checkpoints in a folder."
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="best_models",
        help="Root folder containing checkpoint subfolders.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="rotation_videos",
        help="Output folder for videos.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=72,  # 72 frames → 5 degrees per frame over 360°
        help="Number of frames per full 360-degree rotation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second of the output videos.",
    )
    parser.add_argument(
        "--n_pts_per_ray",
        type=int,
        default=64,  # lower than 128 for faster rendering
        help="Samples per ray for the evaluation renderer.",
    )
    parser.add_argument(
        "--volume_extent",
        type=float,
        default=3.0,
        help="World-space extent used during training.",
    )
    parser.add_argument(
        "--render_scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for render resolution relative to training image size "
            "(e.g. 1.0 = same, 0.5 = half, 2.0 = double)."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    args = parser.parse_args()
    main(args)
