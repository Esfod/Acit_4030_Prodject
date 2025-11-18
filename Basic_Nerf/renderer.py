# render_all_best_models.py

import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    look_at_view_transform,
)


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def get_nerf_class(nerf_type: str):
    """
    Return the correct NeRF class for 'basic' or 'mip'.
    Tries the Models/ package first and then flat imports.
    """
    if nerf_type == "basic":
        from Models.basic_nerf_model import NeuralRadianceField as NerfClass
    else:  # mip
        from Models.mip_nerf import MipNeuralRadianceField as NerfClass


    return NerfClass


def create_renderer(render_size: int, volume_extent_world: float, device):
    """
    Create the same kind of grid renderer as used during training.
    """
    raysampler_grid = NDCMultinomialRaysampler(
        image_height=render_size,
        image_width=render_size,
        n_pts_per_ray=128,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )

    raymarcher = EmissionAbsorptionRaymarcher()

    renderer_grid = ImplicitRenderer(
        raysampler=raysampler_grid,
        raymarcher=raymarcher,
    ).to(device)

    return renderer_grid


def load_best_model(model_name: str, nerf_type: str, device):
    """
    Load the best checkpoint for one experiment.
    Expects:
        best_models/<model_name>/<model_name>_best.pt
    """
    NerfClass = get_nerf_class(nerf_type)

    ckpt_dir = os.path.join("best_models", model_name)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading checkpoint: {ckpt_path}")
    model = NerfClass().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Handle both pure state_dict and dict with 'model_state_dict'
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model


def render_five_images_for_model(model_name: str, nerf_type: str,
                                 renderer, device, output_root="renders_best"):
    """
    Render 5 random views for a single model and save them to:
        renders_best/<model_name>/0001.png ... 0005.png
    """
    model = load_best_model(model_name, nerf_type, device)

    out_dir = os.path.join(output_root, model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Rendering 5 images for {model_name} -> {out_dir}")

    # Deterministic random views (remove seed for different cameras each run)
    torch.manual_seed(0)

    n_images = 5
    for i in range(1, n_images + 1):
        # Random camera sampling around the origin
        elev = torch.empty(1).uniform_(0, 0)   # degrees
        azim = torch.empty(1).uniform_(180, 180)    # degrees
        dist = torch.empty(1).uniform_(2.0, 2.0)      # distance

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        with torch.no_grad():
            rendered, _ = renderer(
                cameras=cameras,
                volumetric_function=model.batched_forward,
            )

        image = rendered[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()
        filename = f"{i:04d}.png"   # 0001, 0002, ...
        out_path = os.path.join(out_dir, filename)
        plt.imsave(out_path, image)

        print(
            f"    Saved {out_path}  "
            f"(elev={float(elev):.2f}, azim={float(azim):.2f}, dist={float(dist):.2f})"
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # One renderer reused for all models
    renderer = create_renderer(render_size=256,
                               volume_extent_world=3.0,
                               device=device)

    best_models_root = "best_models"
    model_names = [
        d for d in os.listdir(best_models_root)
        if os.path.isdir(os.path.join(best_models_root, d))
    ]
    model_names.sort()

    print("Found models:", ", ".join(model_names))

    for model_name in model_names:
        # Decide architecture automatically from folder name
        if "mip" in model_name.lower():
            nerf_type = "mip"
        else:
            nerf_type = "basic"

        print(f"\nProcessing model: {model_name}  (type: {nerf_type})")
        render_five_images_for_model(
            model_name=model_name,
            nerf_type=nerf_type,
            renderer=renderer,
            device=device,
            output_root="renders_best",
        )

    print("\nAll models rendered.")


if __name__ == "__main__":
    main()
