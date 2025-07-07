import os
from PIL import Image
import torch
from freesplatter.webui.runner import FreeSplatterRunner

# ==== Configuration ====
INPUT_DIR = "./input_views"           # Folder containing multi-view input images
OUTPUT_DIR = "./tmp_outputs"          # Folder to save 3D outputs
DO_REMBG = True                       # Whether to remove background
GS_TYPE = "2DGS"                      # Can be "2DGS" or "3DGS"
MESH_REDUCTION = 0.9                  # Between 0 and 1, e.g., 0.9 means light simplification

# ==== Load multi-view input images ====
image_files = [
    os.path.join(INPUT_DIR, f)
    for f in sorted(os.listdir(INPUT_DIR))
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if len(image_files) < 2:
    raise ValueError("You need at least 2 images for multi-view reconstruction.")

print(f"Found {len(image_files)} input views.")

# ==== Set up device and FreeSplatter ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
runner = FreeSplatterRunner(device)

# ==== Run the multi-view to 3D pipeline ====
images_vis, gs_ply_path, turntable_video_path, mesh_glb_path, pose_fig = runner.run_views_to_3d(
    image_files=image_files,
    do_rembg=DO_REMBG,
    gs_type=GS_TYPE,
    mesh_reduction=MESH_REDUCTION,
    cache_dir=OUTPUT_DIR,
)

# ==== Save outputs ====
print("\n✅ 3D Reconstruction Complete!")
print(f"📍 Gaussian .ply:     {gs_ply_path}")
print(f"📍 Turntable .mp4:    {turntable_video_path}")
print(f"📍 Optimized mesh:     {mesh_glb_path}")

# Optional: Save pose visualization
pose_fig.savefig(os.path.join(OUTPUT_DIR, "pose_plot.png"))
print(f"📍 Camera pose plot:  {os.path.join(OUTPUT_DIR, 'pose_plot.png')}")
