import os
from PIL import Image
import torch
from matplotlib.figure import Figure
from huggingface_hub import login, snapshot_download
from freesplatter.webui.runner import FreeSplatterRunner

# ==== Configuration ====
INPUT_DIR = "./input_views"            # Folder containing multi-view input images
OUTPUT_DIR = "./tmp_outputs"           # Folder to save 3D outputs
MODEL_DIR = "./ckpts/Hunyuan3D-1"      # Where the model will be downloaded
HF_TOKEN = ""  # Your HF access token

DO_REMBG = True                         # Remove background
GS_TYPE = "2DGS"                        # "2DGS" or "3DGS"
MESH_REDUCTION = 0.9                    # Mesh simplification ratio

# ==== Step 1: Authenticate and Download Model from Hugging Face ====
os.makedirs(MODEL_DIR, exist_ok=True)
login(HF_TOKEN)
print("🔑 Hugging Face login successful.")

snapshot_download(
    repo_id="tencent/Hunyuan3D-1",
    repo_type="model",
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False
)
print(f"✅ Model downloaded to: {MODEL_DIR}")

# ==== Step 2: Load Input Views ====
image_files = [
    os.path.join(INPUT_DIR, f)
    for f in sorted(os.listdir(INPUT_DIR))
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if len(image_files) < 2:
    raise ValueError("You need at least 2 images for multi-view reconstruction.")

print(f"🖼️  Found {len(image_files)} input views.")

# ==== Step 3: Set Up FreeSplatter ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
runner = FreeSplatterRunner(device)

# ==== Step 4: Run the Multi-View → 3D Pipeline ====
images_vis, gs_ply_path, turntable_video_path, mesh_glb_path, pose_fig = runner.run_views_to_3d(
    image_files=image_files,
    do_rembg=DO_REMBG,
    gs_type=GS_TYPE,
    mesh_reduction=MESH_REDUCTION,
    cache_dir=OUTPUT_DIR,
)

# ==== Step 5: Save and Report Outputs ====
print("\n✅ 3D Reconstruction Complete!")
print(f"📍 Gaussian .ply:     {gs_ply_path}")
print(f"📍 Turntable .mp4:    {turntable_video_path}")
print(f"📍 Optimized mesh:     {mesh_glb_path}")

pose_plot_path = os.path.join(OUTPUT_DIR, "pose_plot.png")
pose_fig.savefig(pose_plot_path)
print(f"📍 Camera pose plot:  {pose_plot_path}")
