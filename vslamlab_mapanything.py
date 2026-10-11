"""
Module: MapAnything - VSLAM-LAB entry point (mono / rgbd)
- Author: Alejandro Fontan Villacampa
- Version: 2.0
- Created: 2026-09-15
- Updated: 2026-10-11
- License: Apache-2.0 (MapAnything)

One feed-forward pass over every frame of the experiment's rgb csv (the framework's rgb_max / rgb_step /
rgb_placecell keys decide which frames those are) with the model selected by --module:
- mapanything (default): MapAnything itself. mono: images only, or images + calibration.yaml intrinsics with
  --use_calibration 1; rgbd: images + intrinsics + metric depth (path_<depth_name> in the rgb csv). Metric poses.
- vggt, vggt-omega, pi3, pi3x, dust3r, mast3r, must3r, pow3r: external models run through MapAnything's model
  factory wrappers (mono, images only). Each one is fed with its authors' own image preprocessing, so the results
  match their published pipelines rather than MapAnything's aspect-ratio table. Weights land in
  <TORCH_HOME>/hub/checkpoints (the pixi task points TORCH_HOME at Baselines/torch_home; the settings yaml's
  checkpoint_dir overrides it); VGGT-Omega's are gated on Hugging Face (request access + `hf auth login`, or copy
  the .pt files there by hand).
Camera-to-world poses (OpenCV convention, first view as reference) go to <exp_folder>/<exp_it>_KeyFrameTrajectory.csv;
--verbose 1 shows the reconstruction in viser.
"""

import argparse
import math
import os
import time
import webbrowser
from pathlib import Path
from urllib.error import HTTPError

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # upstream's recommendation

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from torchvision import transforms  # noqa: E402

from mapanything.models import MapAnything, model_factory  # noqa: E402
from mapanything.utils.image import load_images, preprocess_inputs  # noqa: E402


def use_hub_cache_on_github_server_errors() -> None:
    """torch.hub.load('<owner>/<repo>') without a ref first opens github.com/<owner>/<repo>/tree/main/ to choose
    main vs master, and raises on any HTTP error other than 404 before looking at its cache (uniception's DINOv2 load
    died on a GitHub 504 although the repo was cached). On a 5xx, use the cached main/master like torch.hub does
    offline; with nothing cached, the error is raised as before."""
    parse_repo_info = torch.hub._parse_repo_info

    def parse_with_cache_fallback(github):
        try:
            return parse_repo_info(github)
        except HTTPError as e:
            if e.code < 500:
                raise
            owner, name = github.split(":")[0].split("/")
            for ref in ("main", "master"):
                if os.path.exists(os.path.join(torch.hub.get_dir(), f"{owner}_{name}_{ref}")):
                    print(f"torch.hub: github.com answered {e.code} for {owner}/{name}, using the cached '{ref}' copy")
                    return owner, name, ref
            raise

    torch.hub._parse_repo_info = parse_with_cache_fallback


use_hub_cache_on_github_server_errors()

# --module -> (model_factory key, image normalisation the wrapper asserts, what comes out)
MODULES = {
    "mapanything": ("mapanything", "dinov2", "MapAnything (--model), metric poses"),
    "vggt": ("vggt", "identity", "VGGT-1B (Wang et al., CVPR 2025), poses up to scale"),
    "vggt-omega": ("vggt_omega", "identity", "VGGT-Omega-1B (--weights), poses up to scale"),
    "pi3": ("pi3", "identity", "pi^3 (Wang et al., 2025), poses up to scale"),
    "pi3x": ("pi3x", "identity", "pi^3-X, poses up to scale"),
    "dust3r": ("dust3r", "dust3r", "DUSt3R 512 dpt + global alignment, poses up to scale"),
    "mast3r": ("mast3r", "dust3r", "MASt3R 512 metric + sparse global alignment"),
    "must3r": ("must3r", "dust3r", "MUSt3R 512 (multi-view memory), metric"),
    "pow3r": ("pow3r_ba", "dust3r", "Pow3R 512 linear + global alignment, poses up to scale"),
}
DUST3R_FAMILY = ("dust3r", "mast3r", "must3r", "pow3r")

# --model (module mapanything) -> Hugging Face id
MODELS = {
    "default": "facebook/map-anything",           # v1.1, CC BY-NC 4.0, best performance
    "apache": "facebook/map-anything-apache",     # v1.1, Apache 2.0
    "v1": "facebook/map-anything-v1",
    "apache-v1": "facebook/map-anything-apache-v1",
}

# Weights: file name in the checkpoint dir -> where it comes from
VGGT_HF_REPO, VGGT_FILE = "facebook/VGGT-1B", "model.pt"      # same file as the pixi task download-vggt-weights
OMEGA_HF_REPO = "facebook/VGGT-Omega"                          # gated
# --weights (module vggt-omega) -> (checkpoint file, image_resolution it was trained for; see README / reproduction.md)
OMEGA_WEIGHTS = {
    "416-reproduce": ("vggt_omega_1b_416_reproduce.pt", 416),  # authors' reference for benchmarking
    "512": ("vggt_omega_1b_512.pt", 512),                       # original release, in-the-wild use
}
NAVER = "https://download.europe.naverlabs.com/ComputerVision"
NAVER_CHECKPOINTS = {
    "dust3r": [f"{NAVER}/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"],
    "mast3r": [f"{NAVER}/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"],
    "must3r": [f"{NAVER}/MUSt3R/MUSt3R_512.pth", f"{NAVER}/MUSt3R/MUSt3R_512_retrieval_trainingfree.pth"],
    "pow3r": [f"{NAVER}/Pow3R/Pow3R_ViTLarge_BaseDecoder_512_linear.pth"],
}
# pi3x / pow3r wrappers draw geometric inputs with these probabilities: 0 = images only (like the other modules)
NO_GEOMETRIC_INPUTS = {"overall_prob": 0.0, "dropout_prob": 1.0, "ray_dirs_prob": 0.0, "depth_prob": 0.0, "cam_prob": 0.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MapAnything (VSLAM-LAB entry point, mono / rgbd)")

    # VSLAM-LAB fixed arguments (BaselineVSLAMLAB.build_execute_command)
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None, help="cam_mono / cam_rgbd (rgb stream), checkpoint_dir")
    parser.add_argument("--verbose", type=str, default="0", help="1 opens the viser viewer with the reconstruction")
    parser.add_argument("--mode", type=str, default="mono", choices=["mono", "rgbd"], help="rgbd: module mapanything only")
    parser.add_argument("--rgb_max", type=int, default=None, help="consumed by VSLAM-LAB (Run/run_functions.py) when writing rgb_csv; accepted here so the command line parses")

    # Model selection
    parser.add_argument("--module", type=str, default="mapanything", choices=sorted(MODULES), help="model run in this pass")
    parser.add_argument("--conf_threshold", type=float, default=10.0, help="viewer only: percentage of lowest-confidence points hidden")

    # module mapanything
    parser.add_argument("--model", type=str, default="default", help=f"mapanything: one of {sorted(MODELS)} or a Hugging Face id")
    parser.add_argument("--use_calibration", type=int, default=0, help="mapanything mono: 1 feeds the calibration.yaml intrinsics (pinhole only); rgbd always does")
    parser.add_argument("--memory_efficient", type=int, default=1, help="mapanything: run the dense prediction heads in minibatches (more views for the same memory)")
    parser.add_argument("--minibatch_size", type=int, default=0, help="mapanything: 0 adaptive from free GPU memory; 1 smallest footprint")

    # module vggt
    parser.add_argument("--precision", type=str, default="autocast", choices=["autocast", "bf16"],
                        help="vggt: autocast = fp32 weights with bf16 autocast (upstream demo); bf16 = whole model in bfloat16, about half the memory")

    # module vggt-omega
    parser.add_argument("--weights", type=str, default="416-reproduce", choices=sorted(OMEGA_WEIGHTS), help="vggt-omega checkpoint; sets the preprocessing resolution")
    parser.add_argument("--image_mode", type=str, default="balanced", choices=["balanced", "max_size"], help="vggt-omega load_and_preprocess_images mode (upstream default: balanced)")

    # modules dust3r / mast3r / pow3r
    parser.add_argument("--scene_graph", type=str, default="complete", help="dust3r / mast3r / pow3r pair graph for the global alignment: complete, swin-<k>, logwin-<k>, oneref-<i>")
    return parser


def load_settings(settings_yaml: Path | None) -> dict:
    if settings_yaml is None or not settings_yaml.is_file():
        return {}
    with open(settings_yaml, "r") as f:
        return yaml.safe_load(f) or {}


def load_camera(calibration_yaml: Path, cam_name: str) -> dict:
    with open(calibration_yaml, "r") as f:
        cameras = (yaml.safe_load(f) or {}).get("cameras", [])
    for cam in cameras:
        if cam.get("cam_name") == cam_name:
            return cam
    raise SystemExit(f"camera '{cam_name}' not found in {calibration_yaml}")


def intrinsics_from_camera(cam: dict) -> np.ndarray:
    coeffs = cam.get("distortion_coefficients")
    if coeffs is not None and any(abs(float(c)) > 1e-12 for c in coeffs):
        print(f"WARNING: camera {cam['cam_name']} has {cam.get('distortion_type', 'unknown')} distortion; MapAnything intrinsics "
              f"input assumes undistorted pinhole images")
    fx, fy = cam["focal_length"]
    cx, cy = cam["principal_point"]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def resolve_model_id(model: str) -> str:
    if model in MODELS:
        return MODELS[model]
    if "/" in model:
        return model
    raise SystemExit(f"unknown model '{model}'; expected one of {sorted(MODELS)} or a Hugging Face id")


def load_frames(sequence_path: Path, rgb_csv: Path, cam_name: str) -> tuple[list[str], list[int]]:
    """Frame paths and timestamps (ns) of the camera, in csv order."""
    df = pd.read_csv(rgb_csv)
    image_names = [str(sequence_path / p) for p in df[f"path_{cam_name}"]]
    timestamps = df[f"ts_{cam_name} (ns)"].astype("int64").tolist()
    return image_names, timestamps


def write_trajectory_csv(keyframe_csv: Path, timestamps: list[int], translations: np.ndarray, quaternions_xyzw: np.ndarray) -> None:
    """VSLAM-LAB trajectory: camera-to-world translations (N, 3) and quaternions (N, 4, x y z w) with their timestamps."""
    rows = [[ts, *translations[i], *quaternions_xyzw[i]] for i, ts in enumerate(timestamps)]
    pd.DataFrame(rows, columns=["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]).to_csv(keyframe_csv, index=False)


########################################################################################################################
# Weights
def checkpoint_dir(settings: dict) -> Path:
    """settings checkpoint_dir, else the torch hub checkpoints folder (TORCH_HOME/hub/checkpoints, shared with vggtslam)."""
    ckpt_dir = Path(settings["checkpoint_dir"]) if settings.get("checkpoint_dir") else Path(torch.hub.get_dir()) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def ensure_url_checkpoint(url: str, ckpt_dir: Path) -> Path:
    """<ckpt_dir>/<file name of url>, downloaded when missing or truncated. torch.hub writes to a temp file and renames
    on completion, but a file of the wrong size (killed run, concurrent writers) is re-fetched: the size is checked
    against the server's Content-Length when the server answers."""
    from urllib.request import Request, urlopen
    path = ckpt_dir / url.rsplit("/", 1)[-1]
    expected = None
    try:
        with urlopen(Request(url, method="HEAD"), timeout=20) as response:
            expected = int(response.headers.get("Content-Length") or 0) or None
    except Exception as exc:  # offline: trust what is on disk
        print(f"WARNING: could not query {url} ({type(exc).__name__}); using the local copy unchecked")
    if path.is_file() and path.stat().st_size > 0 and (expected is None or path.stat().st_size == expected):
        return path
    if path.is_file():
        print(f"{path} has {path.stat().st_size} bytes, server reports {expected}: re-downloading")
        path.unlink()
    print(f"Downloading {url} -> {path}")
    torch.hub.download_url_to_file(url, str(path), progress=True)
    if expected is not None and path.stat().st_size != expected:
        raise SystemExit(f"{path} is incomplete ({path.stat().st_size} of {expected} bytes); re-run to download it again")
    return path


def ensure_vggt_weights(ckpt_dir: Path) -> Path:
    """VGGT-1B model.pt in the checkpoint dir; `hf download`-style (resumable, size-checked) when missing."""
    path = ckpt_dir / VGGT_FILE
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        print(f"Downloading {VGGT_HF_REPO}/{VGGT_FILE} -> {path}")
        hf_hub_download(VGGT_HF_REPO, VGGT_FILE, local_dir=str(ckpt_dir))
    return path


def ensure_omega_weights(weights: str, ckpt_dir: Path) -> Path:
    """VGGT-Omega .pt in the checkpoint dir, otherwise the gated Hugging Face download (needs access + login)."""
    filename, _ = OMEGA_WEIGHTS[weights]
    path = ckpt_dir / filename
    if path.is_file():
        return path
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, LocalTokenNotFoundError
    try:
        return Path(hf_hub_download(OMEGA_HF_REPO, filename, local_dir=str(ckpt_dir)))
    except (GatedRepoError, LocalTokenNotFoundError) as exc:
        raise SystemExit(
            f"VGGT-Omega weights are gated: request access at https://huggingface.co/{OMEGA_HF_REPO}, then run "
            f"`hf auth login` (or set HF_TOKEN), or download {filename} by hand into {ckpt_dir} (settings checkpoint_dir). "
            f"({type(exc).__name__})"
        ) from exc


########################################################################################################################
# Native preprocessing of the external modules (each family's own loader / resize rule, identity or dust3r normalisation)
def views_from_tensor(images: torch.Tensor, norm_type: str, device: str) -> list[dict]:
    """(S, 3, H, W) preprocessed frames -> MapAnything view dicts, one view per frame (batch 1)."""
    height, width = images.shape[-2:]
    return [{"img": images[i:i + 1].to(device), "data_norm_type": [norm_type], "true_shape": np.int32([[height, width]]),
             "idx": i, "instance": [str(i)], "label": ["sequence"]} for i in range(len(images))]


def load_images_pi3(image_names: list[str], pixel_limit: int = 255000) -> torch.Tensor:
    """pi3.utils.basic.load_images_as_tensor for a list of files: one size for all frames from the first one, at most
    pixel_limit pixels, multiples of 14, LANCZOS resize, [0, 1] tensors."""
    first = Image.open(image_names[0])
    w0, h0 = first.size
    scale = math.sqrt(pixel_limit / (w0 * h0))
    w_target, h_target = w0 * scale, h0 * scale
    k, m = round(w_target / 14), round(h_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    size = (max(1, k) * 14, max(1, m) * 14)
    to_tensor = transforms.ToTensor()
    return torch.stack([to_tensor(Image.open(p).convert("RGB").resize(size, Image.Resampling.LANCZOS)) for p in image_names])


def load_external_views(module: str, image_names: list[str], args: argparse.Namespace, device: str) -> list[dict]:
    norm_type = MODULES[module][1]
    if module == "vggt":
        from mapanything.models.external.vggt.utils.load_fn import load_and_preprocess_images
        images = load_and_preprocess_images(image_names, mode="crop")  # width 518, height a multiple of 14 (VGGT demo)
    elif module == "vggt-omega":
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        images = load_and_preprocess_images(image_names, mode=args.image_mode, image_resolution=OMEGA_WEIGHTS[args.weights][1])
    elif module in ("pi3", "pi3x"):
        images = load_images_pi3(image_names)
    elif module in DUST3R_FAMILY:
        from dust3r.utils.image import load_images as dust3r_load_images  # long side 512, centre crop to multiples of 16
        views = dust3r_load_images(image_names, size=512, verbose=False)
        for view in views:
            # dust3r stores instance as a plain str; the mast3r/must3r wrappers read view["instance"][0] and expect a
            # one-element list (batch 1). Indexing the str would collapse the views to 10 distinct names.
            view.update(img=view["img"].to(device), data_norm_type=[norm_type], label=["sequence"],
                        instance=[str(view["instance"])])
        return views
    else:
        raise SystemExit(f"no loader for module '{module}'")
    return views_from_tensor(images, norm_type, device)


########################################################################################################################
# Models
def build_external_model(module: str, args: argparse.Namespace, ckpt_dir: Path, exp_folder: Path, device: str) -> torch.nn.Module:
    key = MODULES[module][0]
    kwargs: dict = {"name": key, "torch_hub_force_reload": False}
    if module == "vggt":
        kwargs.update(load_pretrained_weights=False, load_custom_ckpt=True, custom_ckpt_path=str(ensure_vggt_weights(ckpt_dir)),
                      precision=args.precision)
    elif module == "vggt-omega":
        kwargs.update(checkpoint_path=str(ensure_omega_weights(args.weights, ckpt_dir)))
    elif module == "pi3x":
        kwargs.update(geometric_input_config=NO_GEOMETRIC_INPUTS)
    elif module in DUST3R_FAMILY:
        paths = [ensure_url_checkpoint(url, ckpt_dir) for url in NAVER_CHECKPOINTS[module]]
        kwargs.update(ckpt_path=str(paths[0]))
        if module == "dust3r":
            kwargs.update(scene_graph=args.scene_graph)
        elif module == "mast3r":
            cache_dir = exp_folder / "mast3r_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs.update(cache_dir=str(cache_dir), scene_graph=args.scene_graph)
        elif module == "must3r":
            kwargs.update(retrieval_ckpt_path=str(paths[1]))
        elif module == "pow3r":
            kwargs.update(geometric_input_config=NO_GEOMETRIC_INPUTS, scene_graph=args.scene_graph)
    try:
        model = model_factory(key, **kwargs)
    except ImportError as exc:
        raise SystemExit(f"module '{module}' needs the '{key}' extra of MapAnything in the mapanything env "
                         f"(pixi run install-baseline mapanything installs it): {exc}") from exc
    return model.to(device).eval()


########################################################################################################################
def main() -> None:
    args = build_parser().parse_args()
    vis = bool(int(args.verbose))
    settings = load_settings(args.settings_yaml)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    module = args.module
    print(f"Module: {module} ({MODULES[module][2]})")

    if module != "mapanything":
        if args.mode == "rgbd":
            raise SystemExit(f"mode rgbd is only available with module mapanything (module {module} takes images only)")
        if args.use_calibration:
            raise SystemExit(f"use_calibration is only available with module mapanything (module {module} predicts its own intrinsics)")
        if device != "cuda":
            raise SystemExit(f"module {module} needs a CUDA device")

    cam_name = settings.get("cam_rgbd" if args.mode == "rgbd" else "cam_mono", "rgb_0")
    image_names, timestamps = load_frames(args.sequence_path, args.rgb_csv, cam_name)
    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"

    if module == "mapanything":
        run_mapanything(args, settings, cam_name, image_names, timestamps, keyframe_csv, device, vis)
    else:
        run_external(module, args, settings, image_names, timestamps, keyframe_csv, device, vis)


def run_mapanything(args, settings, cam_name, image_names, timestamps, keyframe_csv, device, vis) -> None:
    df = pd.read_csv(args.rgb_csv)
    use_calibration = args.mode == "rgbd" or bool(args.use_calibration)
    intrinsics = depth_names = depth_factor = None
    if use_calibration:
        cam = load_camera(args.calibration_yaml, cam_name)
        intrinsics = intrinsics_from_camera(cam)
    if args.mode == "rgbd":
        depth_name, depth_factor = cam.get("depth_name"), float(cam.get("depth_factor", 1.0))
        if depth_name is None or f"path_{depth_name}" not in df.columns:
            raise SystemExit(f"rgbd mode needs a depth stream: camera {cam_name} has depth_name={depth_name} and the rgb csv "
                             f"columns are {list(df.columns)}")
        depth_names = [str(args.sequence_path / p) for p in df[f"path_{depth_name}"]]

    model_id = resolve_model_id(args.model)
    print(f"Loading MapAnything {model_id}...")
    model = MapAnything.from_pretrained(model_id).to(device).eval()

    # Views: images only through load_images (resize to the 518 mapping + dinov2 normalisation); with intrinsics and/or
    # depth through preprocess_inputs, which resizes the geometric inputs consistently with the images
    inputs = "images"
    if not use_calibration:
        views = load_images(image_names)
    else:
        raw_views = []
        for i, image_name in enumerate(image_names):
            img = cv2.cvtColor(cv2.imread(image_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
            view = {"img": img, "intrinsics": torch.from_numpy(intrinsics.copy())}
            if depth_names is not None:
                depth = cv2.imread(depth_names[i], cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_factor  # metres, 0 = missing
                view["depth_z"] = torch.from_numpy(depth)
                view["is_metric_scale"] = torch.tensor([True])
            raw_views.append(view)
        views = preprocess_inputs(raw_views)
        inputs = "images + intrinsics" + (" + metric depth" if depth_names is not None else "")

    print(f"Running inference on {len(views)} views ({inputs}), memory_efficient={bool(args.memory_efficient)}, "
          f"minibatch_size={'adaptive' if args.minibatch_size <= 0 else args.minibatch_size}...")
    t0 = time.time()
    predictions = model.infer(
        views,
        memory_efficient_inference=bool(args.memory_efficient),
        minibatch_size=None if args.minibatch_size <= 0 else args.minibatch_size,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    h, w = predictions[0]["depth_z"].shape[1:3]
    metric_scale = float(predictions[0]["metric_scaling_factor"][0])
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"Inference took {time.time() - t0:.2f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
              f"processed images {w}x{h}, metric scaling factor {metric_scale:.3f}")

    # camera_poses are camera-to-world in OpenCV convention, first view as reference
    cam_to_world = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in predictions]).astype(np.float64)
    quaternions = R.from_matrix(cam_to_world[:, :3, :3]).as_quat()  # x, y, z, w
    write_trajectory_csv(keyframe_csv, timestamps, cam_to_world[:, :3, 3], quaternions)
    print(f"Trajectory written to {keyframe_csv}")

    if vis:
        points = np.concatenate([p["pts3d"][0].float().cpu().numpy().reshape(-1, 3) for p in predictions])
        colors = np.concatenate([(p["img_no_norm"][0].float().cpu().numpy().reshape(-1, 3) * 255).astype(np.uint8) for p in predictions])
        valid = np.concatenate([p["mask"][0].cpu().numpy().reshape(-1).astype(bool) for p in predictions])
        conf = np.concatenate([p["conf"][0].float().cpu().numpy().reshape(-1) for p in predictions])
        images = np.stack([(p["img_no_norm"][0].float().cpu().numpy() * 255).astype(np.uint8) for p in predictions])
        fov_y = [2 * np.arctan2(images.shape[1] / 2, float(p["intrinsics"][0, 1, 1])) for p in predictions]
        details = (f"**{MODULES['mapanything'][2]}**  \n`{model_id}`  \n{args.mode}: {inputs}  \n"
                   f"{len(views)} frames at {w}x{h}, metric scale {metric_scale:.3f}")
        show_in_viser(points, colors, conf, valid, cam_to_world, images, fov_y, args.conf_threshold,
                      title=f"mapanything ({args.model})", details=details)


def run_external(module, args, settings, image_names, timestamps, keyframe_csv, device, vis) -> None:
    ckpt_dir = checkpoint_dir(settings)
    print(f"Checkpoints in {ckpt_dir}")
    model = build_external_model(module, args, ckpt_dir, args.exp_folder, device)

    views = load_external_views(module, image_names, args, device)
    h, w = views[0]["img"].shape[-2:]
    print(f"Running inference on {len(views)} views ({w}x{h}, {MODULES[module][1]} normalisation)...")
    t0 = time.time()
    with torch.no_grad():  # not inference_mode: the dust3r-family wrappers re-enable grad for their global alignment
        predictions = model(views)
    torch.cuda.synchronize()
    print(f"Inference took {time.time() - t0:.2f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    # Wrappers return camera-to-world translations and quaternions (x, y, z, w; vggt mat_to_quat, scalar-last)
    translations = np.stack([p["cam_trans"][0].float().cpu().numpy() for p in predictions]).astype(np.float64)
    quaternions = np.stack([p["cam_quats"][0].float().cpu().numpy() for p in predictions]).astype(np.float64)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    write_trajectory_csv(keyframe_csv, timestamps, translations, quaternions)
    print(f"Trajectory written to {keyframe_csv}")

    if vis:
        cam_to_world = np.tile(np.eye(4), (len(predictions), 1, 1))
        cam_to_world[:, :3, :3] = R.from_quat(quaternions).as_matrix()
        cam_to_world[:, :3, 3] = translations
        points = np.concatenate([p["pts3d"][0].float().cpu().numpy().reshape(-1, 3) for p in predictions])
        if all("conf" in p for p in predictions):
            conf = np.concatenate([p["conf"][0].float().cpu().numpy().reshape(-1) for p in predictions])
        else:  # a wrapper without per-pixel confidence: the viewer's confidence slider then has nothing to rank
            print(f"(viewer) module '{module}' returns no per-pixel confidence; showing every valid point")
            conf = np.ones(len(points), dtype=np.float32)
        images = np.stack([denormalize(v["img"][0], MODULES[module][1]) for v in views])  # (S, H, W, 3) uint8
        colors = images.reshape(-1, 3) if images.shape[0] * images.shape[1] * images.shape[2] == len(points) else np.full_like(points, 160, dtype=np.uint8)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(conf)
        fov_y = [2 * float(np.arccos(np.clip(p["ray_directions"][0, 0, w // 2, 2].float().cpu().numpy(), -1, 1))) for p in predictions]
        # the per-module knob that picked the weights / alignment, so two viewers of the same module are telling apart
        if module == "vggt-omega":
            variant = f"weights {args.weights}"
        elif module == "vggt":
            variant = f"precision {args.precision}"
        elif module in DUST3R_FAMILY:
            variant = f"scene_graph {args.scene_graph}"
        else:
            variant = "images only"
        details = f"**{MODULES[module][2]}**  \n{variant}  \n{len(views)} frames at {w}x{h}"
        show_in_viser(points, colors, conf, valid, cam_to_world, images, fov_y, args.conf_threshold,
                      title=module, details=details)


def denormalize(img: torch.Tensor, norm_type: str) -> np.ndarray:
    """(3, H, W) normalised view tensor -> (H, W, 3) uint8."""
    x = img.float().cpu()
    if norm_type == "dust3r":
        x = x * 0.5 + 0.5
    elif norm_type == "dinov2":
        x = x * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    return (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def show_in_viser(points: np.ndarray, colors: np.ndarray, conf: np.ndarray, valid: np.ndarray, cam_to_world: np.ndarray,
                  images: np.ndarray, fov_y: list[float], conf_threshold: float = 10.0, port: int = 8080,
                  first_client_timeout_s: float = 60.0, title: str = "mapanything", details: str = "") -> None:
    """Point cloud + camera frustums in viser; opens the browser and returns once every client has disconnected
    (or when nobody connected within first_client_timeout_s, for headless machines).
    points / colors / conf are per pixel over all views, valid marks the pixels that may be shown at all (model mask,
    finite geometry). A GUI slider hides the conf_threshold % lowest-confidence valid points, re-filtered live.
    title names the module in the browser tab and the control panel header; details (markdown) sits above the sliders."""
    import viser
    import viser.transforms as viser_tf

    valid = valid & np.isfinite(points).all(axis=1)
    conf_valid = conf[valid]

    def keep_mask(hidden_percent: float) -> np.ndarray:
        if not valid.any():
            return valid
        return valid & (conf >= np.percentile(conf_valid, hidden_percent))

    keep = keep_mask(conf_threshold)
    center = points[keep].mean(axis=0) if keep.any() else np.zeros(3)
    server = viser.ViserServer(host="0.0.0.0", port=port, label=title)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    if details:
        server.gui.add_markdown(details)

    conf_slider = server.gui.add_slider("hidden low-conf %", min=0.0, max=100.0, step=1.0, initial_value=float(conf_threshold),
                                        hint="percentage of lowest-confidence points hidden")
    size_slider = server.gui.add_slider("point size", min=0.0005, max=0.02, step=0.0005, initial_value=0.002)
    count_text = server.gui.add_text("points shown", initial_value="", disabled=True)

    def draw_points() -> None:
        keep = keep_mask(conf_slider.value)
        # re-adding under the same name replaces the node, so the cloud updates in place
        server.scene.add_point_cloud("points", points=(points[keep] - center).astype(np.float32), colors=colors[keep],
                                     point_size=size_slider.value, point_shape="circle")
        count_text.value = f"{int(keep.sum())} / {int(valid.sum())}"

    conf_slider.on_update(lambda _: draw_points())
    size_slider.on_update(lambda _: draw_points())
    draw_points()
    for i, pose in enumerate(cam_to_world):
        T = viser_tf.SE3.from_matrix(pose)
        h, w = images[i].shape[:2]
        server.scene.add_camera_frustum(f"frame_{i}", fov=fov_y[i], aspect=w / h, scale=0.05, image=images[i],
                                        wxyz=T.rotation().wxyz, position=T.translation() - center)

    url = f"http://localhost:{server.get_port()}"
    print(f"Viser viewer at {url}")
    webbrowser.open(url)
    t0 = time.time()
    had_clients = False
    try:
        while True:
            num_clients = len(server.get_clients())
            had_clients = had_clients or num_clients > 0
            if had_clients and num_clients == 0:
                print("(viser) All clients disconnected. Shutting down server.")
                return
            if not had_clients and time.time() - t0 > first_client_timeout_s:
                print(f"(viser) No client connected within {first_client_timeout_s:.0f} s. Shutting down server.")
                return
            time.sleep(0.5)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
