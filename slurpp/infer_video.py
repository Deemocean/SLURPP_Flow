# Frame-by-frame video wrapper around SlurppPipeline.
# Adapted from infer_real.py: decode video -> restore each frame -> re-encode.
# NOTE: restoration is per-frame (no temporal model), so expect some flicker.
#       --temporal_alpha applies an optional EMA on output frames to reduce it.

import argparse
import logging
import os

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from slurpp import load_stage1
from src.util.seeding import seed_all
from src.util.config_util import recursive_load_config
from slurpp.io import normalize_imgs


def frame_to_input(frame_bgr, res, device):
    """cv2 BGR uint8 frame -> normalized [-1,1] RGB tensor [1,3,res,res], matching UnderwaterRealDataset."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0)  # [1,3,res,res] in [0,1]
    return normalize_imgs(t, device=device)  # -> [-1,1] on device


def restored_to_frame(tensor, out_w, out_h):
    """Restored [1,3,res,res] float in [0,1] -> cv2 BGR uint8 at original (out_w,out_h)."""
    img = torch.clamp(tensor.squeeze(0), 0, 1).permute(1, 2, 0).cpu().numpy()  # HWC RGB [0,1]
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run underwater restoration on a video with SLURPP (frame by frame).")
    parser.add_argument("--config", type=str, default="", help="Path to config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage-1 checkpoint path.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to output (restored) video file.")
    parser.add_argument("--stage2_checkpoint", type=str, default=None, help="Cross-latent decoder checkpoint.")
    parser.add_argument("--denoise_steps", type=int, default=50, help="Diffusion steps (forced to 1 when config one_step).")
    parser.add_argument("--inference_resolution", type=int, default=512, help="Square resolution fed to the model (mult. of 8).")
    parser.add_argument("--temporal_alpha", type=float, default=1.0,
                        help="EMA weight on output frames in (0,1]. 1.0 = off; smaller = smoother but more motion ghosting.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Process at most N frames (-1 = all). Useful for a quick test.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    args = parser.parse_args()

    assert args.inference_resolution % 8 == 0, "inference_resolution must be a multiple of 8 (VAE constraint)."
    assert 0.0 < args.temporal_alpha <= 1.0, "temporal_alpha must be in (0, 1]."

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    seed_all(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        logging.warning("CUDA is not available. Running on CPU will be very slow for video.")
    logging.info(f"device = {device}")

    cfg = recursive_load_config(args.config)
    res = args.inference_resolution

    # -------------------- Model (mirrors infer_real.py) --------------------
    base_ckpt_dir = os.environ["BASE_CKPT_DIR"]
    model_path = f"{base_ckpt_dir}/stable-diffusion-2"

    print("LOADING STAGE 1")
    pipe, inputs_fields, outputs_fields, dual = load_stage1(model_path, args.checkpoint, cfg)

    if args.stage2_checkpoint is not None:
        from stage2 import CrossLatentUNet
        cld = CrossLatentUNet(config_path=f"{model_path}/vae/config.json")
        checkpoint = torch.load(args.stage2_checkpoint)
        cld.load_state_dict(checkpoint["state_dict"])
        cld = cld.to(device)
        del checkpoint
        # Integrate stage-2 into the pipeline (same as infer_real.py): the clear
        # output then already includes the cross-latent-decoder refinement.
        pipe.skip_connection = True
        pipe.vae_cld = cld
        print(f"          ===> Stage-2 checkpoint loaded from: {args.stage2_checkpoint}")
    else:
        print("          ===> No stage2 checkpoint provided")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")
    pipe = pipe.to(device)

    denoise_steps = args.denoise_steps
    if getattr(cfg, "one_step", False):
        print("using one step inference")
        denoise_steps = 1
    if denoise_steps == 1:
        pipe.scheduler.config.timestep_spacing = "trailing"

    # 'clear' is the first output field; that is the restored frame we keep.
    clear_idx = outputs_fields.index("clear") if "clear" in outputs_fields else 0

    # -------------------- Video IO --------------------
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input_video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)
    logging.info(f"input: {args.input_video}  {src_w}x{src_h} @ {src_fps:.2f}fps  frames={total}")

    out_dir = os.path.dirname(os.path.abspath(args.output_video))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter (codec 'mp4v'). Try a .mp4 output path.")

    # -------------------- Frame loop --------------------
    ema = None  # previous smoothed output frame (float [0,1], HWC RGB) for temporal_alpha < 1
    processed = 0
    with torch.no_grad():
        pbar = tqdm(total=total, disable=None, desc="restoring frames")
        while True:
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            inp = frame_to_input(frame_bgr, res, device)
            # For the dual model load_stage1 returns inputs_fields=["u","u"]; like
            # infer_real.py, feed the same underwater frame once per input field.
            inputs = [inp for _ in inputs_fields]
            output_pred = pipe(
                inputs,
                denoising_steps=denoise_steps,
                show_progress_bar=False,
                return_latent=True,
                is_dual=dual,
            )
            restored = output_pred[0][clear_idx:clear_idx + 1]  # [1,3,res,res] in [0,1]

            if args.temporal_alpha < 1.0:
                cur = torch.clamp(restored.squeeze(0), 0, 1).permute(1, 2, 0).cpu().numpy()
                ema = cur if ema is None else args.temporal_alpha * cur + (1.0 - args.temporal_alpha) * ema
                frame_out = restored_to_frame(torch.from_numpy(ema).permute(2, 0, 1).unsqueeze(0), src_w, src_h)
            else:
                frame_out = restored_to_frame(restored, src_w, src_h)

            writer.write(frame_out)
            processed += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    writer.release()
    logging.info(f"done: wrote {processed} frames -> {args.output_video}")
