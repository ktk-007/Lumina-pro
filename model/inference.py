import torch
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
# from torch.cuda.amp import autocast  # Deprecated

from model.architecture import Generator


# ─────────────────────────────────────────
# 1. JOINT BILATERAL UPSAMPLING
# ─────────────────────────────────────────
def joint_bilateral_upsample(ab_lowres, L_highres,
                              d=15, sigma_color=75, sigma_space=75):
    """
    ab_lowres  : (H_low, W_low, 2) numpy, range [-128, 128]
    L_highres  : (H_high, W_high)  numpy, range [0, 100]
    returns    : (H_high, W_high, 2) numpy, range [-128, 128]
    """
    H, W = L_highres.shape
    # print(f"DEBUG: L_highres shape: {L_highres.shape}, dtype: {L_highres.dtype}")
    # print(f"DEBUG: ab_lowres shape: {ab_lowres.shape}, dtype: {ab_lowres.dtype}")

    # Step 1: naive upsample to target size (initialization)
    ab_init = cv2.resize(ab_lowres.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    # Step 2: build guide from high-res L
    L_uint8 = (np.clip(L_highres, 0, 100) / 100.0 * 255).astype(np.uint8)

    # Step 3: joint bilateral filter — edges from L guide ab smoothing
    ab_upsampled = np.zeros_like(ab_init)
    L_guide = L_uint8.astype(np.float32)  # Guide must match src depth
    for c in range(2):
        channel = ab_init[:, :, c].astype(np.float32)
        ab_upsampled[:, :, c] = cv2.ximgproc.jointBilateralFilter(
            L_guide, channel, d, sigma_color, sigma_space
        )

    return ab_upsampled


# ─────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────
def load_model(checkpoint_path, device='cuda'):
    """
    Loads EMA generator from deploy checkpoint.
    Falls back to regular G weights if EMA not found.
    """
    G = Generator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'ema_G' in ckpt:
        # deploy checkpoint from Phase 3
        G.load_state_dict(ckpt['ema_G'], strict=False)
        print("Loaded EMA weights.")
    elif 'G' in ckpt:
        # phase 1 or 2 checkpoint
        G.load_state_dict(ckpt['G'])
        print("Loaded Generator weights.")
    else:
        raise ValueError("No valid weights found in checkpoint.")

    G.eval()
    return G


# ─────────────────────────────────────────
# 3. COLORIZE — CORE FUNCTION
# ─────────────────────────────────────────
def colorize(G, image_input, device='cuda', apply_clahe=True, saturation_boost=1.0):
    """
    G                : loaded Generator in eval mode
    image_input      : PIL Image or numpy (H,W,3) RGB
    saturation_boost : float multiplier for ab channels to increase vibrancy
    returns          : numpy (H,W,3) uint8 RGB colorized image
    """
    # --- accept PIL or numpy ---
    if isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert('RGB'), dtype=np.float32) / 255.0
    else:
        img_rgb = image_input.astype(np.float32) / 255.0

    H_orig, W_orig = img_rgb.shape[:2]

    # --- convert to Lab ---
    img_lab    = rgb2lab(img_rgb).astype(np.float32)
    L_highres  = img_lab[:, :, 0]       # keep original high-res L

    # --- resize L to model input ---
    L_small = cv2.resize(L_highres, (256, 256), interpolation=cv2.INTER_LINEAR)
    L_tensor = torch.from_numpy(L_small / 100.0) \
                    .float() \
                    .unsqueeze(0) \
                    .unsqueeze(0) \
                    .to(device)           # (1, 1, 256, 256)

    # --- forward pass ---
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            ab_pred = G(L_tensor)         # (1, 2, 256, 256)

    ab_pred = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ab_pred = ab_pred * 128.0             # denormalize -> [-128, 128]
    ab_pred = np.clip(ab_pred * saturation_boost, -128.0, 127.0) # boost colors

    # --- joint bilateral upsample to original resolution ---
    if H_orig != 256 or W_orig != 256:
        ab_highres = joint_bilateral_upsample(ab_pred, L_highres)
    else:
        ab_highres = ab_pred

    # --- merge original L + upsampled ab ---
    lab_result = np.stack([
        L_highres,
        ab_highres[:, :, 0],
        ab_highres[:, :, 1]
    ], axis=-1)

    rgb_result = (lab2rgb(lab_result) * 255).clip(0, 255).astype(np.uint8)

    # --- optional CLAHE for local contrast ---
    if apply_clahe:
        lab_clahe       = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2LAB)
        clahe           = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_clahe[:, :, 0] = clahe.apply(lab_clahe[:, :, 0])
        rgb_result      = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return rgb_result


# ─────────────────────────────────────────
# 4. COLORIZE FROM FILE PATH
# ─────────────────────────────────────────
def colorize_file(G, input_path, output_path, device='cuda', apply_clahe=True, saturation_boost=1.0):
    """
    Convenience wrapper — reads file, colorizes, saves result.
    """
    img = Image.open(input_path).convert('RGB')
    result = colorize(G, img, device=device, apply_clahe=apply_clahe, saturation_boost=saturation_boost)
    Image.fromarray(result).save(output_path)
    print(f"Saved -> {output_path}")
    return result
