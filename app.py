import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import time
import base64

# ── page config (must be first) ────────────────────────────────────────────
st.set_page_config(
    page_title="Lumina Pro",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── inject CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #080a0f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8eaf0;
}

/* hide streamlit chrome */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
/* section[data-testid="stSidebar"] { display: none !important; } */
/* div[data-testid="stToolbar"] { display: none !important; } */


/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f15; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 4px; }

/* noise overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ambient glow */
.stApp::after {
    content: '';
    position: fixed;
    top: -30%;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 500px;
    background: radial-gradient(ellipse, rgba(99, 102, 241, 0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* wrapper properties applied to block-container */
.stMainBlockContainer {
    position: relative;
    z-index: 1;
    min-height: 100vh;
    padding: 0 48px 64px !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* ── HEADER ── */
.lp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 32px 0 40px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 48px;
}

.lp-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.lp-logo-mark {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 24px rgba(99,102,241,0.35);
}

.lp-logo-text {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 22px;
    letter-spacing: -0.5px;
    color: #f1f2f6;
}

.lp-logo-text span {
    color: #6366f1;
}

.lp-tagline {
    font-size: 12px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 400;
}

.lp-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    font-family: 'DM Sans', sans-serif;
}

.lp-badge-dot {
    width: 6px;
    height: 6px;
    background: #6366f1;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(99,102,241,0.8);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

/* ── HERO ── */
.lp-hero {
    margin-bottom: 56px;
}

.lp-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(36px, 5vw, 64px);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -2px;
    color: #f1f2f6;
    margin-bottom: 16px;
}

.lp-hero h1 em {
    font-style: normal;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.lp-hero p {
    font-size: 16px;
    color: rgba(255,255,255,0.38);
    font-weight: 300;
    max-width: 480px;
    line-height: 1.6;
    letter-spacing: 0.01em;
}

/* ── GLASS CARD ── */
.glass {
    background: rgba(255,255,255,0.028);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

/* ── UPLOAD ZONE ── */
.upload-zone {
    border: 1.5px dashed rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 56px 32px;
    text-align: center;
    background: rgba(99,102,241,0.03);
    transition: all 0.2s ease;
    cursor: pointer;
    margin-bottom: 28px;
}

.upload-zone:hover {
    border-color: rgba(99,102,241,0.5);
    background: rgba(99,102,241,0.06);
}

.upload-icon {
    width: 52px;
    height: 52px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 20px;
    font-size: 20px;
}

.upload-title {
    font-family: 'Syne', sans-serif;
    font-size: 17px;
    font-weight: 600;
    color: #e8eaf0;
    margin-bottom: 6px;
}

.upload-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.28);
    font-weight: 300;
}

/* ── SECTION LABEL ── */
.section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.25);
    margin-bottom: 14px;
}

/* ── IMAGE PANEL ── */
.img-panel {
    border-radius: 16px;
    overflow: hidden;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
}

.img-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.img-panel-title {
    font-size: 12px;
    font-weight: 500;
    color: rgba(255,255,255,0.45);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.img-panel-badge {
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 100px;
    font-weight: 500;
    letter-spacing: 0.04em;
}

.badge-input {
    background: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.3);
    border: 1px solid rgba(255,255,255,0.08);
}

.badge-output {
    background: rgba(99,102,241,0.12);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.2);
}

/* ── CONTROLS ── */
.control-card {
    padding: 24px;
    border-radius: 20px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 16px;
}

.control-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
    margin-bottom: 18px;
    letter-spacing: -0.2px;
}

/* ── STAT BAR ── */
.stat-bar {
    display: flex;
    gap: 0;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
    margin-top: 40px;
}

.stat-item {
    flex: 1;
    padding: 20px 24px;
    background: rgba(255,255,255,0.02);
    border-right: 1px solid rgba(255,255,255,0.04);
}

.stat-item:last-child { border-right: none; }

.stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.2);
    margin-bottom: 6px;
    font-weight: 500;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: rgba(255,255,255,0.75);
    letter-spacing: -0.3px;
}

.stat-sub {
    font-size: 10px;
    color: rgba(255,255,255,0.2);
    margin-top: 2px;
}

/* ── PROCESSING ── */
.processing-bar {
    height: 2px;
    background: rgba(99,102,241,0.15);
    border-radius: 2px;
    overflow: hidden;
    margin: 12px 0;
}

.processing-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 2px;
    animation: shimmer 1.5s ease-in-out infinite;
}

@keyframes shimmer {
    0% { width: 0%; margin-left: 0%; }
    50% { width: 60%; margin-left: 20%; }
    100% { width: 0%; margin-left: 100%; }
}

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stFileUploader"] {
    background: transparent !important;
}

div[data-testid="stFileUploader"] > div {
    background: rgba(99,102,241,0.04) !important;
    border: 1.5px dashed rgba(99,102,241,0.25) !important;
    border-radius: 16px !important;
    padding: 40px !important;
}

div[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(99,102,241,0.5) !important;
    background: rgba(99,102,241,0.07) !important;
}

div[data-testid="stFileUploader"] label {
    color: rgba(255,255,255,0.5) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* glassmorphic buttons */
div[data-testid="stDownloadButton"] button,
div[data-testid="stButton"] button {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.2px !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stDownloadButton"] button:hover,
div[data-testid="stButton"] button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.25) !important;
    transform: translateY(-2px) !important;
}

/* toggle */
div[data-testid="stToggle"] label {
    color: rgba(255,255,255,0.55) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

/* selectbox */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.6) !important;
}

/* images */
div[data-testid="stImage"] img {
    border-radius: 12px !important;
    width: 100% !important;
}

/* slider */
div[data-testid="stSlider"] {
    padding: 0 !important;
}

div[data-testid="stSlider"] > div > div > div {
    background: rgba(99,102,241,0.3) !important;
}

/* caption */
div[data-testid="stCaptionContainer"] p {
    color: rgba(255,255,255,0.2) !important;
    font-size: 11px !important;
    text-align: center !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* columns gap */
div[data-testid="stHorizontalBlock"] {
    gap: 20px !important;
}

div[data-testid="stVerticalBlock"] > div {
    gap: 0px !important;
}
</style>
""", unsafe_allow_html=True)


# ── model loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(version):
    try:
        from model.architecture import Generator
        from model.inference import colorize

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        G = Generator().to(device)

        import os
        
        if version == "Phase 3 Scenic (Nature/Landscape)":
            paths_to_try = [
                'checkpoints/phase3v2_final.pt',
                'checkpoints/phase3_scenic_final.pt'
            ]
        elif version == "Phase 3 (Vibrant/GAN)":
            paths_to_try = [
                'checkpoints/deploy_coco.pt',
                'checkpoints/phase3_coco_final.pt',
                'checkpoints/phase3_final.pt'
            ]
        else:
            paths_to_try = [
                'checkpoints/phase2_final.pt',
                'checkpoints/phase1_final.pt'
            ]


        for ckpt_path in paths_to_try:
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                if 'ema_G' in ckpt:
                    G.load_state_dict(ckpt['ema_G'], strict=False)
                elif 'G' in ckpt:
                    G.load_state_dict(ckpt['G'])
                G.eval()
                return G, colorize, device, ckpt_path

        return None, None, None, "No valid checkpoint found for this version in 'checkpoints/' directory."

    except Exception as e:
        return None, None, None, str(e)


# ── helpers ─────────────────────────────────────────────────────────────────
def img_to_bytes(img_array):
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def estimate_time(w, h):
    mpx = (w * h) / 1e6
    base = 3.0 + mpx * 1.5
    return f"~{int(base)}-{int(base*1.8)}s"


# ── load model ──────────────────────────────────────────────────────────────
# ── model selection logic ───────────────────────────────────────────────────
if 'selected_version' not in st.session_state:
    st.session_state['selected_version'] = "Phase 3 (Vibrant/GAN)"

selected_version = st.session_state['selected_version']


G, colorize_fn, device, ckpt_info = load_model(selected_version)
model_loaded = G is not None

if not model_loaded and ckpt_info:
    st.error(f"Model Load Error: {ckpt_info}")



# ── layout ───────────────────────────────────────────────────────────────────
# st.markdown('<div class="lp-wrap">', unsafe_allow_html=True)

# HEADER
st.markdown(f"""
<div class="lp-header">
    <div class="lp-logo">
        <div class="lp-logo-mark">✦</div>
        <div>
            <div class="lp-logo-text">Lumina <span>Pro</span></div>
            <div class="lp-tagline">AI Image Colorization</div>
        </div>
    </div>
    <div style="display:flex; gap:12px; align-items:center;">
        <div class="lp-badge">
            <div class="lp-badge-dot"></div>
            {'GPU · ' + torch.cuda.get_device_name(0).split(' ')[-1] if torch.cuda.is_available() else 'CPU Mode'}
        </div>
        <div class="lp-badge">
            {'✓ Model Ready' if model_loaded else '⚠ No Checkpoint'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="lp-hero">
    <h1>Restore color.<br><em>Instantly.</em></h1>
    <p>Upload any grayscale or black & white image. ConvNeXt-UNet with adversarial training does the rest.</p>
</div>
""", unsafe_allow_html=True)

# MAIN GRID
left_col, right_col = st.columns([3, 1.1], gap="large")

with left_col:
    # upload
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your image here",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        label_visibility="collapsed"
    )

    if uploaded:
        img_pil = Image.open(uploaded).convert('RGB')
        
        # Prevent OOM crashes on Streamlit Cloud by capping resolution
        max_size = 1024
        if max(img_pil.size) > max_size:
            img_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            st.toast(f"Image automatically resized to {img_pil.size[0]}x{img_pil.size[1]} to prevent memory crash.")

        img_np  = np.array(img_pil)
        W, H    = img_pil.size

        st.markdown('<br>', unsafe_allow_html=True)

        # image panels
        img_left, img_right = st.columns(2, gap="small")

        with img_left:
            st.markdown("""
            <div class="img-panel">
                <div class="img-panel-header">
                    <span class="img-panel-title">Original</span>
                    <span class="img-panel-badge badge-input">Input</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(img_pil, use_container_width=True)
            st.caption(f"{W} × {H} px")

        with img_right:
            st.markdown("""
            <div class="img-panel">
                <div class="img-panel-header">
                    <span class="img-panel-title">Colorized</span>
                    <span class="img-panel-badge badge-output">Lumina Pro</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if 'result' in st.session_state and st.session_state.get('last_upload') == uploaded.name:
                result = st.session_state['result']
                st.image(result, use_container_width=True)
                st.caption("Colorized · Joint bilateral upsampling")
            else:
                st.markdown("""
                <div style="aspect-ratio:1; background:rgba(255,255,255,0.02);
                     border-radius:12px; border:1px dashed rgba(255,255,255,0.06);
                     display:flex; align-items:center; justify-content:center;
                     min-height:200px;">
                    <span style="font-size:12px; color:rgba(255,255,255,0.15);
                          font-family:'DM Sans',sans-serif; letter-spacing:0.06em;
                          text-transform:uppercase;">Awaiting colorization</span>
                </div>
                """, unsafe_allow_html=True)

        # stat bar
        st.markdown(f"""
        <div class="stat-bar">
            <div class="stat-item">
                <div class="stat-label">Resolution</div>
                <div class="stat-value">{W}×{H}</div>
                <div class="stat-sub">Original dimensions</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Est. Time</div>
                <div class="stat-value">{estimate_time(W,H)}</div>
                <div class="stat-sub">On {'GPU' if torch.cuda.is_available() else 'CPU'}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Model</div>
                <div class="stat-value">ConvNeXt-UNet</div>
                <div class="stat-sub">GAN + Histogram loss</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Output</div>
                <div class="stat-value">Full Res</div>
                <div class="stat-sub">Joint bilateral fusion</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Checkpoint</div>
                <div class="stat-value">{'Phase 3' if 'phase3' in str(ckpt_info) or 'deploy' in str(ckpt_info) else 'Phase 2' if 'phase2' in str(ckpt_info) else 'Phase 1'}</div>
                <div class="stat-sub">{'EMA weights' if 'deploy' in str(ckpt_info) else 'Generator weights'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # empty state
        st.markdown("""
        <div style="height:380px; border-radius:20px;
             background:rgba(255,255,255,0.015);
             border:1px solid rgba(255,255,255,0.04);
             display:flex; flex-direction:column;
             align-items:center; justify-content:center; gap:12px; margin-top:20px;">
            <div style="font-size:32px; opacity:0.15;">✦</div>
            <div style="font-size:13px; color:rgba(255,255,255,0.15);
                 font-family:'DM Sans',sans-serif; letter-spacing:0.06em;
                 text-transform:uppercase;">Upload an image to begin</div>
        </div>
        """, unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)

    # model selection card
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.markdown('<div class="control-title">Model Version</div>', unsafe_allow_html=True)
    
    # We use a selectbox or radio here. Given the design, a selectbox or a custom-styled radio is better.
    # But let's stick to st.radio for clarity as a 'toggler'.
    new_version = st.radio(
        "Select Engine",
        ["Phase 3 (Vibrant/GAN)", "Phase 3 Scenic (Nature/Landscape)", "Phase 2 (Conservative/Structural)"],
        index=["Phase 3 (Vibrant/GAN)", "Phase 3 Scenic (Nature/Landscape)", "Phase 2 (Conservative/Structural)"].index(st.session_state['selected_version']),
        label_visibility="collapsed"
    )

    
    if new_version != st.session_state['selected_version']:
        st.session_state['selected_version'] = new_version
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)


    # enhancement controls
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.markdown('<div class="control-title">Enhancement</div>', unsafe_allow_html=True)
    
    sat_boost = st.slider("Color Saturation", 1.0, 3.0, 1.5, 0.1,
                          help="Boosts the vibrancy of predicted colors. Higher = more colorful.")
    vibrance = st.slider("Vibrance", -1.0, 1.0, 0.0, 0.1,
                         help="Boosts intensity of muted colors without oversaturating already vibrant ones.")
    hue_shift = st.slider("Hue Shift", -90, 90, 0, 1,
                          help="Shifts the overall color spectrum (e.g. to fix green/magenta tints).")
    
    st.markdown('<br>', unsafe_allow_html=True)
    clahe_on   = st.toggle("CLAHE contrast", value=True)
    clip_limit = st.slider("Contrast strength", 1.0, 4.0, 2.0, 0.5,
                           format="%.1f",
                           help="CLAHE clipLimit — higher = more local contrast")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # output settings
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.markdown('<div class="control-title">Output</div>', unsafe_allow_html=True)
    out_format = st.selectbox("Format", ["PNG", "JPEG", "WEBP"],
                              label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # colorize button
    if uploaded and model_loaded:
        if st.button("✦  Colorize", use_container_width=True):
            with st.spinner(""):
                st.markdown("""
                <div style="margin: -8px 0 12px;">
                    <div style="font-size:11px; color:rgba(255,255,255,0.3);
                         font-family:'DM Sans',sans-serif; margin-bottom:6px;
                         letter-spacing:0.06em; text-transform:uppercase;">
                         Processing
                    </div>
                    <div class="processing-bar">
                        <div class="processing-fill"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                try:
                    t0     = time.time()
                    result = colorize_fn(
                        G, img_pil,
                        device           = device,
                        apply_clahe      = clahe_on,
                        saturation_boost = sat_boost,
                        hue_shift        = hue_shift,
                        vibrance         = vibrance
                    )
                    elapsed = time.time() - t0

                    st.session_state['result']      = result
                    st.session_state['last_upload'] = uploaded.name
                    st.session_state['elapsed']     = elapsed

                    st.success(f"Done in {elapsed:.1f}s")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

    elif uploaded and not model_loaded:
        st.markdown("""
        <div style="padding:16px; background:rgba(239,68,68,0.08);
             border:1px solid rgba(239,68,68,0.2); border-radius:12px;
             font-size:13px; color:rgba(239,68,68,0.8);
             font-family:'DM Sans',sans-serif;">
            No checkpoint found. Train the model first.
        </div>
        """, unsafe_allow_html=True)

    # download
    if 'result' in st.session_state and uploaded and \
       st.session_state.get('last_upload') == uploaded.name:

        result = st.session_state['result']
        fmt    = out_format.lower()
        mime   = {'png':'image/png', 'jpeg':'image/jpeg', 'webp':'image/webp'}[fmt]

        img_buf = Image.fromarray(result)
        buf     = io.BytesIO()
        img_buf.save(buf, format=out_format, quality=97 if fmt == 'jpeg' else None)

        st.markdown('<br>', unsafe_allow_html=True)
        st.download_button(
            label     = f"↓  Download {out_format}",
            data      = buf.getvalue(),
            file_name = f"lumina_pro_{uploaded.name.split('.')[0]}.{fmt}",
            mime      = mime,
            use_container_width = True
        )

        if 'elapsed' in st.session_state:
            st.markdown(f"""
            <div style="text-align:center; margin-top:12px;
                 font-size:11px; color:rgba(255,255,255,0.18);
                 font-family:'DM Sans',sans-serif; letter-spacing:0.04em;">
                Processed in {st.session_state['elapsed']:.1f}s
                · {W}×{H}px output
            </div>
            """, unsafe_allow_html=True)


# end of file
