# Lumina Pro: Production-Grade GAN Colorization

Lumina Pro is a state-of-the-art image colorization pipeline leveraging a hybrid **ConvNeXt-UNet** architecture with adversarial refinement. It is designed to transform grayscale images into vibrant, realistically colorized photos using a multi-phase training strategy.

## 🚀 Features
- **Hybrid Architecture**: ConvNeXt-Tiny encoder for deep semantic features + PixelShuffle decoder for high-fidelity spatial reconstruction.
- **Adversarial Training**: PatchGAN discriminator for fine-textured realism and vibrancy.
- **Composite Loss**: Huber (structural), VGG (perceptual), LSGAN (adversarial), and Color Histogram (distributional) losses.
- **Production Ready**: Optimized inference with Joint Bilateral Upsampling for high-resolution output.
- **Interactive UI**: Built with Streamlit for real-time colorization and vibrancy control.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ktk-007/Lumina-pro.git
   cd Lumina-pro
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

### Launch the UI
```bash
streamlit run app.py
```

### Run Inference (Script)
```bash
python model/inference.py --input path/to/image.jpg --output path/to/output.jpg
```

## 🧠 Training Phases

Lumina Pro is trained in three distinct phases:
1. **Phase 1 (Cold Start)**: Frozen encoder, training decoder with Huber + VGG loss.
2. **Phase 2 (Structural Refinement)**: Unfrozen encoder, adding TV and Histogram losses for global consistency.
3. **Phase 3 (Adversarial Refinement)**: Final GAN fine-tuning on diverse datasets (COCO/ImageNet) for professional vibrancy.

## 📊 Version Information
- **Version**: 1.0.0
- **Model Status**: `deploy_coco.pt` (Trained on COCO Val 2017 - 5,000 diverse scenes)
- **Architecture**: ConvNeXt-Tiny + UNet + PatchGAN

## 📜 License
MIT License. Created by ktk-007.
