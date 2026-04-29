import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─────────────────────────────────────────
# 1. HUBER LOSS (Structural)
# ─────────────────────────────────────────
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        return self.loss(pred, target)


# ─────────────────────────────────────────
# 2. VGG PERCEPTUAL LOSS (Semantic Realism)
# ─────────────────────────────────────────
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # conv3_3 = up to layer 14
        # conv4_3 = up to layer 23
        self.slice1 = nn.Sequential(*list(vgg.children())[:14]).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[:23]).eval()

        # freeze VGG — never trains
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def lab_to_pseudo_rgb(self, L, ab):
        """
        Convert predicted Lab back to a 3-channel tensor VGG can process.
        Not true RGB — just a proxy good enough for feature comparison.
        """
        # L: (B,1,H,W) in [0,1] -> scale to [0,100]
        # ab: (B,2,H,W) in [-1,1] -> scale to [-128,128]
        L_scaled  = L  * 100.0
        ab_scaled = ab * 128.0
        lab = torch.cat([L_scaled, ab_scaled], dim=1)  # (B,3,H,W)

        # crude Lab->RGB approximation via clamping for VGG input
        # true conversion needs skimage but we stay in torch for speed
        rgb = (lab - lab.min()) / (lab.max() - lab.min() + 1e-8)
        rgb = rgb.repeat(1, 1, 1, 1) if rgb.shape[1] == 3 else rgb.repeat(1, 3, 1, 1)[:, :3]
        return rgb

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, L, ab_pred, ab_real):
        pred_rgb = self.lab_to_pseudo_rgb(L, ab_pred)
        real_rgb = self.lab_to_pseudo_rgb(L, ab_real)

        pred_rgb = self.normalize(pred_rgb)
        real_rgb = self.normalize(real_rgb)

        # conv3_3 features
        f1_pred = self.slice1(pred_rgb)
        f1_real = self.slice1(real_rgb)
        loss = F.mse_loss(f1_pred, f1_real)

        # conv4_3 features
        f2_pred = self.slice2(pred_rgb)
        f2_real = self.slice2(real_rgb)
        loss += F.mse_loss(f2_pred, f2_real)

        return loss


# ─────────────────────────────────────────
# 3. LSGAN LOSS (Adversarial)
# ─────────────────────────────────────────
class LSGANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def discriminator_loss(self, real_logits, fake_logits):
        # two-sided label smoothing
        real_loss = F.mse_loss(real_logits, torch.ones_like(real_logits) * 0.9)
        fake_loss = F.mse_loss(fake_logits, torch.zeros_like(fake_logits) + 0.1)
        return 0.5 * (real_loss + fake_loss)

    def generator_loss(self, fake_logits):
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


# ─────────────────────────────────────────
# 4. TOTAL VARIATION LOSS (Smoothness)
# ─────────────────────────────────────────
class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ab_pred):
        dh = torch.abs(ab_pred[:, :, 1:, :]  - ab_pred[:, :, :-1, :]).mean()
        dw = torch.abs(ab_pred[:, :, :, 1:]  - ab_pred[:, :, :, :-1]).mean()
        return dh + dw


# ─────────────────────────────────────────
# 5. COLOR HISTOGRAM LOSS (Your Differentiator)
# ─────────────────────────────────────────
class HistogramLoss(nn.Module):
    def __init__(self, bins=64, sigma=0.02):
        super().__init__()
        self.bins  = bins
        self.sigma = sigma
        # fixed bin centers in [-1, 1]
        self.register_buffer(
            'centers',
            torch.linspace(-1.0, 1.0, bins)
        )

    def soft_histogram(self, x):
        """
        x: (B, 1, H, W) — single channel
        returns: (bins,) normalized histogram
        """
        x_flat = x.reshape(-1, 1)                          # (N, 1)
        c      = self.centers.unsqueeze(0)                 # (1, bins)
        weights = torch.exp(
            -0.5 * ((x_flat - c) / self.sigma) ** 2
        )                                                  # (N, bins)
        hist = weights.mean(dim=0)                         # (bins,)
        hist = hist / (hist.sum() + 1e-8)                  # normalize
        return hist

    def forward(self, ab_pred, ab_real):
        loss = 0.0
        for c in range(2):                                 # a and b separately
            h_pred = self.soft_histogram(ab_pred[:, c:c+1])
            h_real = self.soft_histogram(ab_real[:, c:c+1])

            # KL divergence: sum(real * log(real / pred))
            loss += (
                h_real * torch.log(h_real / (h_pred + 1e-8) + 1e-8)
            ).sum()

        return loss


# ─────────────────────────────────────────
# 6. COMBINED LOSS MANAGER
# ─────────────────────────────────────────
class LossManager(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.huber  = HuberLoss(delta=1.0)
        self.vgg    = VGGPerceptualLoss().to(device)
        self.lsgan  = LSGANLoss()
        self.tv     = TVLoss()
        self.hist   = HistogramLoss().to(device)

    def generator_loss(self, L, ab_pred, ab_real,
                       fake_logits=None,
                       w_huber=1.0,
                       w_vgg=0.1,
                       w_gan=0.0,
                       w_tv=0.0,
                       w_hist=0.0):

        l_huber = self.huber(ab_pred, ab_real)
        l_vgg   = self.vgg(L, ab_pred, ab_real)
        l_tv    = self.tv(ab_pred)
        l_hist  = self.hist(ab_pred, ab_real)

        loss = (w_huber * l_huber +
                w_vgg   * l_vgg   +
                w_tv    * l_tv    +
                w_hist  * l_hist)

        l_gan = torch.tensor(0.0)
        if fake_logits is not None and w_gan > 0:
            l_gan = self.lsgan.generator_loss(fake_logits)
            loss += w_gan * l_gan

        return loss, {
            'huber': l_huber.item(),
            'vgg':   l_vgg.item(),
            'gan':   l_gan.item(),
            'tv':    l_tv.item(),
            'hist':  l_hist.item(),
            'total': loss.item()
        }

    def discriminator_loss(self, real_logits, fake_logits):
        return self.lsgan.discriminator_loss(real_logits, fake_logits)