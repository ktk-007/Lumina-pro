import torch
import torch.nn as nn
import timm

# ─────────────────────────────────────────
# 1. EFFICIENT CHANNEL ATTENTION (ECA)
# ─────────────────────────────────────────
class ECA(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.avg    = nn.AdaptiveAvgPool2d(1)
        self.conv   = nn.Conv1d(1, 1, k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)                          # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)      # (B, 1, C)
        y = self.conv(y)                         # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)    # (B, C, 1, 1)
        return x * self.sigmoid(y)


# ─────────────────────────────────────────
# 2. CONVNEXT-STYLE REFINEMENT BLOCK
# ─────────────────────────────────────────
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3,
                      groups=in_ch, bias=False),        # depthwise
            nn.GroupNorm(1, in_ch),                     # layernorm equiv
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1),# pointwise expand
            nn.GELU(),
            nn.Conv2d(out_ch * 4, out_ch, kernel_size=1) # pointwise contract
        )
        # residual projection if channels change
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)


# ─────────────────────────────────────────
# 3. SEMANTIC BOTTLENECK
# ─────────────────────────────────────────
class SemanticBottleneck(nn.Module):
    def __init__(self, in_ch=768, embed_dim=512):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_ch, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):                             # x: (B, 768, H, W)
        vec = self.gap(x).flatten(1)                  # (B, 768)
        vec = self.proj(vec)                           # (B, 512)
        vec = vec[:, :, None, None].expand(           # (B, 512, H, W)
            -1, -1, x.shape[2], x.shape[3]
        )
        return torch.cat([x, vec], dim=1)             # (B, 1280, H, W)


# ─────────────────────────────────────────
# 4. DECODER BLOCK
# ─────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # PixelShuffle: needs in_ch -> out_ch*4 then rearrange to out_ch at 2x res
        self.upsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1),
            nn.PixelShuffle(2)                        # (B, out_ch, 2H, 2W)
        )
        self.eca    = ECA(out_ch + skip_ch)
        self.refine = ConvNeXtBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.upsample(x)                          # upsample
        if skip is not None:
            x = torch.cat([x, skip], dim=1)              # merge skip
        x = self.eca(x)                               # channel attention
        return self.refine(x)                         # refine


# ─────────────────────────────────────────
# 5. FULL GENERATOR
# ─────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.encoder = timm.create_model(
            'convnext_tiny.fb_in22k',
            pretrained=False,
            features_only=True
        )

        # freeze encoder for Phase 1
        for p in self.encoder.parameters():
            p.requires_grad = False

        # --- Bottleneck ---
        self.bottleneck = SemanticBottleneck(in_ch=768, embed_dim=512)

        # --- Decoder ---
        # D4: bottleneck(1280) + S3(384) -> 256
        self.d4 = DecoderBlock(in_ch=1280, skip_ch=384, out_ch=256)
        # D3: 256 + S2(192) -> 128
        self.d3 = DecoderBlock(in_ch=256,  skip_ch=192, out_ch=128)
        # D2: 128 + S1(96)  -> 64
        self.d2 = DecoderBlock(in_ch=128,  skip_ch=96,  out_ch=64)
        # D1: 64 -> 32 (128x128)
        self.d1 = DecoderBlock(in_ch=64,   skip_ch=0,   out_ch=32)
        # D0: 32 -> 16 (256x256)
        self.d0 = DecoderBlock(in_ch=32,   skip_ch=0,   out_ch=16)

        # --- Output head ---
        self.head = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Tanh()                                 # output in [-1, 1]
        )

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("Encoder unfrozen.")

    def forward(self, L):
        # L: (B, 1, H, W) -> replicate to 3 channels for ConvNeXt
        x = L.repeat(1, 3, 1, 1)                     # (B, 3, H, W)

        # --- Encode ---
        features = self.encoder(x)
        s1 = features[0]   # (B, 96,  64, 64)
        s2 = features[1]   # (B, 192, 32, 32)
        s3 = features[2]   # (B, 384, 16, 16)
        s4 = features[3]   # (B, 768,  8,  8)

        # --- Bottleneck ---
        b = self.bottleneck(s4)                       # (B, 1280, 8, 8)

        # --- Decode ---
        x = self.d4(b,  s3)                           # (B, 256, 16, 16)
        x = self.d3(x,  s2)                           # (B, 128, 32, 32)
        x = self.d2(x,  s1)                           # (B,  64, 64, 64)
        x = self.d1(x)                                # (B,  32, 128, 128)
        x = self.d0(x)                                # (B,  16, 256, 256)

        return self.head(x)                           # (B,   2, 256, 256)


# ─────────────────────────────────────────
# 6. PATCHGAN DISCRIMINATOR
# ─────────────────────────────────────────
class PatchGAN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3,   64,  norm=False),   # L(1) + ab(2) = 3
            *block(64,  128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1) # patch output
        )

    def forward(self, L, ab):
        x = torch.cat([L, ab], dim=1)      # (B, 3, H, W)
        return self.model(x)