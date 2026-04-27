import torch
from model.architecture import Generator, PatchGAN

G = Generator().cuda()
D = PatchGAN().cuda()

L  = torch.randn(2, 1, 256, 256).cuda()
ab = torch.randn(2, 2, 256, 256).cuda()

ab_pred = G(L)
print("Generator output:", ab_pred.shape)   # (2, 2, 256, 256)
print("ab range:", ab_pred.min().item(), "->", ab_pred.max().item())

patch = D(L, ab_pred)
print("Discriminator output:", patch.shape) # (2, 1, 30, 30)

print("Architecture OK")