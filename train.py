import os
import numpy as np
from skimage.color import lab2rgb
# os.environ['WANDB_DISABLED'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
# from torch.cuda.amp import GradScaler, autocast  # Deprecated
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from model.dataset      import ColorizationDataset
from model.architecture import Generator, PatchGAN
from model.losses       import LossManager
# All wandb references removed for local training performance


# ─────────────────────────────────────────
# CONFIG — change these per phase
# ─────────────────────────────────────────
CFG = {
    'phase'      : 4,           # Phase 4 = Scenic Fine-tuning
    'batch_size' : 2,
    'epochs'     : 15,          # 15 epochs for maximum quality



    'resume'     : 'checkpoints/deploy_coco.pt',
    'data_dir'   : 'data/val2017',
    'scenic_dir' : 'data/landscape', # Path to Kaggle Landscape dataset
    'ckpt_dir'   : 'checkpoints',
    'img_size'   : 256,
    'num_workers': 0,

    # fine-tuning learning rates (lower for stability)
    'lr_encoder' : 5e-6,
    'lr_decoder' : 5e-5,
    'lr_disc'    : 5e-7,


    # loss weights
    'w_huber'    : 1.0,
    'w_vgg'      : 0.5,
    'w_tv'       : 0.01,
    'w_hist'     : 0.1,
    'w_gan'      : 0.0,

    # GAN ramp
    'gan_warmup' : 15,
    'gan_max'    : 1.0,

    # EMA
    'ema_decay'  : 0.999,
    'ema_start'  : 8,

    # saving
    'save_every' : 5,
}

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_lambda_gan(epoch, warmup, max_lambda):
    """GAN weight: 0 before warmup, then linear ramp to max."""
    if epoch < warmup:
        return 0.0
    total_epochs = CFG['epochs']
    ramp_epochs = max(1, total_epochs - warmup)
    progress = min((epoch - warmup) / ramp_epochs, 1.0)
    return max_lambda * progress


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  Saved -> {path}")


def load_checkpoint(path, G, D=None, opt_G=None, opt_D=None, scaler=None):
    ckpt = torch.load(path, map_location='cuda')
    if 'G' in ckpt:
        G.load_state_dict(ckpt['G'], strict=False)
    elif 'ema_G' in ckpt:
        G.load_state_dict(ckpt['ema_G'], strict=False)
        
    if D is not None and 'D' in ckpt:
        D.load_state_dict(ckpt['D'])
    if opt_G is not None and 'opt_G' in ckpt:
        opt_G.load_state_dict(ckpt['opt_G'])
    if opt_D is not None and 'opt_D' in ckpt:
        opt_D.load_state_dict(ckpt['opt_D'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    start_epoch = 0  # always start fresh for new phase
    print(f"  Resumed from {path}")
    return start_epoch


# ─────────────────────────────────────────
# PHASE 1 — Frozen encoder, Huber + VGG
# ─────────────────────────────────────────
def run_phase1(G, loader, losses, device, cfg):
    print("\n=== PHASE 1: Cold Start ===")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, G.parameters()),
        lr=cfg['lr_decoder'],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    if cfg['resume']:
        start_epoch = load_checkpoint(cfg['resume'], G, scaler=scaler)

    for epoch in range(start_epoch, cfg['epochs']):
        G.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Phase1 Epoch {epoch+1}/{cfg['epochs']}")

        for L, ab_real in loop:
            L, ab_real = L.to(device), ab_real.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                ab_pred = G(L)
                loss, log = losses.generator_loss(
                    L, ab_pred, ab_real,
                    w_huber = cfg['w_huber'],
                    w_vgg   = cfg['w_vgg'],
                    w_gan   = 0.0,
                    w_tv    = 0.0,
                    w_hist  = 0.0
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            loop.set_postfix({
                'loss' : f"{log['total']:.4f}",
                'huber': f"{log['huber']:.4f}",
                'vgg'  : f"{log['vgg']:.4f}"
            })

            # Logging local stats only

        scheduler.step()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")
        # Epoch complete

        if (epoch + 1) % cfg['save_every'] == 0:
            log_sample_images(G, loader, device, epoch, phase=1)
            save_checkpoint({
                'epoch' : epoch,
                'G'     : G.state_dict(),
                'opt_G' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, f"{cfg['ckpt_dir']}/phase1_epoch_{epoch+1:03d}.pt")

    # save final
    save_checkpoint({
        'epoch': cfg['epochs'],
        'G'    : G.state_dict(),
    }, f"{cfg['ckpt_dir']}/phase1_final.pt")
    print("Phase 1 complete.")


# ─────────────────────────────────────────
# PHASE 2 — Unfreeze encoder, add TV loss
# ─────────────────────────────────────────
def run_phase2(G, loader, losses, device, cfg):
    print("\n=== PHASE 2: Structural Refinement ===")

    G.unfreeze_encoder()

    optimizer = torch.optim.AdamW([
        {'params': G.encoder.parameters(),    'lr': cfg['lr_encoder']},
        {'params': G.bottleneck.parameters(), 'lr': cfg['lr_decoder']},
        {'params': G.d4.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d3.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d2.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d1.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.head.parameters(),       'lr': cfg['lr_decoder']},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    if cfg['resume']:
        start_epoch = load_checkpoint(cfg['resume'], G, opt_G=optimizer, scaler=scaler)

    for epoch in range(start_epoch, cfg['epochs']):
        G.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Phase2 Epoch {epoch+1}/{cfg['epochs']}")

        for L, ab_real in loop:
            L, ab_real = L.to(device), ab_real.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                ab_pred = G(L)
                loss, log = losses.generator_loss(
                    L, ab_pred, ab_real,
                    w_huber = cfg['w_huber'],
                    w_vgg   = cfg['w_vgg'],
                    w_gan   = 0.0,
                    w_tv    = cfg['w_tv'],
                    w_hist  = 0.0
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            loop.set_postfix({
                'loss' : f"{log['total']:.4f}",
                'huber': f"{log['huber']:.4f}",
                'vgg'  : f"{log['vgg']:.4f}",
                'tv'   : f"{log['tv']:.4f}"
            })

            # Logging local stats only

        avg = total_loss / len(loader)
        scheduler.step(avg)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")

        if (epoch + 1) % cfg['save_every'] == 0:
            log_sample_images(G, loader, device, epoch, phase=2)
            save_checkpoint({
                'epoch' : epoch,
                'G'     : G.state_dict(),
                'opt_G' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, f"{cfg['ckpt_dir']}/phase2_epoch_{epoch+1:03d}.pt")

    save_checkpoint({
        'epoch': cfg['epochs'],
        'G'    : G.state_dict(),
    }, f"{cfg['ckpt_dir']}/phase2_coco_final.pt")
    print("Phase 2 complete.")


# ─────────────────────────────────────────
def save_samples(G, loader, device, epoch, phase, n=4):
    import warnings
    warnings.filterwarnings('ignore')
    G.eval()
    os.makedirs('data/samples', exist_ok=True)

    with torch.no_grad():
        L, ab_real = next(iter(loader))
        n = min(n, L.shape[0])  # clamp to actual batch size
        L       = L[:n].to(device)
        ab_real = ab_real[:n].to(device)
        with torch.amp.autocast('cuda'):
            ab_pred = G(L)

    for i in range(n):
        l  = L[i, 0].cpu().numpy()    * 100.0
        ab = ab_pred[i].cpu().numpy() * 128.0
        gt = ab_real[i].cpu().numpy() * 128.0

        lab_pred = np.stack([l, ab[0], ab[1]], axis=-1)
        rgb_pred = (np.clip(lab2rgb(lab_pred), 0, 1) * 255).astype(np.uint8)

        lab_real = np.stack([l, gt[0], gt[1]], axis=-1)
        rgb_real = (np.clip(lab2rgb(lab_real), 0, 1) * 255).astype(np.uint8)

        gray     = (np.clip(l / 100.0, 0, 1) * 255).astype(np.uint8)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)

        combined = np.concatenate([gray_rgb, rgb_pred, rgb_real], axis=1)
        Image.fromarray(combined).save(
            f'data/samples/phase{phase}_epoch{epoch+1:03d}_img{i}.jpg'
        )

    print(f"  Samples saved -> data/samples/")
    G.train()


# ─────────────────────────────────────────
# PHASE 3 — GAN + Histogram + EMA (Stabilized)
# ─────────────────────────────────────────
def run_phase3(G, loader, losses, device, cfg):
    print("\n=== PHASE 3: Adversarial Refinement ===")
    print("  Stability config:")
    print(f"    lr_disc          = {cfg['lr_disc']}")
    print(f"    D train freq     = every 8 batches")
    print(f"    Instance noise   = 0.05")
    print(f"    Grad clip D      = 0.5")
    print(f"    Label smoothing  = real=0.9, fake=0.1")
    print(f"    GAN warmup       = epoch {cfg['gan_warmup']}")
    print(f"    GAN max weight   = {cfg['gan_max']}")

    D = PatchGAN().to(device)

    G.unfreeze_encoder()

    opt_G = torch.optim.AdamW([
        {'params': G.encoder.parameters(),    'lr': cfg['lr_encoder']},
        {'params': G.bottleneck.parameters(), 'lr': cfg['lr_decoder']},
        {'params': G.d4.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d3.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d2.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d1.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d0.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.head.parameters(),       'lr': cfg['lr_decoder']},
    ], weight_decay=1e-4)

    opt_D = torch.optim.AdamW(
        D.parameters(), lr=cfg['lr_disc'], weight_decay=1e-4,
        betas=(0.5, 0.999)  # lower beta1 for GAN stability
    )

    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')

    # EMA
    ema_G = AveragedModel(
        G, multi_avg_fn=get_ema_multi_avg_fn(cfg['ema_decay'])
    )

    start_epoch = 0
    if cfg['resume']:
        start_epoch = load_checkpoint(
            cfg['resume'], G, D, opt_G, opt_D, scaler_G
        )

    # Dynamic monitoring state
    d_train_freq = 8
    instance_noise = 0.05
    grad_clip_d = 0.5
    d_collapse_counter = 0
    param_change_log = []

    print(f"\n  Starting: {len(loader)} batches/epoch, {cfg['epochs']} epochs")
    print(f"  GAN activates at epoch {cfg['gan_warmup']}\n")

    for epoch in range(start_epoch, cfg['epochs']):
        G.train()
        D.train()

        total_G = 0.0
        total_D = 0.0
        d_step_count = 0
        lam_gan = get_lambda_gan(epoch, cfg['gan_warmup'], cfg['gan_max'])

        loop = tqdm(loader, desc=f"Phase3 Epoch {epoch+1}/{cfg['epochs']}")

        for batch_idx, (L, ab_real) in enumerate(loop):
            L, ab_real = L.to(device), ab_real.to(device)

            # ── D step — throttled, only when GAN is active ──
            d_loss_val = 0.0
            if batch_idx % d_train_freq == 0 and lam_gan > 0:
                opt_D.zero_grad()
                with torch.amp.autocast('cuda'):
                    ab_fake = G(L).detach()

                    # Instance noise injection
                    noise_real = torch.randn_like(ab_real) * instance_noise
                    noise_fake = torch.randn_like(ab_fake) * instance_noise

                    real_logits = D(L, ab_real + noise_real)
                    fake_logits = D(L, ab_fake + noise_fake)
                    d_loss      = losses.discriminator_loss(real_logits, fake_logits)

                scaler_D.scale(d_loss).backward()
                scaler_D.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip_d)
                scaler_D.step(opt_D)
                scaler_D.update()
                d_loss_val = d_loss.item()
                d_step_count += 1

            total_D += d_loss_val

            # ── Generator step ──────────────────────
            opt_G.zero_grad()
            with torch.amp.autocast('cuda'):
                ab_fake     = G(L)
                fake_logits = D(L, ab_fake) if lam_gan > 0 else None
                g_loss, log = losses.generator_loss(
                    L, ab_fake, ab_real,
                    fake_logits = fake_logits,
                    w_huber     = cfg['w_huber'],
                    w_vgg       = cfg['w_vgg'],
                    w_gan       = lam_gan,
                    w_tv        = cfg['w_tv'],
                    w_hist      = cfg['w_hist']
                )

            # NaN guard
            if torch.isnan(g_loss):
                print(f"\n  ⚠ NaN detected at epoch {epoch+1}, batch {batch_idx}. Stopping.")
                return

            scaler_G.scale(g_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            total_G += g_loss.item()

            loop.set_postfix({
                'G'   : f"{log['total']:.3f}",
                'D'   : f"{d_loss_val:.3f}",
                'gan' : f"{log['gan']:.3f}",
                'hist': f"{log['hist']:.4f}",
                'λgan': f"{lam_gan:.2f}"
            })

        # EMA update
        if epoch >= cfg['ema_start']:
            ema_G.update_parameters(G)

        # Epoch summary
        avg_G = total_G / len(loader)
        avg_D = total_D / max(d_step_count, 1)
        print(f"\n  Epoch {epoch+1}/{cfg['epochs']}")
        print(f"    G/total    = {avg_G:.4f}")
        print(f"    D/loss     = {avg_D:.4f}  (D steps: {d_step_count})")
        print(f"    G/gan      = {log['gan']:.4f}")
        print(f"    lambda_gan = {lam_gan:.3f}")
        print(f"    D freq     = every {d_train_freq} batches")

        # Dynamic D/loss collapse detection
        if lam_gan > 0:
            if avg_D < 0.10:
                d_collapse_counter += 1
                print(f"    ⚠ D/loss below 0.10 for {d_collapse_counter} consecutive epoch(s)")
                if d_collapse_counter >= 3:
                    old_freq = d_train_freq
                    d_train_freq = min(d_train_freq + 4, 16)
                    msg = f"Epoch {epoch+1}: D collapse ({avg_D:.4f}). D freq {old_freq} -> {d_train_freq}"
                    param_change_log.append(msg)
                    print(f"    🔧 INTERVENTION: {msg}")
                    d_collapse_counter = 0
            else:
                d_collapse_counter = 0

            if avg_G > 5.0:
                old_max = cfg['gan_max']
                cfg['gan_max'] = 0.3
                msg = f"Epoch {epoch+1}: G exploded ({avg_G:.4f}). gan_max {old_max} -> 0.3"
                param_change_log.append(msg)
                print(f"    🔧 INTERVENTION: {msg}")

            if 0.15 <= avg_D <= 0.45:
                print(f"    ✅ D/loss HEALTHY")
            elif avg_D < 0.05:
                print(f"    🔴 D/loss CRITICAL")
            elif avg_D < 0.15:
                print(f"    🟡 D/loss LOW — monitoring")
            else:
                print(f"    🟡 D/loss HIGH ({avg_D:.4f})")

        if (epoch + 1) % cfg['save_every'] == 0:
            save_samples(G, loader, device, epoch, phase=3)
            save_checkpoint({
                'epoch'  : epoch,
                'G'      : G.state_dict(),
                'D'      : D.state_dict(),
                'ema_G'  : ema_G.module.state_dict(),
                'opt_G'  : opt_G.state_dict(),
                'opt_D'  : opt_D.state_dict(),
                'scaler_G': scaler_G.state_dict(),
            }, f"{cfg['ckpt_dir']}/phase3_coco_epoch_{epoch+1:03d}.pt")

    # Save final checkpoints
    save_checkpoint({
        'ema_G'  : ema_G.module.state_dict(),
        'G'      : G.state_dict(),
        'config' : {'img_size': 256, 'version': '2.0-coco'}
    }, f"{cfg['ckpt_dir']}/phase3_coco_final.pt")

    save_checkpoint({
        'ema_G'  : ema_G.module.state_dict(),
        'config' : {'img_size': 256, 'version': '2.0-coco'}
    }, f"{cfg['ckpt_dir']}/deploy_coco.pt")

    if param_change_log:
        print("\n  === PARAMETER CHANGE LOG ===")
        for msg in param_change_log:
            print(f"    {msg}")
    else:
        print("\n  === No dynamic interventions were needed ===")

    print("\nPhase 3 complete. Deploy checkpoint saved as deploy_coco.pt")


# ─────────────────────────────────────────
# PHASE 4 — Scenic Fine-tuning (Mixed Batch)
# ─────────────────────────────────────────
def run_phase3_v2_scenic(G, losses, device, cfg):
    print("\n=== PHASE 3 v2: Scenic Fine-Tuning (Nature/Landscape) ===")
    print("  Strategy: Full Epoch on Super-Dataset (Balanced Nature + Objects)")
    
    # 1. Create Loader from the new Super-Dataset
    ds_scenic = ColorizationDataset(cfg['scenic_dir'], img_size=cfg['img_size'], augment=True)
    loader_scenic = DataLoader(ds_scenic, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])


    D = PatchGAN().to(device)
    G.unfreeze_encoder()

    opt_G = torch.optim.AdamW([
        {'params': G.encoder.parameters(),    'lr': cfg['lr_encoder']},
        {'params': G.bottleneck.parameters(), 'lr': cfg['lr_decoder']},
        {'params': G.d4.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d3.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d2.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.d1.parameters(),         'lr': cfg['lr_decoder']},
        {'params': G.head.parameters(),       'lr': cfg['lr_decoder']},
    ], weight_decay=1e-4)

    opt_D = torch.optim.AdamW(D.parameters(), lr=cfg['lr_disc'], betas=(0.5, 0.999))
    scaler_G, scaler_D = torch.amp.GradScaler('cuda'), torch.amp.GradScaler('cuda')
    ema_G = AveragedModel(G, multi_avg_fn=get_ema_multi_avg_fn(cfg['ema_decay']))

    if cfg['resume']:
        load_checkpoint(cfg['resume'], G, D, opt_G, opt_D, scaler_G)

    for epoch in range(cfg['epochs']):
        G.train(); D.train()
        loop = tqdm(loader_scenic, desc=f"Scenic Epoch {epoch+1}/{cfg['epochs']}")
        
        for batch_idx, (L, ab_real) in enumerate(loop):
            # Process EVERY image in the Super-Dataset (no skipping)
            L, ab_real = L.to(device), ab_real.to(device)


            # Standard GAN Step
            opt_G.zero_grad(); opt_D.zero_grad()
            with torch.amp.autocast('cuda'):
                ab_fake = G(L)
                fake_logits = D(L, ab_fake)
                real_logits = D(L, ab_real)
                
                # We use high histogram loss weight here to force the scenic colors
                g_loss, log = losses.generator_loss(
                    L, ab_fake, ab_real, fake_logits=fake_logits,
                    w_huber=1.0, w_vgg=0.5, w_gan=0.5, w_tv=0.01, w_hist=0.2 
                )
                d_loss = losses.discriminator_loss(real_logits, fake_logits.detach())

            scaler_G.scale(g_loss).backward()
            scaler_G.step(opt_G); scaler_G.update()
            
            scaler_D.scale(d_loss).backward()
            scaler_D.step(opt_D); scaler_D.update()
            
            ema_G.update_parameters(G)
            loop.set_postfix({'G': f"{g_loss.item():.3f}", 'D': f"{d_loss.item():.3f}"})

        save_checkpoint({
            'ema_G': ema_G.module.state_dict(),
            'G': G.state_dict(),
        }, f"{cfg['ckpt_dir']}/phase3v2_final.pt")

    print("\nScenic Fine-tuning Complete. Saved as phase3v2_final.pt")



def log_sample_images(G, loader, device, epoch, phase, n=4):
    import os
    import numpy as np
    from skimage.color import lab2rgb

    G.eval()
    with torch.no_grad():
        L, ab_real = next(iter(loader))
        L       = L[:n].to(device)
        ab_real = ab_real[:n].to(device)
        with torch.amp.autocast('cuda'):
            ab_pred = G(L)

    os.makedirs('data/samples', exist_ok=True)

    for i in range(n):
        l  = L[i, 0].cpu().numpy()      * 100.0
        ab = ab_pred[i].cpu().numpy()   * 128.0
        gt = ab_real[i].cpu().numpy()   * 128.0

        # predicted
        lab_pred = np.stack([l, ab[0], ab[1]], axis=-1)
        rgb_pred = (np.clip(lab2rgb(lab_pred), 0, 1) * 255).astype(np.uint8)

        # ground truth
        lab_real = np.stack([l, gt[0], gt[1]], axis=-1)
        rgb_real = (np.clip(lab2rgb(lab_real), 0, 1) * 255).astype(np.uint8)

        # grayscale
        gray     = (np.clip(l / 100.0, 0, 1) * 255).astype(np.uint8)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)

        # save side by side locally
        combined = np.concatenate([gray_rgb, rgb_pred, rgb_real], axis=1)
        Image.fromarray(combined).save(
            f'data/samples/phase{phase}_epoch{epoch+1:03d}_sample{i}.jpg'
        )

    print(f"  Samples saved -> data/samples/")
    G.train()

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)

    # dataset
    dataset = ColorizationDataset(
        CFG['data_dir'],
        img_size  = CFG['img_size'],
        augment   = True
    )
    loader = DataLoader(
        dataset,
        batch_size  = CFG['batch_size'],
        shuffle     = True,
        num_workers = CFG['num_workers'],
        pin_memory  = True
    )

    # model + losses
    G      = Generator().to(device)
    losses = LossManager(device)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    if   CFG['phase'] == 1:
        run_phase1(G, loader, losses, device, CFG)
    elif CFG['phase'] == 2:
        CFG['resume'] = CFG['resume'] or 'checkpoints/phase1_final.pt'
        run_phase2(G, loader, losses, device, CFG)
    elif CFG['phase'] == 3:
        CFG['resume'] = CFG['resume'] or 'checkpoints/phase2_final.pt'
        run_phase3(G, loader, losses, device, CFG)
    elif CFG['phase'] == 4:
        CFG['resume'] = CFG['resume'] or 'checkpoints/deploy_coco.pt'
        run_phase3_v2_scenic(G, losses, device, CFG)

    else:
        print("Set CFG['phase'] to 1, 2, or 3")
    print("Training session finished.")


if __name__ == '__main__':
    main()
