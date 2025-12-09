"""
Training loop for Domain-Adaptive Stable Diffusion (DASD).
"""

import os
import json
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .losses import MMDAlignmentLoss, FeatureExtractor
from .pipeline import save_checkpoint


def train_step(
    target_sample,
    source_sample,
    models,
    feat_extractor,
    mmd_loss_fn,
    config,
):
    """Execute a single training step.

    Args:
        target_sample: Target domain sample dict
        source_sample: Source domain sample dict
        models: Dict containing unet, vae, text_encoder, noise_scheduler
        feat_extractor: FeatureExtractor instance
        mmd_loss_fn: MMD loss function
        config: Configuration object

    Returns:
        tuple: (total_loss, diffusion_loss, alignment_loss)
    """
    device = config.DEVICE

    # Get pixel values and input ids
    t_pix = target_sample["pixel_values"].unsqueeze(0).to(device)
    t_ids = target_sample["input_ids"].unsqueeze(0).to(device)
    s_pix = source_sample["pixel_values"].unsqueeze(0).to(device)
    s_ids = source_sample["input_ids"].unsqueeze(0).to(device)

    # Encode images to latents
    with torch.no_grad():
        t_lat = models["vae"].encode(t_pix).latent_dist.sample() * 0.18215
        s_lat = models["vae"].encode(s_pix).latent_dist.sample() * 0.18215

    # Sample noise and timesteps
    noise_t = torch.randn_like(t_lat)
    noise_s = torch.randn_like(s_lat)
    timesteps = torch.randint(
        0, models["noise_scheduler"].config.num_train_timesteps,
        (1,), device=device
    ).long()

    # Add noise to latents
    noisy_t = models["noise_scheduler"].add_noise(t_lat, noise_t, timesteps)
    noisy_s = models["noise_scheduler"].add_noise(s_lat, noise_s, timesteps)

    # Get text embeddings
    t_emb = models["text_encoder"](t_ids)[0]
    s_emb = models["text_encoder"](s_ids)[0]

    # Forward pass for target domain
    feat_extractor.clear()
    pred_t = models["unet"](noisy_t, timesteps, t_emb).sample
    feat_t = feat_extractor.get()

    # Forward pass for source domain
    feat_extractor.clear()
    pred_s = models["unet"](noisy_s, timesteps, s_emb).sample
    feat_s = feat_extractor.get()

    # Compute losses
    loss_diff = F.mse_loss(pred_t, noise_t)
    loss_align = mmd_loss_fn(feat_s, feat_t)
    loss = (loss_diff + config.LAMBDA_ALIGN * loss_align) / config.GRADIENT_ACCUMULATION

    return loss, loss_diff.item(), loss_align.item()


def train(config, models, sampler, token_ids):
    """Main training loop.

    Args:
        config: Configuration object
        models: Dict containing all model components and optimizers
        sampler: MixedDomainBatchSampler instance
        token_ids: Dict mapping domain names to token IDs

    Returns:
        list: Training loss log
    """
    print("\n" + "=" * 60)
    print("Starting DASD Training")
    print("=" * 60)
    print(f"Loss: L = L_diffusion + {config.LAMBDA_ALIGN} * L_MMD")
    print(f"Max steps: {config.MAX_STEPS}")
    print(f"Gradient accumulation: {config.GRADIENT_ACCUMULATION}")

    device = config.DEVICE

    # Setup feature extractor and MMD loss
    feat_extractor = FeatureExtractor(models["unet"])
    mmd_loss_fn = MMDAlignmentLoss()

    # Set models to training mode
    models["unet"].train()
    models["text_encoder"].train()

    # Training state
    step = 0
    losses_log = []
    accum_diff, accum_align, accum_steps = 0, 0, 0

    # Get token embedding for gradient masking
    token_embedding = models["text_encoder"].get_input_embeddings()

    # Progress bar
    pbar = tqdm(total=config.MAX_STEPS, desc="Training")

    while step < config.MAX_STEPS:
        # Get paired batch
        target_sample, source_sample, domain = sampler.get_paired_batch()

        # Use automatic mixed precision if on CUDA
        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            loss, loss_diff, loss_align = train_step(
                target_sample,
                source_sample,
                models,
                feat_extractor,
                mmd_loss_fn,
                config,
            )

        # Backward pass
        loss.backward()

        # Accumulate losses
        accum_diff += loss_diff
        accum_align += loss_align
        accum_steps += 1

        # Optimizer step after gradient accumulation
        if accum_steps >= config.GRADIENT_ACCUMULATION:
            # Mask gradients for token embeddings (only update domain tokens)
            with torch.no_grad():
                grad = token_embedding.weight.grad
                if grad is not None:
                    mask = torch.zeros_like(grad)
                    for tid in token_ids.values():
                        mask[tid] = 1
                    token_embedding.weight.grad = grad * mask

            # Update parameters
            models["opt_unet"].step()
            models["opt_text"].step()
            models["opt_unet"].zero_grad()
            models["opt_text"].zero_grad()

            # Log losses
            avg_diff = accum_diff / accum_steps
            avg_align = accum_align / accum_steps
            losses_log.append({
                "step": step,
                "diffusion_loss": avg_diff,
                "alignment_loss": avg_align,
            })

            # Update progress bar
            pbar.set_postfix({
                "L_diff": f"{avg_diff:.4f}",
                "L_MMD": f"{avg_align:.4f}"
            })

            # Reset accumulators
            accum_diff, accum_align, accum_steps = 0, 0, 0
            step += 1
            pbar.update(1)

    pbar.close()
    print("\nTraining complete!")

    # Cleanup
    feat_extractor.remove_hook()

    return losses_log


def save_training_results(config, models, token_ids, losses_log):
    """Save all training results.

    Args:
        config: Configuration object
        models: Dict containing model components
        token_ids: Dict mapping domain names to token IDs
        losses_log: List of loss records
    """
    # Save checkpoint
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoint-final")
    save_checkpoint(
        checkpoint_dir,
        models["unet"],
        models["text_encoder"],
        models["tokenizer"],
        token_ids,
        config,
    )

    # Save training log
    log_path = os.path.join(config.OUTPUT_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(losses_log, f, indent=2)
    print(f"  Saved training log to {log_path}")


def plot_losses(losses_log, output_dir):
    """Plot and save training loss curves.

    Args:
        losses_log: List of loss records
        output_dir: Directory to save plot
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        steps = [l["step"] for l in losses_log]
        diff_losses = [l["diffusion_loss"] for l in losses_log]
        align_losses = [l["alignment_loss"] for l in losses_log]

        ax1.plot(steps, diff_losses, "b-")
        ax1.set_title("Diffusion Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, align_losses, "r-")
        ax2.set_title("MMD Alignment Loss")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_losses.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved loss plot to {plot_path}")

    except ImportError:
        print("  Warning: matplotlib not available, skipping loss plot")
