"""
Model loading and setup for Domain-Adaptive Stable Diffusion (DASD).
"""

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


def load_stable_diffusion(config):
    """Load Stable Diffusion components.

    Args:
        config: Configuration object with MODEL_ID and DEVICE

    Returns:
        dict: Dictionary containing tokenizer, scheduler, text_encoder, vae, unet
    """
    print(f"Loading Stable Diffusion from {config.MODEL_ID}...")

    tokenizer = CLIPTokenizer.from_pretrained(
        config.MODEL_ID, subfolder="tokenizer"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.MODEL_ID, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.MODEL_ID, subfolder="text_encoder"
    ).to(config.DEVICE)
    vae = AutoencoderKL.from_pretrained(
        config.MODEL_ID, subfolder="vae"
    ).to(config.DEVICE)
    unet = UNet2DConditionModel.from_pretrained(
        config.MODEL_ID, subfolder="unet"
    ).to(config.DEVICE)

    # Freeze VAE and text encoder initially
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    print("  Models loaded successfully")

    return {
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
    }


def add_domain_tokens(tokenizer, text_encoder, config):
    """Add domain-specific tokens to the tokenizer and text encoder.

    Args:
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder
        config: Configuration object with TARGET_DOMAINS

    Returns:
        dict: Mapping from domain name to token ID
    """
    print("\nAdding domain tokens...")

    # Get all domain tokens
    all_tokens = [info["token"] for info in config.TARGET_DOMAINS.values()]

    # Add tokens to tokenizer
    tokenizer.add_tokens(all_tokens)

    # Resize text encoder embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize token embeddings from init_word
    token_ids = {}
    for domain, info in config.TARGET_DOMAINS.items():
        target_id = tokenizer.convert_tokens_to_ids(info["token"])
        init_id = tokenizer.encode(info["init_word"], add_special_tokens=False)[0]

        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[target_id] = \
                text_encoder.get_input_embeddings().weight[init_id].clone()

        token_ids[domain] = target_id
        print(f"  Added {info['token']} (initialized from '{info['init_word']}')")

    return token_ids


def add_lora_to_unet(unet, config):
    """Add LoRA adapters to the UNet.

    Args:
        unet: UNet2DConditionModel
        config: Configuration object with LoRA parameters

    Returns:
        UNet with LoRA adapters
    """
    print("\nAdding LoRA adapters to UNet...")
    print(f"  Target modules: {config.LORA_TARGET_MODULES}")

    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    return unet


def setup_optimizers(unet, text_encoder, config):
    """Setup optimizers for LoRA and domain token training.

    Args:
        unet: UNet with LoRA adapters
        text_encoder: CLIP text encoder
        config: Configuration object with learning rates

    Returns:
        tuple: (unet_optimizer, text_optimizer)
    """
    # Get LoRA parameters
    lora_params = [p for p in unet.parameters() if p.requires_grad]

    # Get token embedding
    token_embedding = text_encoder.get_input_embeddings()

    # Create optimizers
    opt_unet = torch.optim.AdamW(lora_params, lr=config.LR_UNET)
    opt_text = torch.optim.AdamW([token_embedding.weight], lr=config.LR_TEXT)

    print(f"\nOptimizers configured:")
    print(f"  UNet LoRA LR: {config.LR_UNET}")
    print(f"  Text embedding LR: {config.LR_TEXT}")

    return opt_unet, opt_text


def setup_training(config):
    """Complete training setup: load models, add LoRA, add tokens, setup optimizers.

    Args:
        config: Configuration object

    Returns:
        dict: All components needed for training
    """
    # Load models
    models = load_stable_diffusion(config)

    # Add domain tokens
    token_ids = add_domain_tokens(
        models["tokenizer"],
        models["text_encoder"],
        config
    )

    # Add LoRA to UNet
    models["unet"] = add_lora_to_unet(models["unet"], config)

    # Setup optimizers
    opt_unet, opt_text = setup_optimizers(
        models["unet"],
        models["text_encoder"],
        config
    )

    return {
        **models,
        "token_ids": token_ids,
        "opt_unet": opt_unet,
        "opt_text": opt_text,
    }
