"""
Configuration module for Domain-Adaptive Stable Diffusion (DASD).
"""

import os
import torch


class Config:
    """Configuration class for DASD training and inference."""

    # Base model
    MODEL_ID = "runwayml/stable-diffusion-v1-5"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # SOURCE DOMAIN (Natural Images)
    SOURCE_DATASET = {
        "dataset_id": "huggan/few-shot-art-painting",
        "image_key": "image",
        "caption_key": None,
        "is_list": False,
        "max_samples": 300
    }

    # TARGET DOMAINS
    TARGET_DOMAINS = {
        "satellite": {
            "token": "<satellite>",
            "init_word": "aerial",
            "dataset_id": "arampacha/rsicd",
            "caption_key": "captions",
            "is_list": True,
            "max_samples": 400
        },
        "xray": {
            "token": "<xray>",
            "init_word": "radiograph",
            "dataset_id": "hf-vision/chest-xray-pneumonia",
            "caption_key": None,
            "is_list": False,
            "max_samples": 400
        },
    }

    # INFERENCE CALIBRATION
    INFERENCE_CALIBRATION = {
        "satellite": {
            "guidance_scale": 7.0,
            "num_inference_steps": 30,
            "scheduler": "DDIM",
        },
        "xray": {
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "scheduler": "DDIM",
        },
        "default": {
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "scheduler": "DDPM",
        }
    }

    # Mixing Ratio (3:1 target:source)
    TARGET_SOURCE_RATIO = 3

    # Training hyperparameters
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 4
    LR_UNET = 1e-5
    LR_TEXT = 5e-5
    MAX_STEPS = 1500
    LAMBDA_ALIGN = 0.01

    # LoRA configuration
    LORA_RANK = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = [
        # Attention layers
        "to_k", "to_q", "to_v", "to_out.0",
        # Mid-level convolution layers
        "conv1", "conv2"
    ]

    # Output directory
    OUTPUT_DIR = "dasd_output"

    @classmethod
    def setup(cls):
        """Create output directory and print configuration."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        print(f"Configuration initialized")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Output directory: {cls.OUTPUT_DIR}")
        print(f"\nTarget Domains:")
        for name, info in cls.TARGET_DOMAINS.items():
            print(f"  - {name}: {info['token']}")
        return cls
