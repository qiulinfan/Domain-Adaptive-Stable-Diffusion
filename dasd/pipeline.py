"""
Inference pipeline for Domain-Adaptive Stable Diffusion (DASD).
"""

import os
import json
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
)
from peft import PeftModel


def get_scheduler(scheduler_name, model_id):
    """Get the appropriate scheduler based on domain calibration.

    Args:
        scheduler_name: Name of scheduler (DDPM, DDIM, PNDM, Euler)
        model_id: Model ID for loading scheduler config

    Returns:
        Scheduler instance
    """
    schedulers = {
        "DDPM": DDPMScheduler,
        "DDIM": DDIMScheduler,
        "PNDM": PNDMScheduler,
        "Euler": EulerDiscreteScheduler,
    }
    scheduler_class = schedulers.get(scheduler_name, DDPMScheduler)
    return scheduler_class.from_pretrained(model_id, subfolder="scheduler")


def detect_domain_from_prompt(prompt, domain_tokens):
    """Detect which domain token is in the prompt.

    Args:
        prompt: Text prompt
        domain_tokens: Dict mapping domain names to token strings

    Returns:
        Domain name or 'default' if none found
    """
    for domain, token in domain_tokens.items():
        if token in prompt:
            return domain
    return "default"


class CalibratedDASDPipeline:
    """
    DASD Pipeline with domain-specific inference calibration.

    Automatically adjusts CFG scale, sampling steps, and scheduler
    based on the detected domain in the prompt.

    Supports optional automatic domain classification for users who
    don't know about domain tokens.
    """

    def __init__(self, pipe, calibration_config, domain_tokens, model_id):
        """Initialize calibrated pipeline.

        Args:
            pipe: StableDiffusionPipeline instance
            calibration_config: Dict of domain-specific generation settings
            domain_tokens: Dict mapping domain names to token strings
            model_id: Base model ID for scheduler loading
        """
        self.pipe = pipe
        self.calibration = calibration_config
        self.domain_tokens = domain_tokens
        self.model_id = model_id
        self._classifier = None  # Lazy initialization

    @property
    def classifier(self):
        """Lazy-load domain classifier when first needed."""
        if self._classifier is None:
            from .classifier import DomainClassifier
            self._classifier = DomainClassifier(
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                domain_tokens=self.domain_tokens,
                device=self.pipe.device,
            )
        return self._classifier

    def enable_auto_domain(self):
        """Enable automatic domain classification (pre-load classifier)."""
        _ = self.classifier  # Force initialization
        print("  Domain classifier enabled")

    def generate(self, prompt, manual_settings=None, seed=None, auto_domain=False):
        """Generate image with automatic domain-specific calibration.

        Args:
            prompt: Text prompt (with or without domain token)
            manual_settings: Optional dict to override calibration
            seed: Optional random seed for reproducibility
            auto_domain: If True and no domain token in prompt, auto-classify
                        and inject the appropriate domain token

        Returns:
            tuple: (PIL Image, detected domain, settings used)
        """
        # Check if domain token already in prompt
        domain = detect_domain_from_prompt(prompt, self.domain_tokens)

        # Auto-classify and inject token if requested and no token found
        if auto_domain and domain == "default":
            modified_prompt, classified_domain, score = self.classifier.inject_token(
                prompt, position="suffix"
            )
            if classified_domain is not None:
                prompt = modified_prompt
                domain = classified_domain

        # Get calibration settings
        settings = self.calibration.get(domain, self.calibration["default"]).copy()

        # Override with manual settings if provided
        if manual_settings:
            settings.update(manual_settings)

        # Set scheduler based on domain
        scheduler_name = settings.get("scheduler", "DDPM")
        self.pipe.scheduler = get_scheduler(scheduler_name, self.model_id)

        # Setup generator if seed provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        # Generate with calibrated settings
        image = self.pipe(
            prompt,
            guidance_scale=settings["guidance_scale"],
            num_inference_steps=settings["num_inference_steps"],
            generator=generator,
        ).images[0]

        return image, domain, settings

    def generate_with_auto_domain(self, prompt, seed=None):
        """Generate image with automatic domain detection.

        Convenience method that always uses auto_domain=True.

        Args:
            prompt: Text prompt WITHOUT domain token
            seed: Optional random seed

        Returns:
            tuple: (PIL Image, detected domain, settings, modified prompt)
        """
        # Classify the prompt
        classified_domain, score = self.classifier.classify(prompt)
        modified_prompt, _, _ = self.classifier.inject_token(
            prompt, domain=classified_domain, position="suffix"
        )

        # Generate
        image, domain, settings = self.generate(modified_prompt, seed=seed)

        return image, domain, settings, modified_prompt

    def classify_prompt(self, prompt):
        """Classify a prompt without generating an image.

        Args:
            prompt: Text prompt to classify

        Returns:
            dict: Classification results with domain, score, and all similarities
        """
        domain, score = self.classifier.classify(prompt)
        all_sims = self.classifier.get_all_similarities(prompt)

        return {
            "best_domain": domain,
            "best_score": score,
            "all_similarities": all_sims,
            "suggested_prompt": f"{prompt} {self.domain_tokens[domain]}"
        }

    def generate_comparison(self, base_prompt, domains=None, seed=None):
        """Generate the same prompt across all domains with calibrated settings.

        Args:
            base_prompt: Base text prompt without domain token
            domains: List of domains to generate (default: all)
            seed: Optional random seed

        Returns:
            list: List of dicts with domain, image, settings, prompt
        """
        if domains is None:
            domains = list(self.domain_tokens.keys())

        results = []
        for domain in domains:
            token = self.domain_tokens[domain]
            prompt = f"{base_prompt} {token}"
            image, _, settings = self.generate(prompt, seed=seed)
            results.append({
                "domain": domain,
                "image": image,
                "settings": settings,
                "prompt": prompt
            })
        return results


def load_trained_pipeline(checkpoint_dir, config, device=None):
    """Load a trained DASD pipeline from checkpoint.

    Args:
        checkpoint_dir: Directory containing saved checkpoint
        config: Configuration object
        device: Device to load model on (default: from config)

    Returns:
        CalibratedDASDPipeline instance
    """
    if device is None:
        device = config.DEVICE

    print(f"Loading trained model from {checkpoint_dir}...")

    # Determine dtype
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        config.MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)

    # Load LoRA weights
    pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint_dir)
    print("  Loaded LoRA weights")

    # Load domain tokens
    tokens_path = os.path.join(checkpoint_dir, "domain_tokens.pt")
    tokens_data = torch.load(tokens_path, map_location=device)

    domain_tokens = {}
    for domain, data in tokens_data.items():
        token_str = data["token_string"]
        token_emb = data["token_embedding"]
        domain_tokens[domain] = token_str

        # Add token to tokenizer and encoder
        pipe.tokenizer.add_tokens([token_str])
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        tok_id = pipe.tokenizer.convert_tokens_to_ids(token_str)

        with torch.no_grad():
            pipe.text_encoder.get_input_embeddings().weight[tok_id] = token_emb.to(device)

        print(f"  Loaded domain token: {token_str}")

    # Load inference calibration config
    calib_path = os.path.join(checkpoint_dir, "inference_calibration.json")
    with open(calib_path, "r") as f:
        calibration_config = json.load(f)

    # Enable memory optimization
    pipe.enable_attention_slicing()

    # Create calibrated pipeline
    dasd_pipeline = CalibratedDASDPipeline(
        pipe, calibration_config, domain_tokens, config.MODEL_ID
    )

    print("  Pipeline ready")
    return dasd_pipeline


def save_checkpoint(save_dir, unet, text_encoder, tokenizer, token_ids, config):
    """Save training checkpoint.

    Args:
        save_dir: Directory to save checkpoint
        unet: Trained UNet with LoRA
        text_encoder: Text encoder with domain tokens
        tokenizer: Tokenizer with added tokens
        token_ids: Dict mapping domain names to token IDs
        config: Configuration object
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving checkpoint to {save_dir}...")

    # Save LoRA weights
    unet.save_pretrained(save_dir)
    print("  Saved LoRA weights")

    # Save domain tokens
    tokens_data = {}
    for domain, tid in token_ids.items():
        tokens_data[domain] = {
            "token_embedding": text_encoder.get_input_embeddings().weight[tid].detach().cpu(),
            "token_string": config.TARGET_DOMAINS[domain]["token"]
        }
    torch.save(tokens_data, os.path.join(save_dir, "domain_tokens.pt"))
    print("  Saved domain tokens")

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    print("  Saved tokenizer")

    # Save inference calibration config
    with open(os.path.join(save_dir, "inference_calibration.json"), "w") as f:
        json.dump(config.INFERENCE_CALIBRATION, f, indent=2)
    print("  Saved inference calibration config")
