"""
Domain-Adaptive Stable Diffusion (DASD) - Python API Examples

This file demonstrates how to use the DASD package programmatically.
"""

from dasd import (
    Config,
    setup_training,
    create_datasets,
    train,
    save_training_results,
    plot_losses,
    load_trained_pipeline,
    DomainClassifier,
)

# =============================================================================
# 1. TRAINING
# =============================================================================

def train_dasd():
    """Train the DASD model from scratch."""
    # Initialize configuration
    config = Config.setup()

    # Optional: Customize configuration
    config.MAX_STEPS = 1500
    config.LAMBDA_ALIGN = 0.01
    config.OUTPUT_DIR = "dasd_output"

    # Setup models (loads SD, adds LoRA, adds domain tokens)
    models = setup_training(config)

    # Create datasets (source + target domains)
    source_ds, target_ds, sampler = create_datasets(config, models["tokenizer"])

    # Train the model
    losses_log = train(config, models, sampler, models["token_ids"])

    # Save checkpoint and training results
    save_training_results(config, models, models["token_ids"], losses_log)

    # Plot training curves
    plot_losses(losses_log, config.OUTPUT_DIR)

    return models, losses_log


# =============================================================================
# 2. INFERENCE (Basic)
# =============================================================================

def generate_with_domain_token():
    """Generate images using explicit domain tokens."""
    config = Config()
    pipeline = load_trained_pipeline("dasd_output/checkpoint-final", config)

    # Generate with explicit domain token
    image, domain, settings = pipeline.generate(
        "a chest x-ray showing healthy lungs <xray>"
    )
    image.save("xray_output.png")

    # Generate satellite imagery
    image, domain, settings = pipeline.generate(
        "an airport with runways from above <satellite>"
    )
    image.save("satellite_output.png")

    return image


# =============================================================================
# 3. INFERENCE WITH AUTO-DOMAIN (Domain Classifier)
# =============================================================================

def generate_with_auto_domain():
    """Generate images with automatic domain detection.

    The domain classifier analyzes the prompt and automatically
    injects the appropriate domain token.
    """
    config = Config()
    pipeline = load_trained_pipeline("dasd_output/checkpoint-final", config)

    # Method 1: Use auto_domain flag
    # No need to specify domain token - it's detected automatically
    image, domain, settings = pipeline.generate(
        "chest scan showing pneumonia",
        auto_domain=True
    )
    print(f"Detected domain: {domain}")
    image.save("auto_xray.png")

    # Method 2: Use generate_with_auto_domain() convenience method
    image, domain, settings, modified_prompt = pipeline.generate_with_auto_domain(
        "aerial view of airport with planes"
    )
    print(f"Detected domain: {domain}")
    print(f"Modified prompt: {modified_prompt}")
    image.save("auto_satellite.png")

    return image


# =============================================================================
# 4. DOMAIN CLASSIFICATION (Without Generation)
# =============================================================================

def classify_prompts():
    """Classify prompts into domains without generating images.

    Useful for:
    - Testing which domain a prompt matches
    - Building interactive systems
    - Batch processing mixed-domain prompts
    """
    config = Config()
    pipeline = load_trained_pipeline("dasd_output/checkpoint-final", config)

    test_prompts = [
        "medical radiograph of chest",
        "satellite view of city",
        "aerial photo of farmland",
        "thoracic x-ray scan",
    ]

    for prompt in test_prompts:
        result = pipeline.classify_prompt(prompt)
        print(f"Prompt: {prompt}")
        print(f"  Best domain: {result['best_domain']}")
        print(f"  Confidence: {result['best_score']:.4f}")
        print(f"  Suggested: {result['suggested_prompt']}")
        print()


# =============================================================================
# 5. BATCH GENERATION
# =============================================================================

def batch_generate():
    """Generate multiple images with different prompts."""
    config = Config()
    pipeline = load_trained_pipeline("dasd_output/checkpoint-final", config)

    prompts = [
        ("satellite", "a river winding through farmland"),
        ("satellite", "a highway interchange from above"),
        ("xray", "a chest x-ray showing lungs"),
        ("xray", "a medical radiograph scan"),
    ]

    for i, (domain, prompt) in enumerate(prompts):
        token = pipeline.domain_tokens[domain]
        full_prompt = f"{prompt} {token}"
        image, _, _ = pipeline.generate(full_prompt, seed=42 + i)
        image.save(f"batch_{domain}_{i}.png")


# =============================================================================
# 6. COMPARISON GENERATION
# =============================================================================

def generate_domain_comparison():
    """Generate the same prompt across all domains for comparison."""
    config = Config()
    pipeline = load_trained_pipeline("dasd_output/checkpoint-final", config)

    base_prompt = "a view of buildings and roads"
    results = pipeline.generate_comparison(base_prompt, seed=42)

    for result in results:
        print(f"Domain: {result['domain']}")
        print(f"Settings: {result['settings']}")
        result['image'].save(f"comparison_{result['domain']}.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python API.py <command>")
        print("Commands: train, generate, auto, classify, batch, compare")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        train_dasd()
    elif command == "generate":
        generate_with_domain_token()
    elif command == "auto":
        generate_with_auto_domain()
    elif command == "classify":
        classify_prompts()
    elif command == "batch":
        batch_generate()
    elif command == "compare":
        generate_domain_comparison()
    else:
        print(f"Unknown command: {command}")
