#!/usr/bin/env python3
"""
Domain-Adaptive Stable Diffusion (DASD) - Main Entry Point

Usage:
    python run.py train              # Train the model
    python run.py evaluate           # Evaluate trained model
    python run.py generate           # Generate images with trained model
    python run.py classify           # Classify prompts into domains
    python run.py all                # Train, evaluate, and generate

Options:
    --output_dir PATH      Output directory (default: dasd_output)
    --checkpoint PATH      Checkpoint directory for evaluation/generation
    --max_steps N          Maximum training steps (default: 1500)
    --prompt TEXT          Custom prompt for generation/classification
    --domain DOMAIN        Domain for generation (satellite, xray)
    --num_images N         Number of images to generate (default: 4)
    --auto-domain          Auto-detect domain from prompt (no token needed)
"""

import argparse
import os
import sys
import torch

from dasd.config import Config
from dasd.datasets import create_datasets
from dasd.models import setup_training
from dasd.train import train, save_training_results, plot_losses
from dasd.pipeline import load_trained_pipeline
from dasd.evaluate import (
    CLIPEvaluator,
    evaluate_dasd,
    print_results,
    plot_results,
    generate_comparison_grid,
)


def run_training(config):
    """Run the training pipeline."""
    print("\n" + "=" * 60)
    print("DASD Training Pipeline")
    print("=" * 60)

    # Setup configuration
    config.setup()

    # Setup models and optimizers
    models = setup_training(config)

    # Create datasets
    source_ds, target_ds, sampler = create_datasets(config, models["tokenizer"])

    # Train
    losses_log = train(config, models, sampler, models["token_ids"])

    # Save results
    save_training_results(config, models, models["token_ids"], losses_log)

    # Plot losses
    plot_losses(losses_log, config.OUTPUT_DIR)

    print(f"\nTraining complete! Checkpoint saved to {config.OUTPUT_DIR}/checkpoint-final")

    return models, losses_log


def run_evaluation(config, checkpoint_dir=None):
    """Run evaluation on trained model."""
    print("\n" + "=" * 60)
    print("DASD Evaluation Pipeline")
    print("=" * 60)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoint-final")

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint not found at {checkpoint_dir}")
        print("Please train the model first with: python run.py train")
        return None

    # Load trained pipeline
    dasd_pipeline = load_trained_pipeline(checkpoint_dir, config)

    # Load base pipeline for comparison
    print("\nLoading base Stable Diffusion for comparison...")
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if config.DEVICE == "cuda" else torch.float32
    base_pipeline = StableDiffusionPipeline.from_pretrained(
        config.MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    ).to(config.DEVICE)
    base_pipeline.enable_attention_slicing()

    # Initialize evaluator
    evaluator = CLIPEvaluator(device=config.DEVICE)

    # Run evaluation
    results = evaluate_dasd(
        dasd_pipeline,
        base_pipeline,
        evaluator,
        dasd_pipeline.domain_tokens,
    )

    # Print and plot results
    print_results(results)
    plot_results(results, config.OUTPUT_DIR)

    # Generate comparison grid
    generate_comparison_grid(
        dasd_pipeline,
        base_pipeline,
        dasd_pipeline.domain_tokens,
        config.OUTPUT_DIR,
    )

    return results


def run_generation(config, checkpoint_dir=None, prompts=None, domain=None,
                   num_images=4, auto_domain=False):
    """Generate images with trained model."""
    print("\n" + "=" * 60)
    print("DASD Image Generation")
    print("=" * 60)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoint-final")

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint not found at {checkpoint_dir}")
        print("Please train the model first with: python run.py train")
        return

    # Load trained pipeline
    dasd_pipeline = load_trained_pipeline(checkpoint_dir, config)

    # Enable auto-domain if requested
    if auto_domain:
        print("\nEnabling automatic domain classification...")
        dasd_pipeline.enable_auto_domain()

    # Default prompts if none provided
    if prompts is None:
        from dasd.evaluate import SATELLITE_PROMPTS, XRAY_PROMPTS

        if auto_domain:
            # Use prompts without domain specification for auto-detection
            prompts = [
                "airport with runways from above",
                "chest scan showing lungs",
                "river through farmland aerial view",
                "medical radiograph of chest",
            ][:num_images]
        elif domain == "satellite":
            prompts = SATELLITE_PROMPTS[:num_images]
        elif domain == "xray":
            prompts = XRAY_PROMPTS[:num_images]
        else:
            # Mix of both domains
            prompts = []
            for i in range(num_images // 2):
                prompts.append(("satellite", SATELLITE_PROMPTS[i]))
                prompts.append(("xray", XRAY_PROMPTS[i]))

    # Generate images
    output_subdir = os.path.join(config.OUTPUT_DIR, "generated")
    os.makedirs(output_subdir, exist_ok=True)

    print(f"\nGenerating {len(prompts)} images...")
    if auto_domain:
        print("(Using automatic domain classification)")

    for i, item in enumerate(prompts):
        if isinstance(item, tuple):
            dom, prompt = item
            token = dasd_pipeline.domain_tokens.get(dom, "")
            full_prompt = f"{prompt} {token}"
            image, detected_domain, settings = dasd_pipeline.generate(full_prompt)
        elif auto_domain:
            # Use auto-domain detection
            image, detected_domain, settings, full_prompt = dasd_pipeline.generate_with_auto_domain(item)
            dom = detected_domain
            prompt = item
            print(f"  [{i + 1}/{len(prompts)}] \"{prompt[:40]}...\" -> detected: {detected_domain}")
        else:
            dom = domain or "satellite"
            prompt = item
            token = dasd_pipeline.domain_tokens.get(dom, "")
            full_prompt = f"{prompt} {token}"
            image, detected_domain, settings = dasd_pipeline.generate(full_prompt)
            print(f"  [{i + 1}/{len(prompts)}] {prompt[:50]}...")

        # Save image
        filename = f"{dom}_{i + 1:02d}.png"
        image.save(os.path.join(output_subdir, filename))

    print(f"\nGenerated images saved to {output_subdir}")


def run_classify(config, checkpoint_dir=None, prompts=None):
    """Classify prompts into domains using the domain classifier."""
    print("\n" + "=" * 60)
    print("DASD Domain Classification")
    print("=" * 60)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoint-final")

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint not found at {checkpoint_dir}")
        print("Please train the model first with: python run.py train")
        return

    # Load trained pipeline
    dasd_pipeline = load_trained_pipeline(checkpoint_dir, config)

    # Default test prompts if none provided
    if prompts is None:
        prompts = [
            "airport with runways",
            "chest scan showing healthy lungs",
            "satellite view of a city",
            "medical x-ray of ribcage",
            "aerial photo of farmland",
            "radiograph showing pneumonia",
            "top-down view of harbor",
            "thoracic scan of patient",
        ]

    print("\nClassifying prompts...\n")
    print("-" * 70)
    print(f"{'Prompt':<40} {'Domain':<12} {'Score':<8}")
    print("-" * 70)

    for prompt in prompts:
        result = dasd_pipeline.classify_prompt(prompt)

        print(f"{prompt[:38]:<40} {result['best_domain']:<12} {result['best_score']:.4f}")

        # Show all similarities
        sims = result['all_similarities']
        for domain, score in sims.items():
            if domain != result['best_domain']:
                print(f"{'  ':<40} {domain:<12} {score:.4f}")

    print("-" * 70)
    print("\nExample usage with auto-domain:")
    print('  pipeline.generate("chest scan", auto_domain=True)')
    print('  pipeline.generate_with_auto_domain("airport aerial view")')


def main():
    parser = argparse.ArgumentParser(
        description="Domain-Adaptive Stable Diffusion (DASD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "command",
        choices=["train", "evaluate", "generate", "classify", "all"],
        help="Command to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dasd_output",
        help="Output directory for checkpoints and results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory for evaluation/generation",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for generation/classification",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["satellite", "xray"],
        default=None,
        help="Domain for generation",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--auto-domain",
        action="store_true",
        help="Automatically detect domain from prompt (no token needed)",
    )

    args = parser.parse_args()

    # Configure
    config = Config()
    config.OUTPUT_DIR = args.output_dir

    if args.max_steps is not None:
        config.MAX_STEPS = args.max_steps

    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Run command
    if args.command == "train":
        run_training(config)

    elif args.command == "evaluate":
        run_evaluation(config, args.checkpoint)

    elif args.command == "generate":
        prompts = None
        if args.prompt:
            prompts = [args.prompt]
        run_generation(
            config,
            args.checkpoint,
            prompts=prompts,
            domain=args.domain,
            num_images=args.num_images,
            auto_domain=args.auto_domain,
        )

    elif args.command == "classify":
        prompts = None
        if args.prompt:
            prompts = [args.prompt]
        run_classify(config, args.checkpoint, prompts)

    elif args.command == "all":
        # Train
        run_training(config)

        # Evaluate
        run_evaluation(config)

        # Generate
        run_generation(config, num_images=args.num_images)

    print("\nDone!")


if __name__ == "__main__":
    main()
