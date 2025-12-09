"""
Evaluation utilities for Domain-Adaptive Stable Diffusion (DASD).
"""

import os
import torch
import numpy as np
from tqdm.auto import tqdm


# Default evaluation prompts
SATELLITE_PROMPTS = [
    "an airport with runways from above",
    "a river winding through farmland",
    "a coastal harbor with ships",
    "a highway interchange",
    "a residential neighborhood from above",
    "a stadium surrounded by parking lots",
    "a solar farm with panel arrays",
    "a bridge crossing a wide river",
    "an industrial complex with warehouses",
    "a golf course with green fairways",
    "a train station with rail tracks",
    "a circular roundabout intersection",
    "a port with container ships",
    "a dam and reservoir from above",
    "an oil refinery with storage tanks",
    "a university campus with buildings",
    "a shopping mall with parking lots",
    "a wind farm with turbines",
    "a beach coastline from above",
    "a military base with runways",
]

XRAY_PROMPTS = [
    "a chest x-ray showing lungs",
    "a chest radiograph scan",
    "a chest x-ray with rib cage",
    "a thoracic x-ray frontal view",
    "a chest x-ray of a patient",
    "a chest x-ray showing pneumonia signs",
    "a frontal chest radiograph scan",
    "a chest x-ray with clear airways",
    "a thoracic x-ray showing spine",
    "a chest x-ray of adult patient",
    "a lung radiograph showing bronchi",
    "a chest scan with diaphragm visible",
    "a chest x-ray with heart silhouette",
    "a chest radiograph showing clavicles",
    "a chest x-ray anterior view",
    "a chest x-ray with mediastinum",
    "a chest radiograph of healthy lungs",
    "a thoracic scan showing vertebrae",
    "a chest x-ray with shoulder bones",
    "a chest radiograph posterior view",
]

DOMAIN_PROMPTS = {
    "satellite": SATELLITE_PROMPTS,
    "xray": XRAY_PROMPTS,
}


class CLIPEvaluator:
    """CLIP-based evaluation for text-image alignment."""

    def __init__(self, device="cuda"):
        """Initialize CLIP evaluator.

        Args:
            device: Device to run CLIP model on
        """
        from transformers import CLIPProcessor, CLIPModel

        print("Loading CLIP model for evaluation...")
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        print("  CLIP model loaded")

    def compute_clip_score(self, image, prompt):
        """Compute CLIP similarity score between image and prompt.

        Args:
            image: PIL Image
            prompt: Text prompt

        Returns:
            float: CLIP similarity score in [0, 1]
        """
        inputs = self.processor(
            text=[prompt],
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Normalize to [0, 1] range
        score = outputs.logits_per_image.item() / 100
        return score


def evaluate_dasd(
    dasd_pipeline,
    base_pipeline,
    evaluator,
    domain_tokens,
    prompts=None,
    seed=42,
):
    """Evaluate DASD against baseline Stable Diffusion.

    Args:
        dasd_pipeline: CalibratedDASDPipeline instance
        base_pipeline: Base StableDiffusionPipeline
        evaluator: CLIPEvaluator instance
        domain_tokens: Dict mapping domain names to token strings
        prompts: Optional dict of domain -> prompt list
        seed: Random seed for reproducibility

    Returns:
        dict: Results for each domain with base_sd and dasd scores
    """
    if prompts is None:
        prompts = DOMAIN_PROMPTS

    # Style suffixes for base SD
    style_suffixes = {
        "satellite": ", satellite imagery, aerial view, top-down perspective",
        "xray": ", x-ray radiograph, medical imaging, grayscale, chest scan",
    }

    results = {domain: {"base_sd": [], "dasd": []} for domain in prompts.keys()}

    print("\nEvaluating CLIP-Scores...")

    for domain, domain_prompts in prompts.items():
        print(f"\n=== {domain.upper()} Domain ===")
        token = domain_tokens.get(domain)

        for i, prompt in enumerate(tqdm(domain_prompts, desc=f"  {domain}")):
            # Base SD with style suffix
            style_prompt = prompt + style_suffixes.get(domain, "")
            generator = torch.Generator(device=base_pipeline.device).manual_seed(seed + i)
            base_img = base_pipeline(
                style_prompt,
                generator=generator,
                num_inference_steps=30
            ).images[0]
            base_score = evaluator.compute_clip_score(base_img, prompt)
            results[domain]["base_sd"].append(base_score)

            # DASD with domain token
            dasd_prompt = f"{prompt} {token}"
            dasd_img, _, _ = dasd_pipeline.generate(dasd_prompt, seed=seed + i)
            dasd_score = evaluator.compute_clip_score(dasd_img, prompt)
            results[domain]["dasd"].append(dasd_score)

    return results


def print_results(results):
    """Print evaluation results summary.

    Args:
        results: Dict from evaluate_dasd
    """
    print("\n" + "=" * 60)
    print("CLIP-Score Results")
    print("=" * 60)

    all_base = []
    all_dasd = []

    for domain, scores in results.items():
        base_mean = np.mean(scores["base_sd"])
        base_std = np.std(scores["base_sd"])
        dasd_mean = np.mean(scores["dasd"])
        dasd_std = np.std(scores["dasd"])

        print(f"\n{domain.upper()} Domain:")
        print(f"  Base SD: {base_mean:.4f} +/- {base_std:.4f}")
        print(f"  DASD:    {dasd_mean:.4f} +/- {dasd_std:.4f}")

        all_base.extend(scores["base_sd"])
        all_dasd.extend(scores["dasd"])

    print(f"\nOverall Average:")
    print(f"  Base SD: {np.mean(all_base):.4f}")
    print(f"  DASD:    {np.mean(all_dasd):.4f}")


def plot_results(results, output_dir):
    """Plot and save evaluation results.

    Args:
        results: Dict from evaluate_dasd
        output_dir: Directory to save plot
    """
    try:
        import matplotlib.pyplot as plt

        domains = list(results.keys())
        n_domains = len(domains)

        fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 5))
        if n_domains == 1:
            axes = [axes]

        methods = ["Base SD", "DASD"]
        colors = ["#4285F4", "#34A853"]

        for i, domain in enumerate(domains):
            scores = results[domain]
            means = [np.mean(scores["base_sd"]), np.mean(scores["dasd"])]
            stds = [np.std(scores["base_sd"]), np.std(scores["dasd"])]

            axes[i].bar(methods, means, yerr=stds, capsize=5, color=colors)
            axes[i].set_title(f"{domain.upper()} Domain", fontsize=12, fontweight="bold")
            axes[i].set_ylabel("CLIP-Score")
            axes[i].set_ylim(0, max(means) * 1.3)

        plt.suptitle("CLIP-Score: Base SD vs DASD", fontsize=14, fontweight="bold")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "clip_scores.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nSaved evaluation plot to {plot_path}")

    except ImportError:
        print("  Warning: matplotlib not available, skipping plot")


def generate_comparison_grid(
    dasd_pipeline,
    base_pipeline,
    domain_tokens,
    output_dir,
    n_samples=4,
):
    """Generate and save a comparison grid of Base SD vs DASD.

    Args:
        dasd_pipeline: CalibratedDASDPipeline instance
        base_pipeline: Base StableDiffusionPipeline
        domain_tokens: Dict mapping domain names to token strings
        output_dir: Directory to save images
        n_samples: Number of samples per domain
    """
    try:
        import matplotlib.pyplot as plt

        prompts = {
            "satellite": SATELLITE_PROMPTS[:n_samples],
            "xray": XRAY_PROMPTS[:n_samples],
        }

        style_suffixes = {
            "satellite": ", satellite imagery, aerial view",
            "xray": ", x-ray radiograph, medical imaging",
        }

        domains = list(prompts.keys())
        n_cols = len(domains) * 2  # Base + DASD for each domain

        fig, axes = plt.subplots(n_samples, n_cols, figsize=(5 * n_cols, 5 * n_samples))

        col_titles = []
        for domain in domains:
            col_titles.extend([f"Base SD\n({domain})", f"DASD\n({domain})"])

        print("\nGenerating comparison grid...")

        for row in range(n_samples):
            col = 0
            for domain in domains:
                prompt = prompts[domain][row]
                token = domain_tokens.get(domain, "")

                # Base SD
                style_prompt = prompt + style_suffixes.get(domain, "")
                base_img = base_pipeline(style_prompt, num_inference_steps=30).images[0]
                axes[row, col].imshow(base_img)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(col_titles[col], fontsize=10, fontweight="bold")
                col += 1

                # DASD
                dasd_prompt = f"{prompt} {token}"
                dasd_img, _, _ = dasd_pipeline.generate(dasd_prompt)
                axes[row, col].imshow(dasd_img)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(col_titles[col], fontsize=10, fontweight="bold")
                col += 1

            print(f"  Row {row + 1}/{n_samples} complete")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "comparison_grid.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved comparison grid to {plot_path}")

    except ImportError:
        print("  Warning: matplotlib not available, skipping comparison grid")
