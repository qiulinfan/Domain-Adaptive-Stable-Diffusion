"""
Domain-Adaptive Stable Diffusion (DASD)

A parameter-efficient adaptation framework for cross-domain image synthesis
using Stable Diffusion with LoRA, domain tokens, and MMD alignment.
"""

from .config import Config
from .datasets import (
    SourceDataset,
    TargetDomainDataset,
    MixedDomainBatchSampler,
    create_datasets,
)
from .losses import MMDAlignmentLoss, FeatureExtractor
from .models import (
    load_stable_diffusion,
    add_domain_tokens,
    add_lora_to_unet,
    setup_training,
)
from .pipeline import (
    CalibratedDASDPipeline,
    load_trained_pipeline,
    save_checkpoint,
)
from .train import train, save_training_results, plot_losses
from .evaluate import (
    CLIPEvaluator,
    evaluate_dasd,
    print_results,
    plot_results,
)
from .classifier import (
    DomainClassifier,
    DomainClassifierExternal,
    create_classifier_from_pipeline,
)

__version__ = "1.0.0"
__author__ = "Zhanhao Liu, Huanchen Jia, Qiulin Fan, Kunlong Zhang"

__all__ = [
    # Config
    "Config",
    # Datasets
    "SourceDataset",
    "TargetDomainDataset",
    "MixedDomainBatchSampler",
    "create_datasets",
    # Losses
    "MMDAlignmentLoss",
    "FeatureExtractor",
    # Models
    "load_stable_diffusion",
    "add_domain_tokens",
    "add_lora_to_unet",
    "setup_training",
    # Pipeline
    "CalibratedDASDPipeline",
    "load_trained_pipeline",
    "save_checkpoint",
    # Classifier
    "DomainClassifier",
    "DomainClassifierExternal",
    "create_classifier_from_pipeline",
    # Training
    "train",
    "save_training_results",
    "plot_losses",
    # Evaluation
    "CLIPEvaluator",
    "evaluate_dasd",
    "print_results",
    "plot_results",
]
