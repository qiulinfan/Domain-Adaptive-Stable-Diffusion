"""
Dataset classes for Domain-Adaptive Stable Diffusion (DASD).
"""

import random
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


class BaseImageDataset(Dataset):
    """Base dataset class with common preprocessing."""

    def __init__(self, tokenizer, max_length=77, image_size=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def tokenize(self, caption):
        """Tokenize a caption string."""
        return self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)


class SourceDataset(BaseImageDataset):
    """Source domain dataset (natural images) for MMD alignment."""

    def __init__(self, config_dict, tokenizer):
        super().__init__(tokenizer)
        self.config_dict = config_dict
        print(f"Loading SOURCE dataset: {config_dict['dataset_id']}")

        try:
            self.data = load_dataset(config_dict["dataset_id"], split="train")
        except Exception as e:
            print(f"  Warning: Failed to load dataset, using cifar10 as fallback")
            self.data = load_dataset("cifar10", split="train")
            self.config_dict["caption_key"] = None

        max_samples = config_dict.get("max_samples", 300)
        if len(self.data) > max_samples:
            indices = random.sample(range(len(self.data)), max_samples)
            self.data = self.data.select(indices)

        print(f"  Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get image
        image = item.get("image") or item.get("img")
        if hasattr(image, 'convert'):
            image = image.convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")

        # Get caption
        caption_key = self.config_dict.get("caption_key")
        if caption_key:
            caption = item.get(caption_key, "a photo")
            if isinstance(caption, list):
                caption = caption[0]
        else:
            caption = "a natural photograph"

        return {
            "pixel_values": self.transforms(image),
            "input_ids": self.tokenize(caption),
            "is_source": True
        }


class TargetDomainDataset(BaseImageDataset):
    """Target domain dataset with domain token appended to captions."""

    # Default captions for domains without caption_key
    DEFAULT_CAPTIONS = {
        "xray": "a chest x-ray radiograph image",
        "satellite": "a satellite aerial view image",
    }

    def __init__(self, domain_name, domain_config, tokenizer):
        super().__init__(tokenizer)
        self.domain_name = domain_name
        self.domain_config = domain_config
        self.token = domain_config["token"]

        print(f"Loading TARGET dataset [{domain_name}]: {domain_config['dataset_id']}")
        self.data = load_dataset(domain_config["dataset_id"], split="train")

        max_samples = domain_config.get("max_samples", 400)
        if len(self.data) > max_samples:
            indices = random.sample(range(len(self.data)), max_samples)
            self.data = self.data.select(indices)

        print(f"  Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].convert("RGB")

        # Get caption
        caption_key = self.domain_config.get("caption_key")
        if caption_key:
            caption = item.get(caption_key, "an image")
            if isinstance(caption, list):
                caption = caption[0]
        else:
            caption = self.DEFAULT_CAPTIONS.get(self.domain_name, "a domain-specific image")

        # Append domain token
        caption_with_token = f"{caption} {self.token}"

        return {
            "pixel_values": self.transforms(image),
            "input_ids": self.tokenize(caption_with_token),
            "domain": self.domain_name
        }


class MixedDomainBatchSampler:
    """Sampler that provides paired batches with 3:1 target:source ratio."""

    def __init__(self, source_dataset, target_datasets):
        self.source = source_dataset
        self.targets = target_datasets
        self.target_names = list(target_datasets.keys())
        self.source_idx = 0
        self.target_indices = {name: 0 for name in self.target_names}
        self.current_target = 0

    def get_paired_batch(self):
        """Get a paired batch of target and source samples."""
        # Select current target domain (round-robin)
        domain = self.target_names[self.current_target]
        target_ds = self.targets[domain]

        # Get samples
        target_sample = target_ds[self.target_indices[domain] % len(target_ds)]
        source_sample = self.source[self.source_idx % len(self.source)]

        # Update indices
        self.target_indices[domain] += 1
        self.source_idx += 1
        self.current_target = (self.current_target + 1) % len(self.target_names)

        return target_sample, source_sample, domain

    def reset(self):
        """Reset all indices."""
        self.source_idx = 0
        self.target_indices = {name: 0 for name in self.target_names}
        self.current_target = 0


def create_datasets(config, tokenizer):
    """Create source and target datasets.

    Args:
        config: Configuration object
        tokenizer: CLIP tokenizer

    Returns:
        tuple: (source_dataset, target_datasets_dict, sampler)
    """
    print("\nLoading datasets...")

    source_ds = SourceDataset(config.SOURCE_DATASET, tokenizer)
    target_ds = {
        name: TargetDomainDataset(name, cfg, tokenizer)
        for name, cfg in config.TARGET_DOMAINS.items()
    }
    sampler = MixedDomainBatchSampler(source_ds, target_ds)

    total_target = sum(len(d) for d in target_ds.values())
    print(f"\nDataset summary:")
    print(f"  Source samples: {len(source_ds)}")
    print(f"  Target samples: {total_target}")

    return source_ds, target_ds, sampler
