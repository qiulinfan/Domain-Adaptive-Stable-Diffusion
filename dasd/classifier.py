"""
Domain Classifier for Domain-Adaptive Stable Diffusion (DASD).

This module provides automatic domain detection using CLIP embeddings.
When users don't specify a domain token, the classifier analyzes the prompt
and automatically injects the most appropriate domain token.

From the design document:
> If the user is a domain-specific user, they can select "special_domain == true",
> and the domain classifier will automatically add the domain token.
"""

import torch
import torch.nn.functional as F


class DomainClassifier:
    """
    CLIP-based domain classifier for automatic domain token injection.

    Uses cosine similarity between prompt embeddings and domain token
    embeddings to determine the most appropriate domain for a given prompt.

    Example:
        prompt: "chest scan"
        classifier -> domain: xray
        inject token: "<xray>"
        -> "a <xray> chest scan"
    """

    def __init__(self, text_encoder, tokenizer, domain_tokens, device="cuda"):
        """Initialize domain classifier.

        Args:
            text_encoder: CLIP text encoder
            tokenizer: CLIP tokenizer
            domain_tokens: Dict mapping domain names to token strings
                          e.g., {"satellite": "<satellite>", "xray": "<xray>"}
            device: Device to run computations on
        """
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.domain_tokens = domain_tokens
        self.device = device
        self.domain_names = list(domain_tokens.keys())

        # Pre-compute domain token embeddings
        self.domain_embeddings = self._compute_domain_embeddings()

    def _compute_domain_embeddings(self):
        """Compute CLIP embeddings for each domain token."""
        embeddings = {}

        self.text_encoder.eval()
        with torch.no_grad():
            for domain, token in self.domain_tokens.items():
                # Encode the domain token
                inputs = self.tokenizer(
                    token,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                # Get text embedding (use pooled output)
                outputs = self.text_encoder(inputs.input_ids)
                # Use the embedding of the token position (after BOS token)
                embedding = outputs.last_hidden_state[:, 1, :]  # Shape: (1, 768)
                embeddings[domain] = F.normalize(embedding, p=2, dim=-1)

        return embeddings

    def _get_prompt_embedding(self, prompt):
        """Get CLIP embedding for a prompt.

        Args:
            prompt: Text prompt string

        Returns:
            Normalized embedding tensor
        """
        self.text_encoder.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.text_encoder(inputs.input_ids)
            # Use mean pooling over all tokens for better semantic representation
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            embedding = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def classify(self, prompt, threshold=None):
        """Classify prompt into a domain.

        Args:
            prompt: Text prompt to classify
            threshold: Optional minimum similarity threshold.
                      If best similarity is below threshold, returns None.

        Returns:
            tuple: (domain_name, similarity_score) or (None, score) if below threshold
        """
        prompt_emb = self._get_prompt_embedding(prompt)

        # Compute similarity with each domain
        similarities = {}
        for domain, domain_emb in self.domain_embeddings.items():
            sim = F.cosine_similarity(prompt_emb, domain_emb).item()
            similarities[domain] = sim

        # Find best matching domain
        best_domain = max(similarities, key=similarities.get)
        best_score = similarities[best_domain]

        if threshold is not None and best_score < threshold:
            return None, best_score

        return best_domain, best_score

    def inject_token(self, prompt, domain=None, position="prefix"):
        """Inject domain token into prompt.

        Args:
            prompt: Original text prompt
            domain: Domain name (if None, auto-classify)
            position: Where to inject token ("prefix" or "suffix")

        Returns:
            tuple: (modified_prompt, detected_domain, similarity_score)
        """
        # Auto-classify if domain not specified
        if domain is None:
            domain, score = self.classify(prompt)
            if domain is None:
                return prompt, None, 0.0
        else:
            score = 1.0  # Manual specification

        token = self.domain_tokens.get(domain)
        if token is None:
            return prompt, None, 0.0

        # Inject token
        if position == "prefix":
            modified = f"{token} {prompt}"
        else:
            modified = f"{prompt} {token}"

        return modified, domain, score

    def get_all_similarities(self, prompt):
        """Get similarity scores for all domains.

        Args:
            prompt: Text prompt

        Returns:
            dict: Domain name -> similarity score
        """
        prompt_emb = self._get_prompt_embedding(prompt)

        similarities = {}
        for domain, domain_emb in self.domain_embeddings.items():
            sim = F.cosine_similarity(prompt_emb, domain_emb).item()
            similarities[domain] = sim

        return similarities


class DomainClassifierExternal:
    """
    Domain classifier using a separate CLIP model (not the fine-tuned one).

    This is useful when you want to classify prompts before loading
    the trained DASD model, or when using an external CLIP for classification.
    """

    def __init__(self, domain_descriptions=None, device="cuda"):
        """Initialize external domain classifier.

        Args:
            domain_descriptions: Optional dict mapping domain names to descriptions
                                 e.g., {"satellite": "aerial satellite imagery",
                                        "xray": "medical x-ray radiograph"}
            device: Device to run CLIP on
        """
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

        # Default domain descriptions if not provided
        self.domain_descriptions = domain_descriptions or {
            "satellite": "satellite aerial imagery, top-down view, remote sensing",
            "xray": "medical x-ray radiograph, chest scan, medical imaging",
        }

        # Pre-compute domain embeddings
        self.domain_embeddings = self._compute_embeddings()

    def _compute_embeddings(self):
        """Compute CLIP text embeddings for domain descriptions."""
        embeddings = {}

        with torch.no_grad():
            for domain, description in self.domain_descriptions.items():
                inputs = self.processor(
                    text=[description],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                text_features = self.model.get_text_features(**inputs)
                embeddings[domain] = F.normalize(text_features, p=2, dim=-1)

        return embeddings

    def classify(self, prompt):
        """Classify a prompt into a domain.

        Args:
            prompt: Text prompt to classify

        Returns:
            tuple: (best_domain, similarity_score, all_similarities)
        """
        with torch.no_grad():
            inputs = self.processor(
                text=[prompt],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            prompt_features = self.model.get_text_features(**inputs)
            prompt_features = F.normalize(prompt_features, p=2, dim=-1)

        similarities = {}
        for domain, domain_emb in self.domain_embeddings.items():
            sim = F.cosine_similarity(prompt_features, domain_emb).item()
            similarities[domain] = sim

        best_domain = max(similarities, key=similarities.get)
        return best_domain, similarities[best_domain], similarities


def create_classifier_from_pipeline(pipeline):
    """Create a domain classifier from a CalibratedDASDPipeline.

    Args:
        pipeline: CalibratedDASDPipeline instance

    Returns:
        DomainClassifier instance
    """
    return DomainClassifier(
        text_encoder=pipeline.pipe.text_encoder,
        tokenizer=pipeline.pipe.tokenizer,
        domain_tokens=pipeline.domain_tokens,
        device=pipeline.pipe.device,
    )
