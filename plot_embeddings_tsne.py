"""
Plot t-SNE visualization of claim embeddings from the hover dataset.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import torch

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_claims(data_path: str) -> tuple[list[str], list[str]]:
    """Load claims and labels from dataset."""
    claims = []
    labels = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            claims.append(item['claim'])
            labels.append(item['label'])
    return claims, labels


def get_embeddings(claims: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Get embeddings for all claims."""
    print(f"Loading model: {model_name}")
    encoder = SentenceTransformer(model_name).to(device)

    print(f"Encoding {len(claims)} claims...")
    embeddings = encoder.encode(claims, convert_to_tensor=True, show_progress_bar=True)
    return embeddings.cpu().numpy()


def plot_tsne(embeddings: np.ndarray, labels: list[str], output_path: str, perplexity: int = 30):
    """Apply t-SNE and plot the results."""
    print(f"Applying t-SNE (perplexity={perplexity})...")

    # Adjust perplexity if needed (must be less than n_samples)
    n_samples = len(embeddings)
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 4)
        print(f"  Adjusted perplexity to {perplexity} (n_samples={n_samples})")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create color mapping for labels
    unique_labels = list(set(labels))
    color_map = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]

    # Plot
    plt.figure(figsize=(12, 8))

    # Plot each label group separately for legend
    for label in unique_labels:
        mask = [l == label for l in labels]
        indices = [i for i, m in enumerate(mask) if m]
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            c=[color_map[label]],
            label=f"{label} (n={len(indices)})",
            alpha=0.6,
            s=50
        )

    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.title(f't-SNE Visualization of Claim Embeddings\n(n={n_samples}, perplexity={perplexity})')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot t-SNE of claim embeddings")
    parser.add_argument('--benchmark', type=str, default='hover', help='Benchmark dataset name')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='Sentence transformer model')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity parameter')
    parser.add_argument('--output', type=str, default=None, help='Output path for the plot')
    args = parser.parse_args()

    # Paths
    data_path = f'datasets/{args.benchmark}/data.jsonl'
    output_path = args.output or f'results/{args.benchmark}_embeddings_tsne.png'

    # Load data
    print(f"Loading dataset: {data_path}")
    claims, labels = load_claims(data_path)
    print(f"Loaded {len(claims)} claims")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Get embeddings
    embeddings = get_embeddings(claims, args.model)
    print(f"Embeddings shape: {embeddings.shape}")

    # Plot t-SNE
    plot_tsne(embeddings, labels, output_path, args.perplexity)

    print("Done!")


if __name__ == '__main__':
    main()
