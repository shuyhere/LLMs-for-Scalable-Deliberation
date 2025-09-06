import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
import warnings

# Add the src directory to the path to import OpenAIClient
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def load_comments(json_path: Path) -> List[str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    comments = data.get("comments", [])
    out: List[str] = []
    for c in comments:
        comment = (c or {}).get("comment") if isinstance(c, dict) else None
        if comment:
            out.append(str(comment))
    return out


def safe_imports():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore
        import numpy as np  # type: ignore
        import umap  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import hdbscan  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Please install dependencies: pip install sentence-transformers scikit-learn umap-learn matplotlib numpy hdbscan"
        ) from e
    return SentenceTransformer, TfidfVectorizer, KMeans, np, umap, plt, hdbscan, DBSCAN, AgglomerativeClustering, silhouette_score


@dataclass
class ClusterResult:
    labels: List[int]
    label_to_indices: Dict[int, List[int]]
    label_to_top_texts: Dict[int, List[str]]
    label_to_repr_idx: Dict[int, int]
    embedding_2d: "np.ndarray"
    label_to_summary: Dict[int, Dict[str, Any]]  # LLM-generated summaries


@dataclass
class ClusterSummary:
    cluster_id: int
    main_topic: str
    key_opinions: List[str]
    stance: str
    sentiment: str
    representative_texts: List[str]


def compute_embeddings(model_name: str, texts: Sequence[str]) -> "np.ndarray":
    SentenceTransformer, _, _, np, _, _, _, _, _, _ = safe_imports()
    model = SentenceTransformer(model_name)
    emb = model.encode(list(texts), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


def perform_clustering_hdbscan(embeddings: "np.ndarray", min_cluster_size: int = 5, min_clusters: int = 2) -> Tuple[List[int], Dict[int, List[int]]]:
    """Perform HDBSCAN clustering for adaptive cluster discovery."""
    _, _, _, np, _, _, hdbscan, _, _, _ = safe_imports()
    
    n = embeddings.shape[0]
    if n == 0:
        return [], {}
    
    # Adjust min_cluster_size based on dataset size
    min_cluster_size = max(2, min(min_cluster_size, max(2, n // 20)))
    
    # Try HDBSCAN first
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric='euclidean',
        cluster_selection_epsilon=0.0,
        cluster_selection_method='eom'
    )
    
    labels = clusterer.fit_predict(embeddings)
    
    # Check if we have at least min_clusters
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    # If we have fewer clusters than required, use KMeans fallback
    if len(unique_labels) < min_clusters:
        from sklearn.cluster import KMeans
        print(f"HDBSCAN found only {len(unique_labels)} clusters, using KMeans with k={min_clusters}")
        km = KMeans(n_clusters=min_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
    
    # Convert to list and handle noise points (-1 labels)
    labels = list(map(int, labels))
    
    # Group indices by label
    label_to_indices: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        if lab != -1:  # Ignore noise points
            label_to_indices.setdefault(lab, []).append(idx)
    
    # Always reassign noise points to nearest cluster to ensure all samples are clustered
    noise_indices = [i for i, l in enumerate(labels) if l == -1]
    if noise_indices and label_to_indices:
        print(f"Reassigning {len(noise_indices)} noise points to nearest clusters...")
        import numpy as np
        for idx in noise_indices:
            # Find nearest cluster centroid
            min_dist = float('inf')
            best_label = 0
            for lab, idxs in label_to_indices.items():
                cluster_emb = embeddings[idxs]
                centroid = cluster_emb.mean(axis=0)
                dist = np.linalg.norm(embeddings[idx] - centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_label = lab
            labels[idx] = best_label
            label_to_indices[best_label].append(idx)
    
    return labels, label_to_indices


def perform_clustering_optimal(embeddings: "np.ndarray", max_k: int = 15) -> Tuple[List[int], Dict[int, List[int]]]:
    """Find optimal number of clusters using silhouette score."""
    _, _, KMeans, np, _, _, _, _, _, silhouette_score = safe_imports()
    
    n = embeddings.shape[0]
    if n == 0:
        return [], {}
    
    # Try different k values and find the best one
    max_k = min(max_k, n - 1)
    best_score = -1
    best_k = 2
    
    for k in range(2, min(max_k + 1, n)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal k={best_k} with silhouette score={best_score:.3f}")
    
    # Perform final clustering with best k
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = list(map(int, km.fit_predict(embeddings)))
    
    label_to_indices: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        label_to_indices.setdefault(lab, []).append(idx)
    
    return labels, label_to_indices


def perform_clustering(embeddings: "np.ndarray", k: Optional[int], method: str = "hdbscan", is_binary: bool = False) -> Tuple[List[int], Dict[int, List[int]]]:
    """Perform clustering using specified method."""
    min_clusters = 2 if is_binary else 2  # Always ensure at least 2 clusters
    
    if method == "hdbscan":
        return perform_clustering_hdbscan(embeddings, min_clusters=min_clusters)
    elif method == "optimal":
        return perform_clustering_optimal(embeddings)
    else:  # kmeans
        _, _, KMeans, np, _, _, _, _, _, _ = safe_imports()
        n = embeddings.shape[0]
        if n == 0:
            return [], {}
        if k is None:
            # Heuristic for k when not provided
            k = max(2, min(10, int(math.sqrt(max(2, n // 10)))))
        k = max(2, min(k, max(2, n)))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = list(map(int, km.fit_predict(embeddings)))
        label_to_indices: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels):
            label_to_indices.setdefault(lab, []).append(idx)
        return labels, label_to_indices


def extract_representative_texts(
    texts: Sequence[str],
    embeddings: "np.ndarray",
    label_to_indices: Dict[int, List[int]],
    top_k: int = 5,
) -> Tuple[Dict[int, List[str]], Dict[int, List[int]]]:
    """Select top-k representative samples (closest to centroid) per cluster."""
    import numpy as np  # type: ignore

    label_to_texts: Dict[int, List[str]] = {}
    label_to_top_idxs: Dict[int, List[int]] = {}
    for lab, idxs in label_to_indices.items():
        if not idxs:
            label_to_texts[lab] = []
            label_to_top_idxs[lab] = []
            continue
        cluster_emb = embeddings[idxs]
        centroid = cluster_emb.mean(axis=0, keepdims=True)
        sims = (cluster_emb @ centroid.T).ravel()
        order = sims.argsort()[::-1]
        sel: List[int] = [idxs[int(i)] for i in order[: max(1, top_k)]]
        label_to_top_idxs[lab] = sel
        label_to_texts[lab] = [texts[i] for i in sel]
    return label_to_texts, label_to_top_idxs


def generate_cluster_summary_with_llm(
    cluster_texts: List[str],
    cluster_id: int,
    openai_client: Optional[Any] = None
) -> Dict[str, Any]:
    """Generate cluster summary using LLM."""
    if not openai_client:
        # Return basic summary without LLM
        return {
            "cluster_id": cluster_id,
            "main_topic": f"Cluster {cluster_id}",
            "key_opinions": [],
            "stance": "Mixed",
            "sentiment": "Neutral",
            "sample_size": len(cluster_texts)
        }
    
    # Prepare prompt for LLM
    texts_sample = "\n".join([f"- {text[:200]}..." if len(text) > 200 else f"- {text}" 
                              for text in cluster_texts[:10]])  # Use top 10 texts
    
    prompt = f"""Analyze the following comments from a discussion cluster and provide a structured summary:

Comments:
{texts_sample}

Please provide:
1. Main Topic: A concise description of what this cluster is discussing (1-2 sentences)
2. Key Opinions: List 3-5 main opinions or viewpoints expressed
3. Overall Stance: Describe the general position (e.g., "Supportive", "Critical", "Mixed", "Neutral")
4. Sentiment: Overall emotional tone (e.g., "Positive", "Negative", "Neutral", "Mixed")

Format your response as JSON with keys: main_topic, key_opinions (list), stance, sentiment"""

    try:
        response = openai_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert at analyzing discussion comments and identifying themes, opinions, and stances."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        # Parse the response
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            summary = json.loads(json_match.group())
        else:
            # Fallback parsing
            summary = {
                "main_topic": "Analysis pending",
                "key_opinions": [],
                "stance": "Unknown",
                "sentiment": "Unknown"
            }
        
        summary["cluster_id"] = cluster_id
        summary["sample_size"] = len(cluster_texts)
        return summary
        
    except Exception as e:
        print(f"Error generating summary for cluster {cluster_id}: {e}")
        return {
            "cluster_id": cluster_id,
            "main_topic": f"Cluster {cluster_id} (LLM analysis failed)",
            "key_opinions": [],
            "stance": "Unknown",
            "sentiment": "Unknown",
            "sample_size": len(cluster_texts)
        }


def find_representatives(
    embeddings: "np.ndarray",
    label_to_indices: Dict[int, List[int]],
) -> Dict[int, int]:
    import numpy as np  # type: ignore

    repr_map: Dict[int, int] = {}
    for lab, idxs in label_to_indices.items():
        if not idxs:
            continue
        cluster_emb = embeddings[idxs]
        centroid = cluster_emb.mean(axis=0, keepdims=True)
        # cos distance since embeddings normalized; equivalent to 1 - dot
        sims = (cluster_emb @ centroid.T).ravel()
        best_local = int(np.argmax(sims))
        repr_map[lab] = idxs[best_local]
    return repr_map


def reduce_to_2d(embeddings: "np.ndarray", labels: Optional[List[int]] = None) -> "np.ndarray":
    _, _, _, np, umap, _, _, _, _, _ = safe_imports()
    if embeddings.shape[0] <= 1:
        return np.zeros((embeddings.shape[0], 2), dtype="float32")
    
    # UMAP parameters optimized for compact layout
    # Smaller n_neighbors for more local structure, very small min_dist for tight packing
    n_size = embeddings.shape[0]
    n_neighbors = min(30, max(5, n_size // 15))  # Adaptive neighbors
    min_dist = 0.001  # Extremely small for maximum compactness
    
    # Use supervised UMAP if labels are provided
    if labels is not None:
        # For binary or small cluster counts, use even tighter parameters
        n_unique_labels = len(set(labels))
        if n_unique_labels <= 3:
            n_neighbors = min(15, max(5, n_size // 20))
            spread = 0.3
        else:
            spread = 0.4
            
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist,
            metric='cosine',
            spread=spread,  # Very low spread for compact layout
            target_metric='categorical',
            target_weight=0.8,  # High supervision weight for clear separation
            transform_seed=42
        )
        # Convert labels to array, handling -1 (noise) labels
        import numpy as np
        labels_array = np.array(labels)
        labels_array[labels_array == -1] = max(labels) + 1  # Treat noise as separate category
        return reducer.fit_transform(embeddings, y=labels_array).astype("float32")
    else:
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist,
            metric='cosine',
            spread=0.4,  # Low spread for compact layout
            transform_seed=42
        )
        return reducer.fit_transform(embeddings).astype("float32")


def visualize_clusters_enhanced(
    out_pdf: Path,
    embedding_2d: "np.ndarray",
    labels: Sequence[int],
    label_to_top_texts: Dict[int, List[str]],
    label_to_repr_idx: Dict[int, int],
    label_to_summary: Dict[int, Dict[str, Any]],
    label_to_indices: Dict[int, List[int]],
    title: str,
) -> None:
    _, _, _, np, _, plt, _, _, _, _ = safe_imports()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Create square figure with better display
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')  # Light gray background
    
    labels_arr = np.asarray(labels)
    uniq = sorted(set(labels))
    
    # Build beautiful color palette
    if len(uniq) <= 10:
        # Use tab10 for small number of clusters
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(uniq))]
    elif len(uniq) <= 20:
        # Use tab20 for medium number
        cmap = plt.cm.get_cmap("tab20")
        colors = [cmap(i) for i in range(len(uniq))]
    else:
        # Use multiple colormaps for large number
        base_maps = [plt.cm.get_cmap("tab20"), plt.cm.get_cmap("tab20b"), plt.cm.get_cmap("Set3")]
        colors = []
        for i, lab in enumerate(uniq):
            map_idx = i % len(base_maps)
            color_idx = (i // len(base_maps)) % 20
            colors.append(base_maps[map_idx](color_idx / 19.0))

    # Plot clusters
    for i, lab in enumerate(uniq):
        mask = labels_arr == lab
        pts = embedding_2d[mask]
        if pts.size == 0:
            continue
        color = colors[i % len(colors)]
        ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.7, color=color, edgecolors='white', linewidth=0.3)

        # Draw convex hull per cluster
        if pts.shape[0] >= 3:
            hull = _convex_hull(pts)
            if hull is not None:
                hx, hy = zip(*hull)
                ax.fill(hx, hy, facecolor=color, alpha=0.1, edgecolor=color, linewidth=0.8, linestyle=':')

    # Optional: Add small cluster numbers in corners (comment out if not needed)
    # for lab, idx in label_to_repr_idx.items():
    #     mask = labels_arr == lab
    #     pts = embedding_2d[mask]
    #     if pts.size == 0:
    #         continue
    #     cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    #     ax.text(cx, cy, str(lab), fontsize=8, ha="center", va="center",
    #             color='white', weight='bold',
    #             bbox=dict(boxstyle="circle,pad=0.1", fc=colors[lab % len(colors)], alpha=0.8))

    # Calculate data bounds with very tight margins
    margin = 0.02  # Very small margin
    x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
    
    # Add small padding
    x_padding = (x_max - x_min) * margin
    y_padding = (y_max - y_min) * margin
    
    # Set limits to actual data range with minimal padding
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    ax.set_title(title, fontsize=13, weight="bold", pad=8)
    
    # Add grid and ticks for better visualization
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Keep borders visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.0)
    
    ax.set_aspect('auto')  # Allow non-square aspect to fit data better
    
    # Add a compact legend showing cluster info
    legend_elements = []
    for i, lab in enumerate(uniq[:8]):  # Show max 8 in legend for space
        summary = label_to_summary.get(lab, {})
        stance = summary.get("stance", f"Cluster {lab}")[:12]  # Truncate long stances
        count = len(label_to_indices.get(lab, []))
        label = f"{stance} ({count})"
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=colors[i % len(colors)], 
                                    label=label, alpha=0.7))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, 
                 framealpha=0.95, borderpad=0.3, columnspacing=0.4,
                 handlelength=1.0, handletextpad=0.3, ncol=1,
                 fancybox=True, shadow=True)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1, facecolor='white', dpi=150)
    plt.close()
    
    # Also save detailed summary as JSON
    summary_json_path = out_pdf.with_suffix('.json')
    with open(summary_json_path, 'w') as f:
        json.dump(label_to_summary, f, indent=2)
    print(f"Saved detailed summary to {summary_json_path}")


def _convex_hull(points: "np.ndarray") -> Optional[List[Tuple[float, float]]]:
    """Compute convex hull of 2D points using monotonic chain. Returns list of (x,y)."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    P = [(float(x), float(y)) for x, y in points]
    P = sorted(P)
    if len(P) < 3:
        return None

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull


def process_dataset_enhanced(
    json_path: Path,
    model_name: str,
    k: Optional[int],
    output_dir: Path,
    display_name: Optional[str] = None,
    clustering_method: str = "hdbscan",
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini"
) -> Optional[Path]:
    texts = load_comments(json_path)
    if not texts:
        return None
    
    print(f"Processing {json_path.name} with {len(texts)} comments...")
    
    # Compute embeddings
    print("Computing embeddings...")
    emb = compute_embeddings(model_name, texts)
    
    # Check if this is a binary question
    is_binary = "Binary" in json_path.stem or "binary" in json_path.stem.lower()
    
    # Perform clustering
    print(f"Performing {clustering_method} clustering (binary={is_binary})...")
    labels, label_to_indices = perform_clustering(emb, k, method=clustering_method, is_binary=is_binary)
    
    print(f"Found {len(label_to_indices)} clusters with {sum(len(idxs) for idxs in label_to_indices.values())} samples clustered")
    
    # Verify all samples are clustered
    total_samples = len(texts)
    clustered_samples = sum(1 for l in labels if l != -1)
    if clustered_samples < total_samples:
        print(f"Warning: Only {clustered_samples}/{total_samples} samples were clustered")
    

    # Extract representative texts
    top_texts, top_idxs = extract_representative_texts(texts, emb, label_to_indices, top_k=5)
    
    # Generate LLM summaries if requested
    label_to_summary = {}
    if use_llm:
        print("Generating cluster summaries with LLM...")
        try:
            from models.openai_client import OpenAIClient
            openai_client = OpenAIClient(model=llm_model, temperature=0.3)
            
            for lab, idxs in label_to_indices.items():
                cluster_texts = [texts[i] for i in idxs]
                summary = generate_cluster_summary_with_llm(cluster_texts, lab, openai_client)
                label_to_summary[lab] = summary
                print(f"  Cluster {lab}: {summary.get('main_topic', 'Unknown')[:50]}...")
        except Exception as e:
            print(f"Warning: Could not use LLM for summaries: {e}")
            # Generate basic summaries without LLM
            for lab in label_to_indices.keys():
                label_to_summary[lab] = generate_cluster_summary_with_llm([], lab, None)
    else:
        # Generate basic summaries without LLM
        for lab in label_to_indices.keys():
            label_to_summary[lab] = generate_cluster_summary_with_llm([], lab, None)
    
    # Find representative indices
    repr_idx = {lab: (idxs[0] if idxs else label_to_indices[lab][0]) 
                for lab, idxs in top_idxs.items()}
    
    # Reduce to 2D for visualization with labels for better separation
    print("Reducing to 2D for visualization...")
    emb2d = reduce_to_2d(emb, labels)
    
    # Visualize
    title = display_name or json_path.stem
    out_pdf = output_dir / f"{json_path.stem}_semantic_clusters_enhanced.pdf"
    print("Creating visualization...")
    visualize_clusters_enhanced(out_pdf, emb2d, labels, top_texts, repr_idx, label_to_summary, label_to_indices, title)
    
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced semantic clustering with LLM-based analysis."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "datasets" / "cleaned_new_dataset",
        help="Directory containing cleaned JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "results" / "dataset_visulization" / "semantics_enhanced",
        help="Directory to write clustering PDFs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters (for KMeans). If omitted, automatic selection is used.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hdbscan",
        choices=["kmeans", "hdbscan", "optimal"],
        help="Clustering method: 'hdbscan' (adaptive), 'optimal' (find best k), or 'kmeans'",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Use LLM to generate cluster summaries",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for cluster analysis",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process only a specific file (by name or pattern)",
    )
    
    args = parser.parse_args()

    # Preferred display names
    display_name_by_stem = {
        "Openqa-Tipping-System": "Tipping System",
        "Openqa-AI-changes-human-life": "AI Changes Life",
        "Openqa-Trump-cutting-funding": "Academic Funding",
        "Openqa-Influencers-as-a-job": "Influencer",
        "Openqa-Updates-of-electronic-products": "Electronic Products",
        "Binary-Tariff-Policy": "Tariff Policy",
        "Binary-Health-Care-Policy": "Health Care",
        "Binary-Vaccination-Policy": "Vaccination Policy",
        "Binary-Refugee-Policies": "Refugee Policy",
        "Binary-Online-Identity-Policies": "Online Identity",
    }

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    
    # Find JSON files
    if args.file:
        # Process specific file
        json_files = list(input_dir.glob(f"*{args.file}*.json"))
        if not json_files:
            json_files = list(input_dir.glob(f"{args.file}"))
    else:
        json_files = sorted(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} files to process")
    
    for jf in json_files:
        print(f"\n{'='*60}")
        print(f"Processing: {jf.name}")
        display_name = display_name_by_stem.get(jf.stem)
        
        try:
            out_path = process_dataset_enhanced(
                jf, 
                args.model, 
                args.k, 
                output_dir, 
                display_name,
                clustering_method=args.method,
                use_llm=args.use_llm,
                llm_model=args.llm_model
            )
            if out_path:
                print(f"✓ Saved: {out_path}")
        except Exception as e:
            print(f"✗ Error processing {jf.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()