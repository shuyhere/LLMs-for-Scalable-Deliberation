import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List
from wordcloud import WordCloud, STOPWORDS


def normalize_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.[^\s]+", " ", text)
    # Remove non-letter characters (keep spaces and apostrophes inside words)
    text = re.sub(r"[^a-z'\s]", " ", text)
    # Collapse apostrophes-only tokens to space
    text = re.sub(r"\b'+\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in text.split(" ") if t]


def load_comments(json_path: Path) -> List[str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    comments = data.get("comments", [])
    out: List[str] = []
    for c in comments:
        comment = (c or {}).get("comment") if isinstance(c, dict) else None
        if not comment:
            continue
        out.append(comment)
    return out


def build_frequencies(texts: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    stopwords = set(STOPWORDS)
    stopwords.add("think")
    stopwords.add("will")
    for text in texts:
        norm = normalize_text(text)
        tokens = [t for t in tokenize(norm) if t not in stopwords and len(t) > 2]
        counter.update(tokens)
    return counter


def make_wordcloud(counter: Counter, out_path: Path, width: int = 1600, height: int = 900) -> None:
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Required packages not found. Please install: pip install wordcloud matplotlib"
        ) from e

    if not counter:
        # Create an empty canvas to signal no content
        wc = WordCloud(width=width, height=height, background_color="white")
        img = wc.to_image()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        max_words=500,
        prefer_horizontal=0.9,
        collocations=False,
    )
    wc.generate_from_frequencies(dict(counter))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate word clouds from cleaned dataset comments.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "datasets" / "cleaned_new_dataset",
        help="Directory containing cleaned JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "results" / "dataset_visulization" / "wordclouds",
        help="Directory to write wordcloud PDFs",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="Minimum frequency to include a token in the wordcloud",
    )
    args = parser.parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    all_texts: List[str] = []
    per_dataset = []  # (display_name, freq)
    stem_to_path = {p.stem: p for p in json_files}

    # Preferred ordering and display names
    ordered = [
        ("Openqa-Tipping-System", "Tipping System"),
        ("Openqa-AI-changes-human-life", "AI Changes Life"),
        ("Openqa-Trump-cutting-funding", "Academic Funding"),
        ("Openqa-Influencers-as-a-job", "Influencer"),
        ("Openqa-Updates-of-electronic-products", "Electronic Products"),
        ("Binary-Tariff-Policy", "Tariff Policy"),
        ("Binary-Health-Care-Policy", "Health Care"),
        ("Binary-Vaccination-Policy", "Vaccination Policy"),
        ("Binary-Refugee-Policies", "Refugee Policy"),
        ("Binary-Online-Identity-Policies", "Online Identity"),
    ]

    # Process in preferred order; fall back to any extra files afterward
    seen_stems = set()
    for stem, display_name in ordered:
        jf = stem_to_path.get(stem)
        if jf is None:
            continue
        comments = load_comments(jf)
        all_texts.extend(comments)

        freq = build_frequencies(comments)
        if args.min_freq > 1:
            freq = Counter({k: v for k, v in freq.items() if v >= args.min_freq})
        out_path = output_dir / f"{jf.stem}_wordcloud.pdf"
        make_wordcloud(freq, out_path)
        print(f"Saved: {out_path}")
        per_dataset.append((display_name, freq))
        seen_stems.add(stem)

    # Any remaining datasets not in the preferred list
    for jf in json_files:
        if jf.stem in seen_stems:
            continue
        comments = load_comments(jf)
        all_texts.extend(comments)

        freq = build_frequencies(comments)
        if args.min_freq > 1:
            freq = Counter({k: v for k, v in freq.items() if v >= args.min_freq})
        out_path = output_dir / f"{jf.stem}_wordcloud.pdf"
        make_wordcloud(freq, out_path)
        print(f"Saved: {out_path}")
        per_dataset.append((jf.stem.replace("-", " "), freq))

    # Build combined multi-subplot PDF
    if per_dataset:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "matplotlib is required. Please install: pip install matplotlib"
            ) from e

        n = len(per_dataset)
        cols = 5 if n >= 5 else n
        rows = (n + cols - 1) // cols
        fig_width = cols * 6
        fig_height = rows * 4.8
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]

        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                if idx < n:
                    name, freq = per_dataset[idx]
                    wc = WordCloud(
                        width=800,
                        height=450,
                        background_color="white",
                        max_words=500,
                        prefer_horizontal=0.9,
                        collocations=False,
                        stopwords=set(STOPWORDS),
                    )
                    wc.generate_from_frequencies(dict(freq))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.set_title(name, fontsize=14)
                    ax.axis("off")
                else:
                    ax.axis("off")
                idx += 1

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_pdf = output_dir / "all_wordclouds.pdf"
        fig.savefig(combined_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_pdf}")

if __name__ == "__main__":
    main()


