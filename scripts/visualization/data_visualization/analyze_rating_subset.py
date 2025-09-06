#!/usr/bin/env python3
"""
Analyze rating data restricted to the same scope as available pair data.

Scope selection:
- Read pair processed CSV (pilot_pair_60) to collect (topic, comment_num) settings.
- Read rating annotations and map to raw simple data to get topic/comment_num.
- Filter rating annotations to those settings only.

Outputs:
- Distributions for four rating questions
- Summary report with average ratings (overall and per question) and the explicit topic/comment_num lists

Usage: python analyze_rating_subset.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

plt.style.use('default')
sns.set_palette("husl")

RATING_QUESTIONS = [
    "To what extent is your perspective represented in this response?",
    "How informative is this summary?",
    "Do you think this summary presents a neutral and balanced view of the issue?",
    "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding='utf-8')
    except Exception:
        return pd.DataFrame()


def extract_rating_annotations(annotation_dir: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base = Path(annotation_dir)
    if not base.exists():
        return out
    for user_dir in [d for d in base.iterdir() if d.is_dir() and d.name != 'archived_users']:
        fp = user_dir / 'annotated_instances.jsonl'
        if fp.exists():
            anns = load_jsonl(str(fp))
            for a in anns:
                a['user_id'] = user_dir.name
                out.append(a)
    return out


def map_rating_to_raw(annotations: List[Dict[str, Any]], simple_raw_csv: str) -> List[Dict[str, Any]]:
    df = load_csv(simple_raw_csv)
    id2row: Dict[str, Any] = {}
    if not df.empty:
        for _, r in df.iterrows():
            rid = str(r.get('id', ''))
            if rid:
                id2row[rid] = r
    enriched: List[Dict[str, Any]] = []
    for ann in annotations:
        rid = ann.get('id', '')
        meta = {}
        # rating IDs end with _summary/_question; prefer base id
        base_id = rid.replace('_summary', '').replace('_question', '')
        row = id2row.get(base_id)
        if row is not None:
            meta = {
                'topic': row.get('topic', ''),
                'comment_num': row.get('comment_num', 0),
                'model': row.get('model', ''),
            }
        ann = ann.copy()
        ann['raw_metadata'] = meta
        enriched.append(ann)
    return enriched


def collect_pair_scope(pair_processed_csv: str, simple_raw_csv: str) -> set:
    # Prefer topic/comment_num from processed; if missing, fallback via summary ids to simple raw
    dfp = load_csv(pair_processed_csv)
    dfs = load_csv(simple_raw_csv)
    simple_by_id = {str(r.get('id', '')): r for _, r in dfs.iterrows()} if not dfs.empty else {}
    scope = set()
    if dfp.empty:
        return scope
    for _, r in dfp.iterrows():
        topic = r.get('topic', '')
        cn = r.get('comment_num', np.nan)
        if (not topic or (isinstance(topic, float) and np.isnan(topic))) or (isinstance(cn, float) and np.isnan(cn)):
            # try summary ids
            s_a = str(r.get('summary_a_id', ''))
            s_b = str(r.get('summary_b_id', ''))
            src = None
            if s_a in simple_by_id:
                src = simple_by_id[s_a]
            elif s_b in simple_by_id:
                src = simple_by_id[s_b]
            if src is not None:
                topic = src.get('topic', topic)
                cn = src.get('comment_num', cn)
        if topic and not (isinstance(topic, float) and np.isnan(topic)) and cn == cn:
            try:
                scope.add((str(topic), int(cn)))
            except Exception:
                pass
    return scope


def filter_annotations_by_scope(annotations: List[Dict[str, Any]], scope: set) -> List[Dict[str, Any]]:
    if not scope:
        return []
    keep: List[Dict[str, Any]] = []
    for ann in annotations:
        meta = ann.get('raw_metadata', {})
        topic = str(meta.get('topic', ''))
        cn = meta.get('comment_num', 0)
        try:
            cn_int = int(cn)
        except Exception:
            cn_int = 0
        if (topic, cn_int) in scope:
            keep.append(ann)
    return keep


def analyze_ratings(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        'distributions': {q: [] for q in RATING_QUESTIONS},
        'avg_per_question': {},
        'overall_avg': None,
        'per_model_values': defaultdict(lambda: {q: [] for q in RATING_QUESTIONS}),
    }
    all_vals: List[int] = []
    for ann in annotations:
        la = ann.get('label_annotations', {})
        model = ann.get('raw_metadata', {}).get('model', '')
        for q in RATING_QUESTIONS:
            if q in la:
                entry = la[q]
                # find first scale_*
                for k, v in entry.items():
                    if k.startswith('scale_'):
                        try:
                            val = int(v)
                            out['distributions'][q].append(val)
                            all_vals.append(val)
                            if model:
                                out['per_model_values'][model][q].append(val)
                        except Exception:
                            pass
                        break
    for q in RATING_QUESTIONS:
        vals = out['distributions'][q]
        out['avg_per_question'][q] = float(np.mean(vals)) if vals else None
    out['overall_avg'] = float(np.mean(all_vals)) if all_vals else None
    return out


def plot_rating_distributions(analysis: Dict[str, Any], output_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    global_max = 0
    for q in RATING_QUESTIONS:
        counts = Counter(analysis['distributions'][q])
        heights = [counts.get(v, 0) for v in range(1, 6)]
        if heights:
            global_max = max(global_max, max(heights))
    for i, q in enumerate(RATING_QUESTIONS):
        ax = axes[i]
        counts = Counter(analysis['distributions'][q])
        values = list(range(1, 6))
        heights = [counts.get(v, 0) for v in values]
        bars = ax.bar(values, heights, alpha=0.8, color=sns.color_palette("husl", len(values)))
        ax.set_title(f"{q} (n={sum(heights)})")
        ax.set_xlabel('Rating (1-5)')
        ax.set_ylabel('Count')
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks(range(1, 6))
        ymax = max(global_max, max(heights) if heights else 0)
        ax.set_ylim(0, ymax + max(1, int(0.1 * ymax)))
        ax.grid(True, alpha=0.3)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, str(h), ha='center', va='bottom')
    plt.tight_layout()
    out = Path(output_dir) / 'pilot_rating_subset_distributions.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()


def write_summary(analysis: Dict[str, Any], scope: set, output_dir: str) -> None:
    rpt = Path(output_dir) / 'pilot_rating_subset_summary.txt'
    with open(rpt, 'w', encoding='utf-8') as f:
        f.write('PILOT RATING SUBSET SUMMARY\n')
        f.write('='*40 + '\n\n')
        f.write(f'Scope settings (topic, comment_num): {sorted(list(scope))}\n\n')
        f.write(f'Overall average rating: {analysis["overall_avg"] if analysis["overall_avg"] is not None else "NA"}\n\n')
        f.write('Average per question:\n')
        for q, avg in analysis['avg_per_question'].items():
            f.write(f'  - {q}: {avg if avg is not None else "NA"}\n')

        # Per-model averages
        f.write('\nPer-model average ratings (by question):\n')
        model_rows = []
        for model, qmap in analysis['per_model_values'].items():
            f.write(f'  Model: {model}\n')
            overall_vals = []
            for q in RATING_QUESTIONS:
                vals = qmap.get(q, [])
                if vals:
                    avg = float(np.mean(vals))
                    f.write(f'    - {q}: {avg:.3f} (n={len(vals)})\n')
                    overall_vals.extend(vals)
                    model_rows.append({'model': model, 'question': q, 'avg': avg, 'n': len(vals)})
                else:
                    f.write(f'    - {q}: NA (n=0)\n')
                    model_rows.append({'model': model, 'question': q, 'avg': np.nan, 'n': 0})
            f.write(f'    Overall: {float(np.mean(overall_vals)):.3f}\n' if overall_vals else '    Overall: NA\n')

        # Save CSV
        if model_rows:
            dfm = pd.DataFrame(model_rows)
            dfm.to_csv(Path(output_dir) / 'pilot_rating_subset_model_avgs.csv', index=False)


def main():
    # Paths
    base = "/ibex/project/c2328/LLMs-Scalable-Deliberation"
    rating_ann_dir = f"{base}/annotation/summary-rating/annotation_output/pilot_rating"
    raw_simple_csv = f"{base}/annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_simple.csv"
    pair_processed_csv = f"{base}/annotation/summary-rating/data_files/processed/sum_humanstudy_v0903_pilot_pair_60.csv"
    out_dir = f"{base}/results/pilot_rating_subset_analysis"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Scope from pair data
    scope = collect_pair_scope(pair_processed_csv, raw_simple_csv)
    if not scope:
        print("No scope collected from pair data; aborting.")
        return
    print(f"Scope settings: {sorted(list(scope))}")

    # Load and map rating
    anns = extract_rating_annotations(rating_ann_dir)
    anns = map_rating_to_raw(anns, raw_simple_csv)
    anns = filter_annotations_by_scope(anns, scope)
    print(f"Filtered rating annotations: {len(anns)}")

    # Analyze
    analysis = analyze_ratings(anns)

    # Plots and report
    plot_rating_distributions(analysis, out_dir)
    write_summary(analysis, scope, out_dir)

    # Model×Question heatmap of average ratings
    if analysis['per_model_values']:
        models = sorted(list(analysis['per_model_values'].keys()))
        if models:
            mat = np.zeros((len(models), len(RATING_QUESTIONS)))
            mat[:] = np.nan
            for i, m in enumerate(models):
                for j, q in enumerate(RATING_QUESTIONS):
                    vals = analysis['per_model_values'][m].get(q, [])
                    if vals:
                        mat[i, j] = float(np.mean(vals))
            plt.figure(figsize=(max(8, len(RATING_QUESTIONS)*1.2), max(6, len(models)*0.6)))
            im = plt.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
            plt.title('Average Rating per Model and Question (subset)')
            plt.xlabel('Question')
            plt.ylabel('Model')
            plt.xticks(range(len(RATING_QUESTIONS)), RATING_QUESTIONS, rotation=45, ha='right')
            plt.yticks(range(len(models)), models)
            for i in range(len(models)):
                for j in range(len(RATING_QUESTIONS)):
                    if not np.isnan(mat[i, j]):
                        plt.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(str(Path(out_dir) / 'pilot_rating_subset_model_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

    print("\n✅ Pilot rating subset analysis completed!")
    print(f"📁 Output: {out_dir}")
    print("Generated files:")
    print("  - pilot_rating_subset_distributions.png")
    print("  - pilot_rating_subset_summary.txt")
    print("  - pilot_rating_subset_model_avgs.csv")
    print("  - pilot_rating_subset_model_heatmap.png")


if __name__ == "__main__":
    main()


