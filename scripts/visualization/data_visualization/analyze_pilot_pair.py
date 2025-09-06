#!/usr/bin/env python3
"""
Analyze pilot_pair annotation data.

Generates:
- Distributions for four pair-comparison questions (1=A, 2=B)
- Summary report with explicit topic(s) and comment_num(s)
- Per-pair summary CSV and per-question win rates CSV

Usage: python analyze_pilot_pair.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

# Abbreviation maps (reuse from rating)
TOPIC_ABBR_MAP = {
    'Binary-Health-Care-Policy': 'HealthCare',
    'Binary-Online-Identity-Policies': 'OnlineID',
    'Binary-Refugee-Policies': 'Refugee',
    'Binary-Tariff-Policy': 'Tariff',
    'Binary-Vaccination-Policy': 'Vaccination',
    'Openqa-AI-changes-human-life': 'AI-Life',
    'Openqa-Tipping-System': 'Tipping',
    'Openqa-Trump-cutting-funding': 'TrumpFunding',
    'Openqa-Updates-of-electronic-products': 'ElecProducts',
    'Openqa-Influencers-as-a-job': 'Influencers',
}

QUESTION_ABBR_MAP = {
    "Which summary is more representative of your perspective?": 'Representative',
    "Which summary is more informative?": 'Informative',
    "Which summary presents a more neutral and balanced view of the issue?": 'Balanced',
    "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?": 'PolicyUse',
}

def abbreviate_topic(topic: str) -> str:
    return TOPIC_ABBR_MAP.get(topic, topic[:12] + ('…' if len(topic) > 12 else ''))

def abbreviate_question(question: str) -> str:
    return QUESTION_ABBR_MAP.get(question, f"Q:{question[:10]}…" if len(question) > 10 else question)


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data


def load_csv_file(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def load_raw_pair_data(raw_data_dir: str) -> pd.DataFrame:
    # Use processed pair csv
    pair_file = f"{raw_data_dir}/sum_humanstudy_v0903_pilot_pair_60.csv"
    df = load_csv_file(pair_file)
    if df.empty:
        print(f"Warning: raw pair file not found or empty: {pair_file}")
    return df


def load_simple_raw_data(raw_data_root: str) -> pd.DataFrame:
    # Load simple raw to fetch topic/comment_num by summary ids
    simple_file = \
        f"{Path(raw_data_root).parent}/raw/summaries_V0903_for_humanstudy_simple.csv"
    df = load_csv_file(simple_file)
    if df.empty:
        print(f"Warning: simple raw file not found or empty: {simple_file}")
    return df


def extract_pilot_pair_annotations(annotation_dir: str) -> List[Dict[str, Any]]:
    all_annotations = []
    annotation_path = Path(annotation_dir)
    if not annotation_path.exists():
        print(f"Annotation directory not found: {annotation_dir}")
        return all_annotations
    user_dirs = [d for d in annotation_path.iterdir() if d.is_dir() and d.name != 'archived_users']
    for user_dir in user_dirs:
        user_id = user_dir.name
        annotated_file = user_dir / 'annotated_instances.jsonl'
        if annotated_file.exists():
            annotations = load_jsonl_file(str(annotated_file))
            for ann in annotations:
                ann['user_id'] = user_id
                ann['task_type'] = 'pair'
                all_annotations.append(ann)
    return all_annotations


def _id_candidates(rid: str) -> List[str]:
    # Generate robust candidates to match processed CSV IDs
    cands = [rid]
    if rid.endswith('_question'):
        cands.append(rid[:-9])
    # Strip trailing "_pair_<num>_question" or "_pair_<num>"
    # Keep base pair id variant as well
    import re
    m = re.match(r"^(.*?_pair_\d+)(?:_question)?$", rid)
    if m:
        cands.append(m.group(1))
    # Also try removing any trailing segment after first two parts joined by '_'
    parts = rid.split('_')
    if len(parts) >= 3:
        cands.append('_'.join(parts[:3]))
    # Deduplicate preserving order
    seen = set()
    out = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def map_pair_annotations_to_raw(annotations: List[Dict[str, Any]], raw_df: pd.DataFrame, simple_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if raw_df is None or raw_df.empty:
        return annotations
    id_to_row = {}
    for _, row in raw_df.iterrows():
        rid = str(row.get('id', ''))
        if rid:
            id_to_row[rid] = row
    simple_id_to_row = {}
    if simple_df is not None and not simple_df.empty:
        for _, row in simple_df.iterrows():
            sid = str(row.get('id', ''))
            if sid:
                simple_id_to_row[sid] = row
    enriched = []
    for ann in annotations:
        rid = ann.get('id', '')
        meta = {}
        # try multiple normalized candidates
        matched_row = None
        for cand in _id_candidates(rid):
            if cand in id_to_row:
                matched_row = id_to_row[cand]
                break
        # fallback: loose startswith/endswith matching if exact not found
        if matched_row is None:
            for key in id_to_row.keys():
                if rid.startswith(key) or key.startswith(rid):
                    matched_row = id_to_row[key]
                    break
        if matched_row is not None:
            r = matched_row
            topic = r.get('topic', '')
            comment_num = r.get('comment_num', 0)
            dataset_name = r.get('dataset_name', '')
            s_a = str(r.get('summary_a_id', ''))
            s_b = str(r.get('summary_b_id', ''))
            # If topic/comment_num missing from pair row, try resolve from simple raw via summary ids
            if (not topic or (isinstance(topic, float) and np.isnan(topic))) and simple_id_to_row:
                if s_a in simple_id_to_row:
                    topic = simple_id_to_row[s_a].get('topic', topic)
                    comment_num = simple_id_to_row[s_a].get('comment_num', comment_num)
                elif s_b in simple_id_to_row:
                    topic = simple_id_to_row[s_b].get('topic', topic)
                    comment_num = simple_id_to_row[s_b].get('comment_num', comment_num)
            meta = {
                'topic': topic if topic is not None else '',
                'comment_num': int(comment_num) if pd.notna(comment_num) else 0,
                'dataset_name': dataset_name,
                'model_a': r.get('model_a', ''),
                'model_b': r.get('model_b', ''),
                'summary_a_id': s_a,
                'summary_b_id': s_b,
            }
        ann = ann.copy()
        ann['raw_metadata'] = meta
        enriched.append(ann)
    return enriched


PAIR_QUESTIONS = [
    "Which summary is more representative of your perspective?",
    "Which summary is more informative?",
    "Which summary presents a more neutral and balanced view of the issue?",
    "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?",
]

# Pair task uses 1/2 only: 1 => choose first (A), 2 => choose second (B)
# Map to numeric preference for aggregation: A=-1, B=+1
PREF_MAP = {1: -1, 2: 1}


def analyze_pair_scores(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    analysis = {
        'distributions': {q: [] for q in PAIR_QUESTIONS},
        'preference_scores': {q: [] for q in PAIR_QUESTIONS},
        'detailed': [],
        'topic_stats': defaultdict(list),
        'comment_num_stats': defaultdict(list),
        'per_question_outcomes': {q: Counter() for q in PAIR_QUESTIONS},  # A/B/Tie
        'pair_headtohead': defaultdict(lambda: {'A':0,'B':0,'Tie':0}),   # (model_a,model_b)
        'pair_records': [],
        # per-question head-to-head tallies: (question, model_a, model_b) -> {A,B,Tie}
        'per_question_h2h': defaultdict(lambda: {'A':0,'B':0,'Tie':0})
    }
    for ann in annotations:
        la = ann.get('label_annotations', {})
        rid = ann.get('id', '')
        user_id = ann.get('user_id', 'unknown')
        meta = ann.get('raw_metadata', {})
        topic = meta.get('topic', 'Unknown')
        comment_num = meta.get('comment_num', 0)
        model_a = meta.get('model_a', 'ModelA')
        model_b = meta.get('model_b', 'ModelB')
        record = {
            'annotation_id': rid,
            'user_id': user_id,
            'topic': topic,
            'comment_num': comment_num,
            'model_a': model_a,
            'model_b': model_b,
            'ratings': {},
            'prefs': {},
        }
        for q in PAIR_QUESTIONS:
            if q in la:
                entry = la[q]
                rating = None
                for k, v in entry.items():
                    if k.startswith('scale_'):
                        try:
                            rating = int(v)
                        except Exception:
                            pass
                        break
                if rating in (1, 2):
                    analysis['distributions'][q].append(rating)
                    pref = PREF_MAP[rating]
                    analysis['preference_scores'][q].append(pref)
                    record['ratings'][q] = rating
                    record['prefs'][q] = pref
                    # outcome per question
                    if rating == 1:
                        analysis['per_question_outcomes'][q]['A'] += 1
                        analysis['per_question_h2h'][(q, model_a, model_b)]['A'] += 1
                    elif rating == 2:
                        analysis['per_question_outcomes'][q]['B'] += 1
                        analysis['per_question_h2h'][(q, model_a, model_b)]['B'] += 1
        analysis['detailed'].append(record)
        # aggregate topic/comment stats with mean pref across answered qs
        if record['prefs']:
            mean_pref = float(np.mean(list(record['prefs'].values())))
            analysis['topic_stats'][topic].append(mean_pref)
            analysis['comment_num_stats'][comment_num].append(mean_pref)
        # head-to-head per pair across answered qs (majority outcome)
        if record['prefs']:
            avg_pref = float(np.mean(list(record['prefs'].values())))
            key = (model_a, model_b)
            if avg_pref < 0:
                analysis['pair_headtohead'][key]['A'] += 1
            elif avg_pref > 0:
                analysis['pair_headtohead'][key]['B'] += 1
            else:
                analysis['pair_headtohead'][key]['Tie'] += 1
        analysis['pair_records'].append(record)
    return analysis


def plot_pair_distributions(analysis: Dict[str, Any], output_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    global_max = 0
    for q in PAIR_QUESTIONS:
        counts = Counter(analysis['distributions'][q])
        heights = [counts.get(v, 0) for v in (1, 2)]
        if heights:
            global_max = max(global_max, max(heights))
    for i, q in enumerate(PAIR_QUESTIONS):
        ax = axes[i]
        counts = Counter(analysis['distributions'][q])
        values = [1, 2]
        heights = [counts.get(v, 0) for v in values]
        bars = ax.bar(values, heights, alpha=0.8, color=sns.color_palette("husl", len(values)))
        ax.set_title(f"{abbreviate_question(q)} (n={sum(heights)})")
        ax.set_xlabel('Choice (1=A, 2=B)')
        ax.set_ylabel('Count')
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['A', 'B'])
        ymax = max(global_max, max(heights) if heights else 0)
        ax.set_ylim(0, ymax + max(1, int(0.1 * ymax)))
        ax.grid(True, alpha=0.3)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, str(h), ha='center', va='bottom')
    plt.tight_layout()
    out = Path(output_dir) / 'pilot_pair_distributions.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pair_dimension_analysis(analysis: Dict[str, Any], output_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    # Plot 1: Avg preference by topic
    topics = list(analysis['topic_stats'].keys())
    if topics:
        avgs = [np.mean(analysis['topic_stats'][t]) for t in topics]
        ax = axes[0, 0]
        bars = ax.bar(range(len(topics)), avgs, alpha=0.8, color=sns.color_palette("husl", len(topics)))
        ax.set_title('Avg Preference by Topic (A<0 … B>0)')
        ax.set_xlabel('Topic')
        ax.set_ylabel('Avg Preference')
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels([abbreviate_topic(t) for t in topics], rotation=45, ha='right')
        ax.set_ylim(-2, 2)
        for bar, v in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2., v + (0.05 if v>=0 else -0.15), f"{v:.2f}", ha='center', va='bottom')
    # Plot 2: Avg preference by comment_num
    cns = sorted(analysis['comment_num_stats'].keys())
    if cns:
        avgs = [np.mean(analysis['comment_num_stats'][cn]) for cn in cns]
        ax = axes[0, 1]
        bars = ax.bar(cns, avgs, alpha=0.8, color=sns.color_palette("husl", len(cns)))
        ax.set_title('Avg Preference by Comment Number')
        ax.set_xlabel('Comment Number')
        ax.set_ylabel('Avg Preference')
        ax.set_ylim(-2, 2)
        for bar, v in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2., v + (0.05 if v>=0 else -0.15), f"{v:.2f}", ha='center', va='bottom')
    # Plot 3: Heatmap Topic x Question (avg pref)
    # Build frame of mean prefs per topic-question
    rows = []
    for rec in analysis['detailed']:
        if rec['prefs']:
            for q, pref in rec['prefs'].items():
                rows.append({'topic': rec['topic'], 'question': q, 'pref': pref})
    if rows:
        df = pd.DataFrame(rows)
        pv = df.groupby(['topic', 'question'])['pref'].mean().unstack(fill_value=np.nan)
        ax = axes[0, 2]
        im = ax.imshow(pv.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        ax.set_title('Avg Preference: Topic vs Question (A<0 … B>0)')
        ax.set_xlabel('Question')
        ax.set_ylabel('Topic')
        ax.set_xticks(range(len(pv.columns)))
        ax.set_yticks(range(len(pv.index)))
        ax.set_xticklabels([abbreviate_question(c) for c in pv.columns], rotation=45, ha='right')
        ax.set_yticklabels([abbreviate_topic(t) for t in pv.index])
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                val = pv.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)
        plt.colorbar(im, ax=axes[0, 2])
    # Plot 4/5: Inter-annotator distance like rating (avg abs diff across answered questions)
    try:
        per_item: Dict[tuple, Dict[str, Dict[str, int]]] = defaultdict(dict)
        for rec in analysis['detailed']:
            ann_id = rec.get('annotation_id', '')
            base_id = ann_id.replace('_question', '')
            topic = rec.get('topic', 'Unknown')
            comment_num = rec.get('comment_num', 0)
            user_id = rec.get('user_id', 'unknown')
            ratings = rec.get('ratings', {})
            key = (topic, comment_num, base_id)
            per_item[key][user_id] = ratings
        tc_to_dist: Dict[tuple, List[float]] = defaultdict(list)
        tc_total: Dict[tuple, int] = defaultdict(int)
        tc_used: Dict[tuple, int] = defaultdict(int)
        tc_insuff: Dict[tuple, int] = defaultdict(int)
        tc_noover: Dict[tuple, int] = defaultdict(int)
        for (topic, cn, base), umap in per_item.items():
            users = list(umap.keys())
            key = (topic, cn)
            tc_total[key] += 1
            if len(users) < 2:
                tc_insuff[key] += 1
                continue
            pair_diffs = []
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    ri, rj = umap[users[i]], umap[users[j]]
                    diffs = []
                    for q in PAIR_QUESTIONS:
                        if q in ri and q in rj:
                            try:
                                diffs.append(abs(int(ri[q]) - int(rj[q])))
                            except Exception:
                                pass
                    if diffs:
                        pair_diffs.append(float(np.mean(diffs)))
            if pair_diffs:
                tc_to_dist[key].append(float(np.mean(pair_diffs)))
                tc_used[key] += 1
            else:
                tc_noover[key] += 1
        # Build matrix
        topics_all = sorted(set(t for (t, _cn) in tc_total.keys()))
        cns_all = sorted(set(cn for (_t, cn) in tc_total.keys()))
        if topics_all and cns_all:
            heat = np.zeros((len(topics_all), len(cns_all)))
            heat[:] = np.nan
            ti = {t: i for i, t in enumerate(topics_all)}
            ci = {cn: j for j, cn in enumerate(cns_all)}
            for (t, cn), vals in tc_to_dist.items():
                heat[ti[t], ci[cn]] = float(np.mean(vals))
            ax = axes[1, 0]
            masked = np.ma.masked_invalid(heat)
            im = ax.imshow(masked, cmap='Purples', aspect='auto', vmin=0, vmax=4)
            ax.set_title('Inter-annotator Distance (avg across questions)')
            ax.set_xlabel('Comment Number')
            ax.set_ylabel('Topic')
            ax.set_xticks(range(len(cns_all)))
            ax.set_yticks(range(len(topics_all)))
            ax.set_xticklabels(cns_all)
            ax.set_yticklabels([abbreviate_topic(t) for t in topics_all])
            for i in range(len(topics_all)):
                for j in range(len(cns_all)):
                    val = heat[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)
            plt.colorbar(im, ax=ax)
            # Missing report
            rpt = Path(output_dir) / 'pilot_pair_missing_report.txt'
            with open(rpt, 'w', encoding='utf-8') as f:
                f.write('Missing Data Report for Pair Task (Topic x Comment Number)\n')
                f.write('='*70 + '\n')
                f.write('topic_abbr, comment_num, total_items, used_items, skipped_insufficient_annot, skipped_no_overlap\n')
                for t in topics_all:
                    for cn in cns_all:
                        key = (t, cn)
                        f.write(f"{abbreviate_topic(t)}, {cn}, {tc_total.get(key,0)}, {tc_used.get(key,0)}, {tc_insuff.get(key,0)}, {tc_noover.get(key,0)}\n")
        else:
            axes[1, 0].axis('off')
    except Exception:
        axes[1, 0].axis('off')
    # Plot 6: Correlation between questions (preference scores)
    rows = []
    for rec in analysis['detailed']:
        row = {'id': rec['annotation_id']}
        for q in PAIR_QUESTIONS:
            if q in rec['prefs']:
                row[abbreviate_question(q)] = rec['prefs'][q]
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        cols = [c for c in df.columns if c != 'id']
        if len(cols) > 1:
            corr = df[cols].corr()
            ax = axes[1, 1]
            im = ax.imshow(corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax.set_title('Preference Correlation Matrix')
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha='right')
            ax.set_yticklabels(cols)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{corr.values[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)
            plt.colorbar(im, ax=ax)

    # Replace top-left overview with per-question A/B/Tie stacked bars and head-to-head matrix
    # Per-question outcome stacked bars
    ax_out = axes[0, 0]
    categories = ['A', 'Tie', 'B']
    q_labels = [abbreviate_question(q) for q in PAIR_QUESTIONS]
    data = np.array([[analysis['per_question_outcomes'][q][c] for q in PAIR_QUESTIONS] for c in categories])
    cum = np.zeros(len(PAIR_QUESTIONS))
    colors = ['#66c2a5', '#ffd92f', '#fc8d62']
    for i, cat in enumerate(categories):
        ax_out.bar(range(len(PAIR_QUESTIONS)), data[i], bottom=cum, label=cat, color=colors[i])
        cum += data[i]
    ax_out.set_title('Per-Question Outcomes (A vs B vs Tie)')
    ax_out.set_xticks(range(len(PAIR_QUESTIONS)))
    ax_out.set_xticklabels(q_labels, rotation=30, ha='right')
    ax_out.set_ylabel('Count')
    ax_out.legend(loc='upper right', fontsize=8)

    # Head-to-head matrix heatmap (B win rate vs A) over all pairs
    hh = analysis['pair_headtohead']
    if hh:
        pairs = list(hh.keys())
        models_a = sorted(set(a for a, _ in pairs))
        models_b = sorted(set(b for _, b in pairs))
        mat = np.zeros((len(models_a), len(models_b)))
        mat[:] = np.nan
        for i, ma in enumerate(models_a):
            for j, mb in enumerate(models_b):
                if (ma, mb) in hh:
                    stats = hh[(ma, mb)]
                    total = stats['A'] + stats['B'] + stats['Tie']
                    if total > 0:
                        mat[i, j] = stats['B'] / total  # B win rate
        ax_hh = axes[0, 1]
        im = ax_hh.imshow(mat, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
        ax_hh.set_title('Head-to-Head: B Win Rate (model_a vs model_b)')
        ax_hh.set_xlabel('model_b')
        ax_hh.set_ylabel('model_a')
        ax_hh.set_xticks(range(len(models_b)))
        ax_hh.set_yticks(range(len(models_a)))
        ax_hh.set_xticklabels(models_b, rotation=45, ha='right')
        ax_hh.set_yticklabels(models_a)
        for i in range(len(models_a)):
            for j in range(len(models_b)):
                if not np.isnan(mat[i, j]):
                    ax_hh.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)
        plt.colorbar(im, ax=ax_hh)

    # Save per-pair summary CSV with model pair and mean preference and ids
    rows_sum = []
    for rec in analysis['pair_records']:
        if rec['prefs']:
            rows_sum.append({
                'annotation_id': rec['annotation_id'],
                'topic': rec['topic'],
                'comment_num': rec['comment_num'],
                'model_a': rec['model_a'],
                'model_b': rec['model_b'],
                'mean_pref': float(np.mean(list(rec['prefs'].values())))
            })
    if rows_sum:
        df_sum = pd.DataFrame(rows_sum)
        df_sum.to_csv(Path(output_dir) / 'pilot_pair_per_pair_summary.csv', index=False)
    # Final layout
    plt.tight_layout()
    out = Path(output_dir) / 'pilot_pair_dimension_analysis.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()

    # Also write per-question head-to-head win rates across all topics and sample nums
    # Output CSV: question, model_a, model_b, A_wins, B_wins, Ties, total, winrate_A, winrate_B
    rows = []
    for (q, ma, mb), cnt in analysis['per_question_h2h'].items():
        a, b, t = cnt['A'], cnt['B'], cnt['Tie']
        total = a + b + t
        if total > 0:
            rows.append({
                'question': abbreviate_question(q),
                'model_a': ma,
                'model_b': mb,
                'A_wins': a,
                'B_wins': b,
                'Ties': t,
                'total': total,
                'winrate_A': a/total,
                'winrate_B': b/total,
            })
    if rows:
        df_wr = pd.DataFrame(rows)
        df_wr.sort_values(by=['question','model_a','model_b'], inplace=True)
        df_wr.to_csv(Path(output_dir) / 'pilot_pair_winrates_per_question.csv', index=False)


def write_pair_summary_report(analysis: Dict[str, Any], output_dir: str) -> None:
    rpt = Path(output_dir) / 'pilot_pair_summary_report.txt'
    with open(rpt, 'w', encoding='utf-8') as f:
        f.write('PILOT PAIR SUMMARY REPORT\n')
        f.write('='*40 + '\n\n')
        total = len(analysis['detailed'])
        topics_present = sorted(list(analysis['topic_stats'].keys()))
        comment_nums_present = sorted(list(analysis['comment_num_stats'].keys()))
        f.write(f'Total annotations: {total}\n')
        f.write(f'Topics: {len(analysis['topic_stats'])}\n')
        f.write(f'Comment nums: {len(analysis['comment_num_stats'])}\n')
        f.write(f'Topic list: {topics_present}\n')
        f.write(f'Comment num list: {comment_nums_present}\n\n')
        f.write('Distributions per question:\n')
        for q in PAIR_QUESTIONS:
            vals = analysis['distributions'][q]
            if vals:
                cnt = Counter(vals)
                f.write(f"  {abbreviate_question(q)} (n={len(vals)}): ")
                f.write(', '.join([f"{k}:{cnt[k]}" for k in sorted(cnt.keys())]) + '\n')


def main():
    annotation_dir = "/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/pilot_pair"
    raw_data_dir = "/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/data_files/processed"
    output_dir = "/ibex/project/c2328/LLMs-Scalable-Deliberation/results/pilot_pair_analysis"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("PILOT PAIR ANALYSIS")
    print("="*60)
    print(f"Annotation dir: {annotation_dir}")
    print(f"Raw data dir: {raw_data_dir}")
    print(f"Output dir: {output_dir}")

    raw_pair = load_raw_pair_data(raw_data_dir)
    simple_raw = load_simple_raw_data(raw_data_dir)
    anns = extract_pilot_pair_annotations(annotation_dir)
    if not anns:
        print("No pair annotations found.")
        return
    anns = map_pair_annotations_to_raw(anns, raw_pair, simple_raw)

    analysis = analyze_pair_scores(anns)

    print("Creating plots...")
    # Only keep distributions; dimension analysis figure removed per request
    plot_pair_distributions(analysis, output_dir)

    # Show explicit topic and comment_num coverage
    topics_present = sorted(list(analysis['topic_stats'].keys()))
    comment_nums_present = sorted(list(analysis['comment_num_stats'].keys()))
    print(f"Topics present: {topics_present if topics_present else '[]'}")
    print(f"Comment numbers present: {comment_nums_present if comment_nums_present else '[]'}")

    print("Writing report...")
    write_pair_summary_report(analysis, output_dir)

    print("\n✅ Pilot pair analysis completed!")
    print(f"📁 Output: {output_dir}")
    print("Generated files:")
    print("  - pilot_pair_distributions.png")
    print("  - pilot_pair_summary_report.txt")
    # Also saved if available: pilot_pair_per_pair_summary.csv, pilot_pair_winrates_per_question.csv

    # Create a winrates CSV directly from analysis to ensure downstream plots
    def _write_winrates_csv(analysis: Dict[str, Any], out_dir: str) -> Path:
        rows = []
        for (q, ma, mb), cnt in analysis.get('per_question_h2h', {}).items():
            a, b, t = cnt.get('A', 0), cnt.get('B', 0), cnt.get('Tie', 0)
            total = a + b + t
            if total > 0:
                rows.append({
                    'question': abbreviate_question(q),
                    'model_a': ma,
                    'model_b': mb,
                    'A_wins': a,
                    'B_wins': b,
                    'Ties': t,
                    'total': total,
                    'winrate_A': a/total,
                    'winrate_B': b/total,
                })
        csv_path = Path(out_dir) / 'pilot_pair_winrates_per_question.csv'
        if rows:
            df_wr = pd.DataFrame(rows)
            df_wr.sort_values(by=['question','model_a','model_b'], inplace=True)
            df_wr.to_csv(csv_path, index=False)
        return csv_path

    winrate_csv = _write_winrates_csv(analysis, output_dir)

    # Create a figure showing model win rates per question (across all topics and sample nums)
    if winrate_csv.exists():
        try:
            df_wr = pd.read_csv(winrate_csv)
            # Build pivot: question x model, value = aggregate winrate against others (mean of winrate_A when model is A and winrate_B when model is B)
            models = sorted(set(df_wr['model_a']).union(set(df_wr['model_b'])))
            questions = sorted(set(df_wr['question']))
            data = np.zeros((len(questions), len(models)))
            data[:] = np.nan
            for qi, q in enumerate(questions):
                for mi, m in enumerate(models):
                    rows_a = df_wr[(df_wr['question'] == q) & (df_wr['model_a'] == m)]
                    rows_b = df_wr[(df_wr['question'] == q) & (df_wr['model_b'] == m)]
                    vals = []
                    if not rows_a.empty:
                        vals.extend(rows_a['winrate_A'].tolist())
                    if not rows_b.empty:
                        vals.extend(rows_b['winrate_B'].tolist())
                    if vals:
                        data[qi, mi] = float(np.mean(vals))
            # Plot heatmap
            plt.figure(figsize=(max(8, len(models)*0.9), 6))
            im = plt.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
            plt.title('Model Win Rates per Question (A=first, B=second)')
            plt.xlabel('Model')
            plt.ylabel('Question')
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.yticks(range(len(questions)), questions)
            for i in range(len(questions)):
                for j in range(len(models)):
                    if not np.isnan(data[i, j]):
                        plt.text(j, i, f"{data[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(str(Path(output_dir) / 'pilot_pair_model_winrates.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("  - pilot_pair_model_winrates.png")

            # Additional figure: per-question pairwise model-vs-model win rates
            questions = sorted(set(df_wr['question']))
            models_all = sorted(set(df_wr['model_a']).union(set(df_wr['model_b'])))
            n_q = len(questions)
            cols = 2
            rows = int(np.ceil(n_q / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(max(10, len(models_all)*0.9), rows*4))
            if rows == 1 and cols == 1:
                axs = np.array([[axs]])
            elif rows == 1:
                axs = np.array([axs])
            for idx, q in enumerate(questions):
                r = idx // cols
                c = idx % cols
                ax = axs[r, c]
                mat = np.zeros((len(models_all), len(models_all)))
                mat[:] = np.nan
                for i, ma in enumerate(models_all):
                    for j, mb in enumerate(models_all):
                        if ma == mb:
                            continue
                        rows_q = df_wr[(df_wr['question'] == q) & (df_wr['model_a'] == ma) & (df_wr['model_b'] == mb)]
                        if not rows_q.empty:
                            # B win rate for pair (ma, mb)
                            mat[i, j] = float(rows_q.iloc[0]['B_wins'] / rows_q.iloc[0]['total']) if rows_q.iloc[0]['total'] > 0 else np.nan
                im = ax.imshow(mat, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
                ax.set_title(f"Pairwise Win Rate (B vs A) - {q}")
                ax.set_xlabel('model_b')
                ax.set_ylabel('model_a')
                ax.set_xticks(range(len(models_all)))
                ax.set_yticks(range(len(models_all)))
                ax.set_xticklabels(models_all, rotation=45, ha='right')
                ax.set_yticklabels(models_all)
                for i in range(len(models_all)):
                    for j in range(len(models_all)):
                        if not np.isnan(mat[i, j]):
                            ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Hide any unused subplots
            for k in range(n_q, rows*cols):
                r = k // cols
                c = k % cols
                axs[r, c].axis('off')
            plt.tight_layout()
            plt.savefig(str(Path(output_dir) / 'pilot_pair_pairwise_winrates.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("  - pilot_pair_pairwise_winrates.png")
        except Exception as e:
            print(f"Warning: could not create winrate heatmap: {e}")


if __name__ == "__main__":
    main()
