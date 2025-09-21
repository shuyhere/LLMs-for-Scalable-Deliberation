#!/usr/bin/env python3
"""
Inspect all evaluations for a given summary_id:
- Find every pairwise comparison the summary participated in (as A or B)
- List opponent summary_id, per-user outcome (win/loss) per dimension
- Attach parameters for both sides (topic, model, comment_num)

Usage:
  python inspect_summary_comparisons.py --summary-id <SUMMARY_ID>

Outputs (in results/dataset_visulization/analysis_annotation/inspect/):
- comparisons_<SUMMARY_ID>.csv: per-evaluation rows with outcomes and params
- comparisons_<SUMMARY_ID>_opponent_aggregate.csv: aggregated wins/losses vs each opponent per dimension
"""

from pathlib import Path
from typing import Dict, List
import argparse

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
ANNOTATED_CSV = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full/annotated_instances.csv'
TRIPLET_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring.csv'
RAW_SUMMARIES_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_simple.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation/inspect'


def get_dimension_questions() -> Dict[str, Dict[str, str]]:
    return {
        'perspective': {
            'comparison': "Which summary is more representative of your perspective?",
        },
        'informativeness': {
            'comparison': "Which summary is more informative?",
        },
        'neutrality': {
            'comparison': "Which summary presents a more neutral and balanced view of the issue?",
        },
        'policy': {
            'comparison': "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?",
        },
    }


def parse_choice(row: pd.Series, comp_q: str) -> float:
    col_a = f"{comp_q}:::scale_1"
    col_b = f"{comp_q}:::scale_2"
    if col_a in row and pd.notna(row[col_a]) and str(row[col_a]).strip() != "":
        return 1.0
    if col_b in row and pd.notna(row[col_b]) and str(row[col_b]).strip() != "":
        return 0.0
    return np.nan


def load_metadata() -> pd.DataFrame:
    raw = pd.read_csv(RAW_SUMMARIES_CSV)
    # Keep key params
    return raw[['id', 'topic', 'model', 'comment_num']].rename(columns={'id': 'summary_id'})


def main():
    parser = argparse.ArgumentParser(description='Inspect all evaluations for a summary_id')
    parser.add_argument('--summary-id', required=True, help='Target summary_id to inspect')
    args = parser.parse_args()

    target_id = args.summary_id
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load triplet comparisons to find all instances the summary participated in
    trip = pd.read_csv(TRIPLET_CSV)
    comps = trip[trip['type'] == 'comparison'][['id', 'model_a', 'model_b', 'summary_a_id', 'summary_b_id']]
    involved = comps[(comps['summary_a_id'] == target_id) | (comps['summary_b_id'] == target_id)].copy()
    if involved.empty:
        print(f"No comparisons found for summary_id={target_id}")
        return
    print(f"Found {len(involved)} comparison instances for summary_id={target_id}")

    # Load all annotated comparison rows
    ann = pd.read_csv(ANNOTATED_CSV)
    ann_comp = ann[ann['instance_id'].isin(involved['id'])].copy()

    dims = get_dimension_questions()

    records: List[Dict[str, object]] = []
    for _, comp_row in involved.iterrows():
        instance_id = comp_row['id']
        self_role = 'A' if comp_row['summary_a_id'] == target_id else 'B'
        opp_id = comp_row['summary_b_id'] if self_role == 'A' else comp_row['summary_a_id']
        model_self = comp_row['model_a'] if self_role == 'A' else comp_row['model_b']
        model_opp = comp_row['model_b'] if self_role == 'A' else comp_row['model_a']

        ann_rows = ann_comp[ann_comp['instance_id'] == instance_id]
        if ann_rows.empty:
            # No annotations recorded for this instance (should be rare)
            rec = {
                'instance_id': instance_id,
                'dimension': 'NA',
                'user': None,
                'choice': None,
                'self_role': self_role,
                'outcome': None,
                'opponent_summary_id': opp_id,
                'model_self': model_self,
                'model_opponent': model_opp,
            }
            records.append(rec)
            continue

        for _, ann_row in ann_rows.iterrows():
            user = ann_row.get('user') or ann_row.get('user_id')
            for dim, dmap in dims.items():
                choice_a = parse_choice(ann_row, dmap['comparison'])
                if pd.isna(choice_a):
                    continue
                # Determine outcome for our summary
                if self_role == 'A':
                    outcome = 'win' if choice_a == 1.0 else 'loss'
                else:
                    outcome = 'win' if choice_a == 0.0 else 'loss'

                rec = {
                    'instance_id': instance_id,
                    'dimension': dim,
                    'user': user,
                    'choice': 'A' if choice_a == 1.0 else 'B',
                    'self_role': self_role,
                    'outcome': outcome,
                    'opponent_summary_id': opp_id,
                    'model_self': model_self,
                    'model_opponent': model_opp,
                }
                records.append(rec)

    detail_df = pd.DataFrame(records)
    if 'model_opponent' not in detail_df.columns:
        detail_df['model_opponent'] = np.nan
    if 'model_self' not in detail_df.columns:
        detail_df['model_self'] = np.nan
    if 'opponent_summary_id' not in detail_df.columns:
        detail_df['opponent_summary_id'] = np.nan

    print(f"Detail rows collected: {len(detail_df)}")

    # Attach parameters for both sides
    meta = load_metadata()
    # Attach self metadata (same for all rows)
    self_meta = meta[meta['summary_id'] == target_id]
    if not self_meta.empty:
        self_meta = self_meta.iloc[0]
        detail_df['summary_id_self'] = target_id
        detail_df['topic_self'] = self_meta.get('topic')
        detail_df['comment_num_self'] = self_meta.get('comment_num')
    else:
        detail_df['summary_id_self'] = target_id
        detail_df['topic_self'] = np.nan
        detail_df['comment_num_self'] = np.nan

    # Attach opponent metadata by merge on opponent_summary_id
    detail_df = detail_df.merge(meta.add_suffix('_opponent'), left_on='opponent_summary_id', right_on='summary_id_opponent', how='left')
    # If merge created duplicate opponent model columns, coalesce to a single 'model_opponent'
    if 'model_opponent_x' in detail_df.columns or 'model_opponent_y' in detail_df.columns:
        left = detail_df.get('model_opponent_x')
        right = detail_df.get('model_opponent_y')
        if left is not None and right is not None:
            detail_df['model_opponent'] = left.fillna(right)
        elif left is not None:
            detail_df['model_opponent'] = left
        elif right is not None:
            detail_df['model_opponent'] = right
        # Drop intermediate columns
        for c in ['model_opponent_x', 'model_opponent_y']:
            if c in detail_df.columns:
                detail_df.drop(columns=[c], inplace=True)

    # Reorder/clean columns
    cols = [
        'instance_id', 'dimension', 'user', 'self_role', 'choice', 'outcome',
        'model_self', 'model_opponent', 'summary_id_self', 'opponent_summary_id',
        'topic_self', 'comment_num_self', 'topic_opponent', 'comment_num_opponent'
    ]
    present_cols = [c for c in cols if c in detail_df.columns]
    detail_df = detail_df[present_cols]

    # Aggregate by opponent and dimension
    # Guard against missing model_opponent column
    group_keys = ['opponent_summary_id', 'dimension']
    if 'model_opponent' in detail_df.columns:
        group_keys.insert(1, 'model_opponent')
    agg = detail_df.groupby(group_keys).agg(
        n=('outcome', 'count'),
        wins=('outcome', lambda s: int((s == 'win').sum())),
        losses=('outcome', lambda s: int((s == 'loss').sum())),
    ).reset_index()
    agg['win_rate'] = agg.apply(lambda r: float(r['wins']) / r['n'] if r['n'] > 0 else np.nan, axis=1)

    # Save
    out_csv = OUTPUT_DIR / f'comparisons_{target_id}.csv'
    out_agg_csv = OUTPUT_DIR / f'comparisons_{target_id}_opponent_aggregate.csv'
    detail_df.to_csv(out_csv, index=False)
    agg.to_csv(out_agg_csv, index=False)
    print(f"Saved: {out_csv}\nSaved: {out_agg_csv}")


if __name__ == '__main__':
    main()


