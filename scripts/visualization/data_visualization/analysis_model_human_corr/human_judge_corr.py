#!/usr/bin/env python3
"""
Script to process completed human-LLM correlation JSON files.
Filters out checkpoint files and analyzes correlation data.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr


class HumanJudgeCorrelationProcessor:
    """Process and analyze human-LLM correlation data from JSON files."""
    
    def __init__(self, data_dir: str, model_order: Optional[List[str]] = None, overrides: Optional[List[str]] = None):
        """
        Initialize the processor with the data directory.
        
        Args:
            data_dir: Path to directory containing correlation JSON files
        """
        self.data_dir = Path(data_dir)
        self.completed_files = []
        self.correlation_data = []
        
        # Model name mapping for cleaner display
        self.model_name_mapping = {
            'web-rev-claude-sonnet-4-20250514': 'Claude-Sonnet-4',
            'web-rev-claude-opus-4-20250514': 'Claude-Opus-4',
            'web-rev-claude-3-7-sonnet-20250219': 'Claude-Sonnet-3.7',
            'qwen3-235b-a22b': 'Qwen3-235B',
            'qwen3-32b': 'Qwen3-32B',
            'qwen3-30b-a3b': 'Qwen3-30B',
            'qwen3-14b': 'Qwen3-14B',
            'qwen3-8b': 'Qwen3-8B',
            'qwen3-4b': 'Qwen3-4B',
            'qwen3-1.7b': 'Qwen3-1.7B',
            'qwen3-0.6b': 'Qwen3-0.6B',
            'gpt-5': 'GPT-5',
            'gpt-5-mini': 'GPT-5-Mini',
            'gpt-5-nano': 'GPT-5-Nano',
            'gpt-4o-mini': 'GPT-4o-Mini',
            'gemini-2.5-pro': 'Gemini-2.5-Pro',
            'gemini-2.5-flash': 'Gemini-2.5-Flash',
            'gemini-2.5-flash-lite': 'Gemini-2.5-Flash-Lite',
            'grok-4-latest': 'Grok-4-Latest',
            'deepseek-chat': 'DeepSeek-Chat',
            'deepseek-reasoner': 'DeepSeek-Reasoner'
        }
        
        # Dimension name mapping and desired display order
        self.dimension_mapping = {
            'perspective_representation': 'Representiveness',
            'informativeness': 'Informativeness',
            'neutrality_balance': 'Neutrality',
            'policy_approval': 'Policy Approval'
        }
        self.dimension_display_order = [
            'Representiveness',
            'Informativeness',
            'Neutrality',
            'Policy Approval',
        ]
        # Optional model display order (list of display names)
        self.model_display_order: Optional[List[str]] = model_order

        # Optional hard overrides for specific cells (parsed later)
        self.overrides_raw: List[str] = overrides or []
        self.overrides_parsed: List[Dict] = []

    def _normalize_dim(self, name: Optional[str]) -> Optional[str]:
        """Normalize various dimension aliases to canonical keys used internally.

        Canonical keys:
        - 'perspective_representation'
        - 'informativeness'
        - 'neutrality_balance'
        - 'policy_approval'
        """
        if name is None:
            return None
        key = str(name).strip().lower().replace('-', '_').replace(' ', '_')
        aliases = {
            # Representativeness / perspective
            'perspective_representation': 'perspective_representation',
            'perspective': 'perspective_representation',
            'representation': 'perspective_representation',
            'representativeness': 'perspective_representation',
            'representiveness': 'perspective_representation',

            # Informativeness
            'informativeness': 'informativeness',
            'informative': 'informativeness',
            'info': 'informativeness',

            # Neutrality / balance
            'neutrality_balance': 'neutrality_balance',
            'neutrality': 'neutrality_balance',
            'balance': 'neutrality_balance',

            # Policy approval
            'policy_approval': 'policy_approval',
            'approval': 'policy_approval',
            'policy': 'policy_approval',
        }
        return aliases.get(key, name)
        
    def find_completed_files(self) -> List[str]:
        """
        Find all completed correlation JSON files (excluding checkpoint files).
        
        Returns:
            List of completed JSON file paths
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        # Find all JSON files that don't start with 'checkpoint_'
        json_files = list(self.data_dir.glob("*.json"))
        completed_files = [f for f in json_files if not f.name.startswith("checkpoint_")]
        
        self.completed_files = [str(f) for f in completed_files]
        print(f"Found {len(self.completed_files)} completed correlation files:")
        for file in self.completed_files:
            print(f"  - {Path(file).name}")
            
        return self.completed_files
    
    def _compute_phi(self, human: List[int], model: List[int]) -> float:
        """Compute Phi coefficient (equivalent to MCC for binary) from paired binary labels {1,2}.

        Returns NaN if insufficient data or degenerate contingency table.
        """
        if not human or not model or len(human) != len(model):
            return float('nan')
        # Map {1,2} -> {0,1}
        h = np.asarray([1 if x == 2 else 0 for x in human], dtype=float)
        m = np.asarray([1 if x == 2 else 0 for x in model], dtype=float)
        n11 = float(np.sum((h == 1) & (m == 1)))
        n10 = float(np.sum((h == 1) & (m == 0)))
        n01 = float(np.sum((h == 0) & (m == 1)))
        n00 = float(np.sum((h == 0) & (m == 0)))
        denom = np.sqrt((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01))
        if denom == 0:
            return float('nan')
        return ((n11 * n00) - (n10 * n01)) / denom

    def _compute_kappa(self, human: List[int], model: List[int]) -> float:
        """Compute Cohen's kappa for discrete ratings (e.g., 1-5).

        Returns NaN if insufficient data or degenerate.
        """
        if not human or not model or len(human) != len(model):
            return float('nan')
        h = np.asarray(human)
        m = np.asarray(model)
        classes = np.union1d(h, m)
        # Build confusion matrix
        conf = np.zeros((len(classes), len(classes)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for hi, mi in zip(h, m):
            conf[class_to_idx[hi], class_to_idx[mi]] += 1.0
        n = conf.sum()
        if n == 0:
            return float('nan')
        po = np.trace(conf) / n
        row_marg = conf.sum(axis=1) / n
        col_marg = conf.sum(axis=0) / n
        pe = float(np.sum(row_marg * col_marg))
        denom = 1.0 - pe
        if denom == 0:
            return float('nan')
        return (po - pe) / denom

    def load_correlation_data(self) -> List[Dict]:
        """
        Load correlation data from all completed JSON files.
        
        Returns:
            List of correlation data dictionaries
        """
        if not self.completed_files:
            self.find_completed_files()
            
        correlation_data = []
        
        for file_path in self.completed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract model name from filename
                model_name = Path(file_path).stem.replace("human_llm_correlation_", "")
                
                # Extract experiment metadata
                metadata = data.get("experiment_metadata", {})
                
                # Extract correlation results (precomputed)
                correlations = data.get("correlations", {})
                # Also get raw results to allow recomputation for comparison (Phi)
                rating_results = data.get("rating_results", [])
                comparison_results = data.get("comparison_results", [])

                # First, load precomputed metrics as-is
                if correlations:
                    for metric_name, metric_data in correlations.items():
                        correlation_data.append({
                            'model': model_name,
                            'metric': metric_name,
                            'pearson_r': metric_data.get('pearson_r'),
                            'pearson_p': metric_data.get('pearson_p'),
                            'spearman_r': metric_data.get('spearman_r'),
                            'spearman_p': metric_data.get('spearman_p'),
                            'cohen_kappa': metric_data.get('cohen_kappa'),
                            'mae': metric_data.get('mae'),
                            'n_samples': metric_data.get('n_samples'),
                            'temperature': metadata.get('temperature'),
                            'n_rating_samples': metadata.get('n_rating_samples'),
                            'n_comparison_samples': metadata.get('n_comparison_samples')
                        })

                # Recompute correlations for RATING from raw pairs (override JSON values only when enough data)
                if rating_results:
                    dim_pairs_r: Dict[str, Tuple[List[float], List[float]]] = {
                        'perspective_representation': ([], []),
                        'informativeness': ([], []),
                        'neutrality_balance': ([], []),
                        'policy_approval': ([], []),
                    }
                    for item in rating_results:
                        human_r = item.get('human_ratings', {})
                        llm_r = item.get('llm_result', {}).get('ratings', {})
                        keys = set(human_r.keys()) | set(llm_r.keys())
                        for k in keys:
                            dim = self._normalize_dim(k)
                            if dim not in dim_pairs_r:
                                continue
                            h_val = human_r.get(k, human_r.get(dim))
                            m_val = llm_r.get(k, llm_r.get(dim))
                            if h_val is None or m_val is None:
                                continue
                            try:
                                h = float(h_val)
                                m = float(m_val)
                            except (TypeError, ValueError):
                                continue
                            dim_pairs_r[dim][0].append(h)
                            dim_pairs_r[dim][1].append(m)

                    for dim, (h_list, m_list) in dim_pairs_r.items():
                        if len(h_list) < 2:
                            # Not enough data: do not override any precomputed values
                            continue
                        try:
                            pr, pp = pearsonr(h_list, m_list)
                        except Exception:
                            pr, pp = float('nan'), float('nan')
                        try:
                            sr, sp = spearmanr(h_list, m_list)
                            # spearmanr may return nan, ensure float
                            sr = float(sr) if sr is not None else float('nan')
                            sp = float(sp) if sp is not None else float('nan')
                        except Exception:
                            sr, sp = float('nan'), float('nan')
                        try:
                            kappa = self._compute_kappa([int(x) for x in h_list], [int(x) for x in m_list])
                        except Exception:
                            kappa = float('nan')
                        try:
                            mae = float(np.mean(np.abs(np.asarray(h_list) - np.asarray(m_list))))
                        except Exception:
                            mae = float('nan')

                        metric_name = f"rating_{dim}"
                        # Update existing row or append new
                        updated = False
                        for row in correlation_data:
                            if row['model'] == model_name and row['metric'] == metric_name:
                                row['pearson_r'] = pr
                                row['pearson_p'] = pp
                                row['spearman_r'] = sr
                                row['spearman_p'] = sp
                                row['cohen_kappa'] = kappa
                                row['mae'] = mae
                                row['n_samples'] = len(h_list)
                                updated = True
                                break
                        if not updated:
                            correlation_data.append({
                                'model': model_name,
                                'metric': metric_name,
                                'pearson_r': pr,
                                'pearson_p': pp,
                                'spearman_r': sr,
                                'spearman_p': sp,
                                'cohen_kappa': kappa,
                                'mae': mae,
                                'n_samples': len(h_list),
                                'temperature': metadata.get('temperature'),
                                'n_rating_samples': metadata.get('n_rating_samples'),
                                'n_comparison_samples': metadata.get('n_comparison_samples')
                            })

                # Then, recompute Spearman for comparison tasks (1-5 scale)
                if comparison_results:
                    # Collect paired labels per dimension
                    dim_to_pairs: Dict[str, Tuple[List[int], List[int]]] = {
                        'perspective_representation': ([], []),
                        'informativeness': ([], []),
                        'neutrality_balance': ([], []),
                        'policy_approval': ([], []),
                    }
                    for item in comparison_results:
                        human_cmp = item.get('human_comparisons', {})
                        llm_cmp = item.get('llm_result', {}).get('comparisons', {})
                        keys = set(human_cmp.keys()) | set(llm_cmp.keys())
                        for k in keys:
                            dim = self._normalize_dim(k)
                            if dim not in dim_to_pairs:
                                continue
                            h = human_cmp.get(k, human_cmp.get(dim))
                            m = llm_cmp.get(k, llm_cmp.get(dim))
                            if h in (1, 2, 3, 4, 5) and m in (1, 2, 3, 4, 5):
                                dim_to_pairs[dim][0].append(h)
                                dim_to_pairs[dim][1].append(m)

                    for dim, (h_list, m_list) in dim_to_pairs.items():
                        # For comparison task with 1-5 scale, use Spearman correlation instead of Phi
                        if len(h_list) < 2:
                            # Not enough data: do not override or create entries
                            continue
                        try:
                            sr, sp = spearmanr(h_list, m_list)
                            sr = float(sr) if sr is not None else float('nan')
                            sp = float(sp) if sp is not None else float('nan')
                        except Exception:
                            sr, sp = float('nan'), float('nan')
                        
                        metric_name = f"comparison_{dim}"
                        # Try to find if we already appended an entry for this metric; if so, update/add 'spearman_r'
                        updated = False
                        for row in correlation_data:
                            if row['model'] == model_name and row['metric'] == metric_name:
                                row['spearman_r'] = sr
                                row['spearman_p'] = sp
                                row['n_samples'] = len(h_list)
                                updated = True
                                break
                        if not updated:
                            correlation_data.append({
                                'model': model_name,
                                'metric': metric_name,
                                'phi': None,  # Not applicable for 1-5 scale
                                'pearson_r': None,
                                'pearson_p': None,
                                'spearman_r': sr,
                                'spearman_p': sp,
                                'cohen_kappa': None,
                                'mae': None,
                                'n_samples': len(h_list),
                                'temperature': metadata.get('temperature'),
                                'n_rating_samples': metadata.get('n_rating_samples'),
                                'n_comparison_samples': metadata.get('n_comparison_samples')
                            })
                
                print(f"Loaded data for model: {model_name}")
                
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        self.correlation_data = correlation_data
        print(f"Total correlation metrics loaded: {len(correlation_data)}")
        return correlation_data
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame from the correlation data.
        
        Returns:
            DataFrame with correlation metrics
        """
        if not self.correlation_data:
            self.load_correlation_data()
            
        df = pd.DataFrame(self.correlation_data)
        
        # Separate rating and comparison metrics
        df['task_type'] = df['metric'].str.split('_').str[0]
        df['dimension'] = df['metric'].str.split('_', n=1).str[1].apply(self._normalize_dim)
        
        # Apply model name mapping
        df['model_display'] = df['model'].map(self.model_name_mapping).fillna(df['model'])
        # Build default model order from mapping insertion order if not provided
        if not self.model_display_order:
            present = list(df['model_display'].unique())
            mapping_order = [self.model_name_mapping[k] for k in self.model_name_mapping.keys() if self.model_name_mapping.get(k) in present]
            # Append any remaining models not in mapping order
            remaining = [m for m in present if m not in mapping_order]
            self.model_display_order = mapping_order + remaining
        # Enforce model display order
        df['model_display'] = pd.Categorical(
            df['model_display'], categories=self.model_display_order, ordered=True
        )
        
        # Apply dimension name mapping and enforce column order
        df['dimension_display'] = df['dimension'].map(self.dimension_mapping).fillna(df['dimension'])
        df['dimension_display'] = pd.Categorical(
            df['dimension_display'], categories=self.dimension_display_order, ordered=True
        )
        
        # Apply any hard overrides requested
        df = self._apply_overrides_df(df)

        return df

    def _apply_overrides_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply user-provided overrides to the metrics dataframe.

        Override format (repeatable):
        "model,task,dimension,metric,value"
        - model: raw model id or display name (case-insensitive)
        - task: 'rating' or 'comparison'
        - dimension: e.g., 'neutrality' (aliases allowed)
        - metric: one of 'pearson_r', 'spearman_r', 'cohen_kappa', 'mae'
        - value: float
        """
        if not self.overrides_raw:
            return df

        # Parse once
        if not self.overrides_parsed:
            for item in self.overrides_raw:
                try:
                    parts = [p.strip() for p in item.split(',')]
                    if len(parts) != 5:
                        continue
                    model_in, task_in, dim_in, metric_in, val_in = parts
                    dim_norm = self._normalize_dim(dim_in)
                    metric_in = metric_in.strip()
                    if metric_in not in {'pearson_r', 'spearman_r', 'cohen_kappa', 'mae'}:
                        continue
                    value = float(val_in)
                    # Map model to display if possible for easier matching
                    model_display = self.model_name_mapping.get(model_in, model_in)
                    self.overrides_parsed.append({
                        'model_raw': model_in,
                        'model_display': model_display,
                        'task': task_in.lower(),
                        'dimension': dim_norm,
                        'metric': metric_in,
                        'value': value,
                    })
                except Exception:
                    continue

        # Apply overrides
        for ov in self.overrides_parsed:
            mask = (
                (
                    (df['model'].str.lower() == str(ov['model_raw']).lower()) |
                    (df['model_display'].astype(str).str.lower() == str(ov['model_display']).lower())
                ) &
                (df['task_type'] == ov['task']) &
                (df['dimension'].apply(lambda x: self._normalize_dim(x)) == ov['dimension'])
            )
            if mask.any():
                df.loc[mask, ov['metric']] = ov['value']
        return df
    
    def plot_correlation_summary(self, save_path: Optional[str] = None):
        """
        Create summary plots of correlation metrics for both rating and comparison tasks.
        
        Args:
            save_path: Optional path to save the plot
        """
        df = self.create_summary_dataframe()
        
        # Create separate figures per task type
        # Rating: show four metrics side-by-side (Pearson, Spearman, Kappa, MAE)
        rating_df = df[df['task_type'] == 'rating']
        if not rating_df.empty:
            fig_r, axes_r = plt.subplots(1, 3, figsize=(32, 9))
            metrics = [
                ('pearson_r', 'Pearson r'),
                ('spearman_r', 'Spearman r'),
                ('cohen_kappa', "Cohen's κ"),
            ]
            for i, (metric, metric_display) in enumerate(metrics):
                pivot_data = rating_df.pivot_table(
                    values=metric,
                    index='model_display',
                    columns='dimension_display',
                    aggfunc='mean'
                ).reindex(columns=self.dimension_display_order)
                # Enforce model order on index if provided
                if self.model_display_order:
                    pivot_data = pivot_data.reindex(index=self.model_display_order)
                cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
                center = 0
                sns.heatmap(pivot_data, annot=True, annot_kws={'fontsize': 14}, cmap=cmap, center=center,
                            ax=axes_r[i], cbar_kws={'label': ''}, fmt='.3f')
                # Increase colormap (colorbar) font size
                try:
                    cbar = axes_r[i].collections[0].colorbar
                    cbar.ax.tick_params(labelsize=18)
                except Exception:
                    pass
                axes_r[i].set_title(f'Rating - {metric_display}', fontsize=25, fontweight='bold')
                axes_r[i].set_xlabel('')
                axes_r[i].set_ylabel('')
                axes_r[i].tick_params(axis='x', rotation=0, labelsize=18)
                axes_r[i].tick_params(axis='y', labelrotation=30, labelsize=18)
                # Bold x-axis tick labels
                for lbl in axes_r[i].get_xticklabels():
                    lbl.set_fontweight('bold')
            plt.tight_layout()
            if save_path:
                base = Path(save_path).with_suffix('')
                out_path = str(base.parent / 'rating_correlation_summary.pdf')
                plt.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
                print(f"Plot saved to: {out_path}")
            plt.show()

        # Comparison: show Spearman correlation (1-5 scale)
        comparison_df = df[df['task_type'] == 'comparison']
        if not comparison_df.empty:
            # Use Spearman correlation for comparison task with 1-5 scale
            fig_c, ax_c = plt.subplots(1, 1, figsize=(12, 8))
            pivot_data = comparison_df.pivot_table(
                values='spearman_r',
                index='model_display',
                columns='dimension_display',
                aggfunc='mean'
            ).reindex(columns=self.dimension_display_order)
            # Enforce model order on index if provided
            if self.model_display_order:
                pivot_data = pivot_data.reindex(index=self.model_display_order)
            sns.heatmap(pivot_data, annot=True, annot_kws={'fontsize': 14}, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), center=0,
                        ax=ax_c, cbar_kws={'label': ''}, fmt='.3f')
            try:
                cbar = ax_c.collections[0].colorbar
                cbar.ax.tick_params(labelsize=20)
            except Exception:
                pass
            ax_c.set_title('Comparison Task - Spearman correlation', fontsize=25, fontweight='bold')
            ax_c.set_xlabel('')
            ax_c.set_ylabel('')
            ax_c.tick_params(axis='x', rotation=0, labelsize=14)
            ax_c.tick_params(axis='y', rotation=30, labelsize=14)
            for lbl in ax_c.get_xticklabels():
                lbl.set_fontweight('bold')
            plt.tight_layout()
            if save_path:
                base = Path(save_path).with_suffix('')
                out_path = str(base.parent / 'comparison_spearman_summary.pdf')
                plt.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
                print(f"Plot saved to: {out_path}")
            plt.show()

    def plot_rating_compare_triple(self, save_path: Optional[str] = None):
        """
        Create a single-row figure with three subplots:
        - Rating Pearson r
        - Rating Spearman r
        - Comparison Spearman r

        Args:
            save_path: Optional path used to derive the output file location.
        """
        df = self.create_summary_dataframe()

        rating_df = df[df['task_type'] == 'rating']
        comparison_df = df[df['task_type'] == 'comparison']

        fig, axes = plt.subplots(1, 3, figsize=(32, 9))
        titles = [
            ('pearson_r', 'Rating - Pearson'),
            ('spearman_r', 'Rating - Spearman'),
            ('spearman_r', 'Comparison - Spearman'),
        ]

        # Prepare dataframes for each panel
        panels = [
            (rating_df, 'pearson_r'),
            (rating_df, 'spearman_r'),
            (comparison_df, 'spearman_r'),
        ]

        for i, ((task_df, metric), (_, title)) in enumerate(zip(panels, titles)):
            ax = axes[i]
            if task_df.empty:
                ax.set_axis_off()
                ax.set_title(f'{title}\n(no data)', fontsize=20, fontweight='bold')
                continue
            pivot_data = task_df.pivot_table(
                values=metric,
                index='model_display',
                columns='dimension_display',
                aggfunc='mean'
            ).reindex(columns=self.dimension_display_order)
            if self.model_display_order:
                pivot_data = pivot_data.reindex(index=self.model_display_order)

            sns.heatmap(
                pivot_data,
                annot=True,
                annot_kws={'fontsize': 18},
                cmap=sns.color_palette("Blues", as_cmap=True),
                ax=ax,
                cbar_kws={'label': ''},
                fmt='.2f'
            )
            try:
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=18)
            except Exception:
                pass
            ax.set_title(title, fontsize=25, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='x', rotation=15, labelsize=22)
            ax.tick_params(axis='y', labelrotation=0, labelsize=20)

        plt.tight_layout()
        if save_path:
            base = Path(save_path).with_suffix('')
            out_path = str(base.parent / 'rating_compare_triple_summary.pdf')
            plt.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {out_path}")
    
    def plot_individual_heatmaps(self, save_dir: Optional[str] = None):
        """
        Create individual heatmap plots for each metric and task type.
        
        Args:
            save_dir: Optional directory to save the plots
        """
        df = self.create_summary_dataframe()
        
        # Define metrics and their display names (rating keeps Pearson; comparison uses Phi)
        metrics = [
            ('pearson_r', 'Pearson r'),
            ('spearman_r', 'Spearman r'),
            ('cohen_kappa', "Cohen's κ"),
            ('mae', 'MAE'),
        ]
        
        # Define task types
        task_types = ['rating', 'comparison']
        
        for task_type in task_types:
            task_df = df[df['task_type'] == task_type]
            if task_df.empty:
                continue
                
            if task_type == 'comparison':
                # Override to Spearman-only for comparison (1-5 scale)
                metric_list = [('spearman_r', 'Spearman correlation')]
            else:
                metric_list = metrics
            for metric, metric_display in metric_list:
                # Create individual heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                
                pivot_data = task_df.pivot_table(
                    values=metric, 
                    index='model_display', 
                    columns='dimension_display', 
                    aggfunc='mean'
                ).reindex(columns=self.dimension_display_order)
                # Enforce model order on index if provided
                if self.model_display_order:
                    pivot_data = pivot_data.reindex(index=self.model_display_order)
                
                # Use cubehelix colormap for all metrics (including MAE)
                cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
                center = 0 if metric != 'mae' else None
                
                sns.heatmap(pivot_data, annot=True, annot_kws={'fontsize': 14}, cmap=cmap, center=center,
                           ax=ax, cbar_kws={'label': ''}, fmt='.2f')
                try:
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=16)
                except Exception:
                    pass
                
                ax.set_title(f'{task_type.title()} Task - {metric_display}', 
                           fontsize=22, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.tick_params(axis='x', rotation=20, labelsize=18)
                ax.tick_params(axis='y', rotation=0, labelsize=18)
                for lbl in ax.get_xticklabels():
                    lbl.set_fontweight('bold')
                
                plt.tight_layout()
                
                if save_dir:
                    save_path = Path(save_dir) / f"{task_type}_{metric}_heatmap.pdf"
                    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
                    print(f"Individual heatmap saved to: {save_path}")
                
                plt.show()
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of the correlation analysis.
        
        Returns:
            Summary report string
        """
        df = self.create_summary_dataframe()
        
        report = []
        report.append("=" * 60)
        report.append("HUMAN-LLM CORRELATION ANALYSIS SUMMARY")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        models = df['model'].unique()
        report.append(f"Number of models analyzed: {len(models)}")
        report.append(f"Models: {', '.join(models)}")
        report.append("")
        
        # Rating task summary
        rating_df = df[df['task_type'] == 'rating']
        if not rating_df.empty:
            report.append("RATING TASK CORRELATIONS:")
            report.append("-" * 30)
            
            for dimension in rating_df['dimension'].unique():
                dim_data = rating_df[rating_df['dimension'] == dimension]
                report.append(f"\n{dimension.replace('_', ' ').title()}:")
                
                # Best performing models
                best_pearson = dim_data.loc[dim_data['pearson_r'].idxmax()]
                best_spearman = dim_data.loc[dim_data['spearman_r'].idxmax()]
                best_kappa = dim_data.loc[dim_data['cohen_kappa'].idxmax()]
                lowest_mae = dim_data.loc[dim_data['mae'].idxmin()]
                
                report.append(f"  Best Pearson r: {best_pearson['model']} ({best_pearson['pearson_r']:.3f})")
                report.append(f"  Best Spearman r: {best_spearman['model']} ({best_spearman['spearman_r']:.3f})")
                report.append(f"  Best Cohen's κ: {best_kappa['model']} ({best_kappa['cohen_kappa']:.3f})")
                report.append(f"  Lowest MAE: {lowest_mae['model']} ({lowest_mae['mae']:.3f})")
                
                # Average across all models
                avg_pearson = dim_data['pearson_r'].mean()
                avg_spearman = dim_data['spearman_r'].mean()
                avg_kappa = dim_data['cohen_kappa'].mean()
                avg_mae = dim_data['mae'].mean()
                
                report.append(f"  Average Pearson r: {avg_pearson:.3f}")
                report.append(f"  Average Spearman r: {avg_spearman:.3f}")
                report.append(f"  Average Cohen's κ: {avg_kappa:.3f}")
                report.append(f"  Average MAE: {avg_mae:.3f}")
        
        # Comparison task summary
        comparison_df = df[df['task_type'] == 'comparison']
        if not comparison_df.empty:
            report.append("\n\nCOMPARISON TASK CORRELATIONS:")
            report.append("-" * 30)
            
            for dimension in comparison_df['dimension'].unique():
                dim_data = comparison_df[comparison_df['dimension'] == dimension]
                report.append(f"\n{dimension.replace('_', ' ').title()}:")
                
                # Best performing models
                best_pearson = dim_data.loc[dim_data['pearson_r'].idxmax()]
                best_spearman = dim_data.loc[dim_data['spearman_r'].idxmax()]
                best_kappa = dim_data.loc[dim_data['cohen_kappa'].idxmax()]
                lowest_mae = dim_data.loc[dim_data['mae'].idxmin()]
                
                report.append(f"  Best Pearson r: {best_pearson['model']} ({best_pearson['pearson_r']:.3f})")
                report.append(f"  Best Spearman r: {best_spearman['model']} ({best_spearman['spearman_r']:.3f})")
                report.append(f"  Best Cohen's κ: {best_kappa['model']} ({best_kappa['cohen_kappa']:.3f})")
                report.append(f"  Lowest MAE: {lowest_mae['model']} ({lowest_mae['mae']:.3f})")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_detailed_results(self, output_path: str):
        """
        Save detailed results to CSV files.
        
        Args:
            output_path: Directory to save the results
        """
        df = self.create_summary_dataframe()
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        full_path = output_dir / "correlation_results_full.csv"
        df.to_csv(full_path, index=False)
        print(f"Full results saved to: {full_path}")
        
        # Save rating task results
        rating_df = df[df['task_type'] == 'rating']
        if not rating_df.empty:
            rating_path = output_dir / "correlation_results_rating.csv"
            rating_df.to_csv(rating_path, index=False)
            print(f"Rating results saved to: {rating_path}")
        
        # Save comparison task results
        comparison_df = df[df['task_type'] == 'comparison']
        if not comparison_df.empty:
            comparison_path = output_dir / "correlation_results_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Comparison results saved to: {comparison_path}")


def main():
    """Main function to run the correlation analysis."""
    parser = argparse.ArgumentParser(description="Process human-LLM correlation JSON files")
    parser.add_argument("--data_dir", 
                       default="../results/eval_llm_human_correlation",
                       help="Directory containing correlation JSON files")
    parser.add_argument("--output_dir", 
                       default="./results/dataset_visulization/analysis_model_human_corr",
                       help="Directory to save analysis results")
    parser.add_argument('--model_order', nargs='*', default=None,
                        help='Explicit model display order (use display names or raw names)')
    parser.add_argument("--plot", action="store_true", 
                       help="Generate heatmap plots")
    parser.add_argument("--triple", action="store_true",
                       help="Generate triple plot: rating Pearson, rating Spearman, comparison Spearman")
    parser.add_argument('--override', action='append', default=[],
                        help="Override a cell value: 'model,task,dimension,metric,value'.\n"
                             "Example: qwen3-1.7b,rating,neutrality,spearman_r,0.42")
    parser.add_argument("--individual", action="store_true", 
                       help="Generate individual heatmaps for each metric")
    parser.add_argument("--report", action="store_true", 
                       help="Generate summary report")
    parser.add_argument("--save_csv", action="store_true", 
                       help="Save results to CSV files")
    
    args = parser.parse_args()
    
    # Initialize processor
    # Normalize model order via mapping (accept raw or display names)
    model_order_display = None
    if args.model_order:
        mapping = HumanJudgeCorrelationProcessor(args.data_dir).model_name_mapping
        model_order_display = [mapping.get(m, m) for m in args.model_order]

    processor = HumanJudgeCorrelationProcessor(args.data_dir, model_order=model_order_display, overrides=args.override)
    
    # Find and load completed files
    print("Finding completed correlation files...")
    processor.find_completed_files()
    
    print("\nLoading correlation data...")
    processor.load_correlation_data()
    
    # Generate summary report
    if args.report:
        print("\nGenerating summary report...")
        report = processor.generate_summary_report()
        print(report)
        
        # Save report to file
        report_path = Path(args.output_dir) / "correlation_summary_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
    
    # Generate plots
    if args.plot:
        print("\nGenerating summary heatmap...")
        plot_dir = Path(args.output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary heatmap (both rating and comparison tasks)
        summary_plot_path = plot_dir / "correlation_summary.pdf"
        processor.plot_correlation_summary(save_path=str(summary_plot_path))

    # Generate triple plot
    if args.triple:
        print("\nGenerating triple plot (rating Pearson, rating Spearman, comparison Spearman)...")
        plot_dir = Path(args.output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        triple_plot_path = plot_dir / "rating_compare_triple_summary.pdf"
        processor.plot_rating_compare_triple(save_path=str(triple_plot_path))
    
    # Generate individual heatmaps
    if args.individual:
        print("\nGenerating individual heatmaps...")
        plot_dir = Path(args.output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        processor.plot_individual_heatmaps(save_dir=str(plot_dir))
    
    # Save CSV results
    if args.save_csv:
        print("\nSaving detailed results to CSV...")
        processor.save_detailed_results(args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
