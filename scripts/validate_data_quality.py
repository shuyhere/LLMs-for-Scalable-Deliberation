#!/usr/bin/env python3
"""
Data validation script to check the quality of comparison dataset.
Validates similarity between user opinion and summaries, and correlation with scores.
"""

import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the comparison dataset."""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_texts(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract user comment and two summaries from example."""
    prompt = example['prompt']
    
    # Extract user comment
    user_comment_match = re.search(r'User Comment: (.+?)(?=\n\nSummary A:)', prompt, re.DOTALL)
    user_comment = user_comment_match.group(1).strip() if user_comment_match else ""
    
    # Extract Summary A
    summary_a_match = re.search(r'Summary A: (.+?)(?=Summary B:)', prompt, re.DOTALL)
    summary_a = summary_a_match.group(1).strip() if summary_a_match else ""
    
    # Extract Summary B
    summary_b_match = re.search(r'Summary B: (.+?)(?=\n\nPlease provide)', prompt, re.DOTALL)
    summary_b = summary_b_match.group(1).strip() if summary_b_match else ""
    
    return user_comment, summary_a, summary_b

def calculate_tfidf_similarity(texts: List[str]) -> np.ndarray:
    """Calculate TF-IDF similarity between texts."""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except Exception as e:
        print(f"Error in TF-IDF calculation: {e}")
        return np.zeros((len(texts), len(texts)))

def calculate_sentence_transformer_similarity(texts: List[str], device: str = 'auto') -> np.ndarray:
    """Calculate similarity using sentence transformers with GPU support and progress bar."""
    try:
        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {device}")
        
        # Use a lightweight model for faster processing
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Encode texts with progress bar
        embeddings = model.encode(
            texts, 
            batch_size=32,  # Larger batch size for GPU
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    except Exception as e:
        print(f"Error in sentence transformer calculation: {e}")
        return np.zeros((len(texts), len(texts)))

def calculate_detailed_cosine_similarity(user_comments: List[str], summary_as: List[str], 
                                       summary_bs: List[str], all_scores: List[List[float]], 
                                       device: str = 'auto') -> Dict[str, Any]:
    """Calculate detailed cosine similarity between user opinion and each summary sentence."""
    try:
        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Calculating detailed cosine similarities using device: {device}")
        
        # Use sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        results = {
            'examples': [],
            'summary': {},
            'correlation_analysis': {}
        }
        
        # Process each example
        for i in tqdm(range(len(user_comments)), desc="Processing examples"):
            user_comment = user_comments[i]
            summary_a = summary_as[i]
            summary_b = summary_bs[i]
            scores = all_scores[i]
            
            # Split summaries into sentences
            summary_a_sentences = [s.strip() for s in summary_a.split('.') if s.strip()]
            summary_b_sentences = [s.strip() for s in summary_b.split('.') if s.strip()]
            
            # Calculate similarities
            similarities_a = []
            similarities_b = []
            
            if summary_a_sentences and summary_b_sentences:
                # Encode user comment
                user_embedding = model.encode([user_comment], convert_to_numpy=True)[0]
                
                # Encode summary A sentences
                if summary_a_sentences:
                    summary_a_embeddings = model.encode(summary_a_sentences, convert_to_numpy=True)
                    similarities_a = [cosine_similarity([user_embedding], [emb])[0][0] for emb in summary_a_embeddings]
                
                # Encode summary B sentences
                if summary_b_sentences:
                    summary_b_embeddings = model.encode(summary_b_sentences, convert_to_numpy=True)
                    similarities_b = [cosine_similarity([user_embedding], [emb])[0][0] for emb in summary_b_embeddings]
            
            # Calculate average similarities
            avg_sim_a = np.mean(similarities_a) if similarities_a else 0.0
            avg_sim_b = np.mean(similarities_b) if similarities_b else 0.0
            
            # Determine which summary was preferred (scores closer to 1 = A better, closer to 2 = B better)
            avg_score = np.mean(scores)
            preferred_summary = 'A' if avg_score < 1.5 else 'B'
            
            # Store results for this example
            example_result = {
                'example_idx': i,
                'user_comment': user_comment,
                'summary_a_sentences': summary_a_sentences,
                'summary_b_sentences': summary_b_sentences,
                'sentence_similarities_a': similarities_a,
                'sentence_similarities_b': similarities_b,
                'avg_similarity_a': avg_sim_a,
                'avg_similarity_b': avg_sim_b,
                'comparison_scores': scores,
                'avg_score': avg_score,
                'preferred_summary': preferred_summary,
                'higher_similarity_summary': 'A' if avg_sim_a > avg_sim_b else 'B',
                'similarity_matches_preference': (preferred_summary == ('A' if avg_sim_a > avg_sim_b else 'B'))
            }
            
            results['examples'].append(example_result)
        
        # Calculate summary statistics
        total_examples = len(results['examples'])
        similarity_matches = sum(1 for ex in results['examples'] if ex['similarity_matches_preference'])
        
        # Calculate correlations between similarity differences and preference scores
        similarity_diffs = [ex['avg_similarity_a'] - ex['avg_similarity_b'] for ex in results['examples']]
        avg_scores = [ex['avg_score'] for ex in results['examples']]
        
        correlation = np.corrcoef(similarity_diffs, avg_scores)[0, 1] if len(similarity_diffs) > 1 else 0.0
        
        results['summary'] = {
            'total_examples': total_examples,
            'similarity_matches_preference_count': similarity_matches,
            'similarity_matches_preference_rate': similarity_matches / total_examples if total_examples > 0 else 0.0,
            'avg_similarity_a': np.mean([ex['avg_similarity_a'] for ex in results['examples']]),
            'avg_similarity_b': np.mean([ex['avg_similarity_b'] for ex in results['examples']]),
            'similarity_preference_correlation': correlation
        }
        
        # Analyze by dimension
        dimension_names = ['perspective', 'informativeness', 'neutrality', 'policy']
        results['dimension_analysis'] = {}
        
        for dim_idx, dim_name in enumerate(dimension_names):
            print(f"\n🔍 Analyzing dimension: {dim_name.title()}")
            
            # Get scores for this dimension
            dim_scores = [ex['comparison_scores'][dim_idx] for ex in results['examples']]
            
            # Calculate correlation for this dimension
            dim_correlation = np.corrcoef(similarity_diffs, dim_scores)[0, 1] if len(similarity_diffs) > 1 else 0.0
            
            # Analyze matches for this dimension
            dim_matches = []
            for ex in results['examples']:
                dim_score = ex['comparison_scores'][dim_idx]
                preferred_for_dim = 'A' if dim_score < 1.5 else 'B'
                higher_sim = 'A' if ex['avg_similarity_a'] > ex['avg_similarity_b'] else 'B'
                matches = (preferred_for_dim == higher_sim)
                dim_matches.append(matches)
            
            dim_match_rate = sum(dim_matches) / len(dim_matches) if dim_matches else 0.0
            
            # Separate examples by preference for this dimension
            a_preferred_examples = [ex for ex in results['examples'] if ex['comparison_scores'][dim_idx] < 1.5]
            b_preferred_examples = [ex for ex in results['examples'] if ex['comparison_scores'][dim_idx] >= 1.5]
            
            # Calculate average similarities for each preference group
            avg_sim_a_preferred = np.mean([ex['avg_similarity_a'] - ex['avg_similarity_b'] for ex in a_preferred_examples]) if a_preferred_examples else 0.0
            avg_sim_b_preferred = np.mean([ex['avg_similarity_a'] - ex['avg_similarity_b'] for ex in b_preferred_examples]) if b_preferred_examples else 0.0
            
            # Calculate additional statistics for this dimension
            avg_similarity_a = np.mean([ex['avg_similarity_a'] for ex in results['examples']])
            avg_similarity_b = np.mean([ex['avg_similarity_b'] for ex in results['examples']])
            
            results['dimension_analysis'][dim_name] = {
                'total_examples': len(results['examples']),
                'a_preferred_count': len(a_preferred_examples),
                'b_preferred_count': len(b_preferred_examples),
                'correlation_with_similarity_diff': dim_correlation,
                'avg_score': np.mean(dim_scores),
                'similarity_match_rate': dim_match_rate,
                'avg_similarity_diff_a_preferred': avg_sim_a_preferred,
                'avg_similarity_diff_b_preferred': avg_sim_b_preferred,
                'avg_similarity_a': avg_similarity_a,
                'avg_similarity_b': avg_similarity_b,
                'similarity_matches_preference_count': sum(dim_matches),
                'examples': []
            }
            
            # Store examples for this dimension
            for i, ex in enumerate(results['examples']):
                dim_score = ex['comparison_scores'][dim_idx]
                preferred_for_dim = 'A' if dim_score < 1.5 else 'B'
                higher_sim = 'A' if ex['avg_similarity_a'] > ex['avg_similarity_b'] else 'B'
                matches = (preferred_for_dim == higher_sim)
                
                results['dimension_analysis'][dim_name]['examples'].append({
                    'example_idx': i,
                    'dimension_score': dim_score,
                    'preferred_summary': preferred_for_dim,
                    'higher_similarity_summary': higher_sim,
                    'similarity_matches_preference': matches,
                    'avg_similarity_a': ex['avg_similarity_a'],
                    'avg_similarity_b': ex['avg_similarity_b'],
                    'similarity_diff': ex['avg_similarity_a'] - ex['avg_similarity_b']
                })
                
                # Add dimension-specific fields to the main example
                if i < len(results['examples']):
                    results['examples'][i][f'{dim_name}_preferred_summary'] = preferred_for_dim
                    results['examples'][i][f'{dim_name}_higher_similarity_summary'] = higher_sim
                    results['examples'][i][f'{dim_name}_similarity_matches_preference'] = matches
            
            # General correlation analysis (existing)
            results['correlation_analysis'][dim_name] = {
                'correlation_with_similarity_diff': dim_correlation,
                'avg_score': np.mean(dim_scores)
            }
        
        # Add overall summary across all dimensions
        if results['dimension_analysis']:
            total_matches = 0
            total_examples = 0
            avg_correlations = []
            
            for dim_name, dim_data in results['dimension_analysis'].items():
                total_matches += dim_data.get('similarity_matches_preference_count', 0)
                total_examples += dim_data.get('total_examples', 0)
                avg_correlations.append(dim_data.get('correlation_with_similarity_diff', 0))
            
            overall_match_rate = total_matches / total_examples if total_examples > 0 else 0.0
            avg_correlation = np.mean(avg_correlations) if avg_correlations else 0.0
            
            results['overall_dimension_summary'] = {
                'overall_similarity_match_rate': overall_match_rate,
                'overall_similarity_matches_preference_count': total_matches,
                'total_examples_analyzed': total_examples,
                'average_correlation_across_dimensions': avg_correlation,
                'best_performing_dimension': max(results['dimension_analysis'].keys(), key=lambda x: results['dimension_analysis'][x].get('similarity_match_rate', 0)),
                'worst_performing_dimension': min(results['dimension_analysis'].keys(), key=lambda x: results['dimension_analysis'][x].get('similarity_match_rate', 0))
            }
        
        return results
        
    except Exception as e:
        print(f"Error in detailed cosine similarity calculation: {e}")
        return {'examples': [], 'summary': {}, 'correlation_analysis': {}}

def analyze_similarity_correlation(data: List[Dict[str, Any]], device: str = "auto", batch_size: int = 50) -> Dict[str, Any]:
    """Analyze correlation between similarity and scores."""
    results = {
        'tfidf_correlations': [],
        'sentence_transformer_correlations': [],
        'cosine_similarity_analysis': {},
        'examples': [],
        'summary_stats': {
            'total_examples': len(data),
            'avg_user_comment_length': 0,
            'avg_summary_a_length': 0,
            'avg_summary_b_length': 0,
        }
    }
    
    user_comments = []
    summary_as = []
    summary_bs = []
    all_scores = []
    
    print("Processing examples...")
    for i, example in enumerate(data):
        if i % 100 == 0:
            print(f"Processing example {i}/{len(data)}")
        
        try:
            user_comment, summary_a, summary_b = extract_texts(example)
            scores = example['metadata']['comparison_scores']
            
            # Skip if texts are too short or empty
            if len(user_comment) < 10 or len(summary_a) < 20 or len(summary_b) < 20:
                continue
            
            user_comments.append(user_comment)
            summary_as.append(summary_a)
            summary_bs.append(summary_b)
            all_scores.append(scores)
            
            results['examples'].append({
                'user_comment': user_comment[:100] + "..." if len(user_comment) > 100 else user_comment,
                'summary_a': summary_a[:100] + "..." if len(summary_a) > 100 else summary_a,
                'summary_b': summary_b[:100] + "..." if len(summary_b) > 100 else summary_b,
                'scores': scores
            })
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    print(f"Successfully processed {len(user_comments)} examples")
    
    if len(user_comments) == 0:
        print("No valid examples found!")
        return results
    
    # Calculate summary statistics
    results['summary_stats']['avg_user_comment_length'] = np.mean([len(c) for c in user_comments])
    results['summary_stats']['avg_summary_a_length'] = np.mean([len(s) for s in summary_as])
    results['summary_stats']['avg_summary_b_length'] = np.mean([len(s) for s in summary_bs])
    
    # Calculate similarities
    print("Calculating TF-IDF similarities...")
    tfidf_similarities = []
    for i in range(len(user_comments)):
        texts = [user_comments[i], summary_as[i], summary_bs[i]]
        sim_matrix = calculate_tfidf_similarity(texts)
        tfidf_similarities.append({
            'user_to_a': sim_matrix[0, 1],
            'user_to_b': sim_matrix[0, 2],
            'a_to_b': sim_matrix[1, 2]
        })
    
    print("Calculating Sentence Transformer similarities...")
    
    # Batch process for better GPU utilization
    sentence_transformer_similarities = []
    
    for i in tqdm(range(0, len(user_comments), batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, len(user_comments))
        batch_texts = []
        
        # Prepare batch
        for j in range(i, batch_end):
            texts = [user_comments[j], summary_as[j], summary_bs[j]]
            batch_texts.extend(texts)
        
        # Calculate similarities for batch
        if batch_texts:
            sim_matrix = calculate_sentence_transformer_similarity(batch_texts, device=device)
            
            # Extract individual similarities
            for j in range(i, batch_end):
                idx_in_batch = (j - i) * 3
                sentence_transformer_similarities.append({
                    'user_to_a': sim_matrix[idx_in_batch, idx_in_batch + 1],
                    'user_to_b': sim_matrix[idx_in_batch, idx_in_batch + 2],
                    'a_to_b': sim_matrix[idx_in_batch + 1, idx_in_batch + 2]
                })
    
    # Calculate correlations
    print("Calculating correlations...")
    for dim_idx, dim_name in enumerate(['perspective', 'informativeness', 'neutrality', 'policy']):
        scores_dim = [scores[dim_idx] for scores in all_scores]
        
        # TF-IDF correlations
        user_to_a_sim = [sim['user_to_a'] for sim in tfidf_similarities]
        user_to_b_sim = [sim['user_to_b'] for sim in tfidf_similarities]
        
        # Correlation: higher similarity should lead to higher score (1 = A better, 2 = B better)
        # So we expect negative correlation between user_to_a_sim and scores (higher similarity -> score closer to 1)
        # And positive correlation between user_to_b_sim and scores (higher similarity -> score closer to 2)
        
        corr_a_tfidf = np.corrcoef(user_to_a_sim, scores_dim)[0, 1]
        corr_b_tfidf = np.corrcoef(user_to_b_sim, scores_dim)[0, 1]
        
        results['tfidf_correlations'].append({
            'dimension': dim_name,
            'correlation_user_to_a': corr_a_tfidf,
            'correlation_user_to_b': corr_b_tfidf,
            'expected_a_correlation': 'negative (higher similarity -> A preferred)',
            'expected_b_correlation': 'positive (higher similarity -> B preferred)'
        })
        
        # Sentence Transformer correlations
        user_to_a_sim_st = [sim['user_to_a'] for sim in sentence_transformer_similarities]
        user_to_b_sim_st = [sim['user_to_b'] for sim in sentence_transformer_similarities]
        
        corr_a_st = np.corrcoef(user_to_a_sim_st, scores_dim)[0, 1]
        corr_b_st = np.corrcoef(user_to_b_sim_st, scores_dim)[0, 1]
        
        results['sentence_transformer_correlations'].append({
            'dimension': dim_name,
            'correlation_user_to_a': corr_a_st,
            'correlation_user_to_b': corr_b_st,
            'expected_a_correlation': 'negative (higher similarity -> A preferred)',
            'expected_b_correlation': 'positive (higher similarity -> B preferred)'
        })
    
    # Calculate detailed cosine similarity analysis
    print("\n🔍 Calculating detailed cosine similarity analysis...")
    cosine_analysis = calculate_detailed_cosine_similarity(user_comments, summary_as, summary_bs, all_scores, device)
    results['cosine_similarity_analysis'] = cosine_analysis
    
    return results

def print_results(results: Dict[str, Any]):
    """Print validation results in a readable format."""
    print("\n" + "="*80)
    print("DATA VALIDATION RESULTS")
    print("="*80)
    
    # Summary statistics
    stats = results['summary_stats']
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total examples processed: {stats['total_examples']}")
    print(f"   Average user comment length: {stats['avg_user_comment_length']:.1f} chars")
    print(f"   Average Summary A length: {stats['avg_summary_a_length']:.1f} chars")
    print(f"   Average Summary B length: {stats['avg_summary_b_length']:.1f} chars")
    
    # TF-IDF correlations
    print(f"\n🔍 TF-IDF SIMILARITY CORRELATIONS:")
    print(f"{'Dimension':<15} {'User→A Corr':<12} {'User→B Corr':<12} {'A Expected':<15} {'B Expected':<15}")
    print("-" * 80)
    
    for corr in results['tfidf_correlations']:
        print(f"{corr['dimension']:<15} {corr['correlation_user_to_a']:<12.4f} {corr['correlation_user_to_b']:<12.4f} {'Negative':<15} {'Positive':<15}")
    
    # Sentence Transformer correlations
    print(f"\n🧠 SENTENCE TRANSFORMER SIMILARITY CORRELATIONS:")
    print(f"{'Dimension':<15} {'User→A Corr':<12} {'User→B Corr':<12} {'A Expected':<15} {'B Expected':<15}")
    print("-" * 80)
    
    for corr in results['sentence_transformer_correlations']:
        print(f"{corr['dimension']:<15} {corr['correlation_user_to_a']:<12.4f} {corr['correlation_user_to_b']:<12.4f} {'Negative':<15} {'Positive':<15}")
    
    # Interpretation
    print(f"\n📝 INTERPRETATION:")
    print("   - User→A correlation should be NEGATIVE: higher similarity → A preferred (score closer to 1)")
    print("   - User→B correlation should be POSITIVE: higher similarity → B preferred (score closer to 2)")
    print("   - Strong correlations (|r| > 0.3) suggest the data makes sense")
    print("   - Weak correlations (|r| < 0.1) suggest the scoring may not be based on similarity")
    
    # Detailed Cosine Similarity Analysis
    cosine_analysis = results.get('cosine_similarity_analysis', {})
    if cosine_analysis:
        print(f"\n🎯 DETAILED COSINE SIMILARITY ANALYSIS:")
        summary = cosine_analysis.get('summary', {})
        print(f"   Total examples analyzed: {summary.get('total_examples', 0)}")
        print(f"   Similarity matches preference: {summary.get('similarity_matches_preference_count', 0)}/{summary.get('total_examples', 0)} ({summary.get('similarity_matches_preference_rate', 0)*100:.1f}%)")
        print(f"   Average similarity to Summary A: {summary.get('avg_similarity_a', 0):.3f}")
        print(f"   Average similarity to Summary B: {summary.get('avg_similarity_b', 0):.3f}")
        print(f"   Correlation between similarity diff and preference: {summary.get('similarity_preference_correlation', 0):.3f}")
        
        print(f"\n   📈 BY DIMENSION:")
        correlation_analysis = cosine_analysis.get('correlation_analysis', {})
        dimension_analysis = cosine_analysis.get('dimension_analysis', {})
        
        for dim_name in ['perspective', 'informativeness', 'neutrality', 'policy']:
            dim_data = correlation_analysis.get(dim_name, {})
            dim_analysis = dimension_analysis.get(dim_name, {})
            
            print(f"\n     🎯 {dim_name.upper()}:")
            print(f"       Correlation with similarity diff: {dim_data.get('correlation_with_similarity_diff', 0):.3f}")
            print(f"       Average score: {dim_data.get('avg_score', 0):.3f}")
            
            if dim_analysis:
                print(f"       Total examples: {dim_analysis.get('total_examples', 0)}")
                print(f"       Similarity matches preference: {dim_analysis.get('similarity_matches_preference_count', 0)}/{dim_analysis.get('total_examples', 0)} ({dim_analysis.get('similarity_match_rate', 0)*100:.1f}%)")
                print(f"       Average similarity to Summary A: {dim_analysis.get('avg_similarity_a', 0):.3f}")
                print(f"       Average similarity to Summary B: {dim_analysis.get('avg_similarity_b', 0):.3f}")
                print(f"       A preferred: {dim_analysis.get('a_preferred_count', 0)} examples")
                print(f"       B preferred: {dim_analysis.get('b_preferred_count', 0)} examples")
                print(f"       Avg similarity diff (A preferred): {dim_analysis.get('avg_similarity_diff_a_preferred', 0):.3f}")
                print(f"       Avg similarity diff (B preferred): {dim_analysis.get('avg_similarity_diff_b_preferred', 0):.3f}")
                
                # Show sample examples for this dimension
                examples = dim_analysis.get('examples', [])
                if examples:
                    print(f"       📝 Sample examples for {dim_name}:")
                    for i, ex in enumerate(examples[:2]):  # Show first 2 examples per dimension
                        print(f"         Example {ex['example_idx']+1}:")
                        print(f"           Dimension score: {ex['dimension_score']:.1f}")
                        print(f"           Preferred summary: {ex['preferred_summary']}")
                        print(f"           Higher similarity summary: {ex['higher_similarity_summary']}")
                        print(f"           Similarity matches preference: {'✅' if ex['similarity_matches_preference'] else '❌'}")
                        print(f"           Avg similarity A: {ex['avg_similarity_a']:.3f}")
                        print(f"           Avg similarity B: {ex['avg_similarity_b']:.3f}")
                        print(f"           Similarity diff: {ex['similarity_diff']:.3f}")
        
        # Overall summary across all dimensions
        if dimension_analysis:
            print(f"\n   📊 OVERALL SUMMARY ACROSS ALL DIMENSIONS:")
            total_matches = 0
            total_examples = 0
            avg_correlations = []
            
            for dim_name, dim_data in dimension_analysis.items():
                total_matches += dim_data.get('similarity_matches_preference_count', 0)
                total_examples += dim_data.get('total_examples', 0)
                avg_correlations.append(dim_data.get('correlation_with_similarity_diff', 0))
            
            overall_match_rate = total_matches / total_examples if total_examples > 0 else 0.0
            avg_correlation = np.mean(avg_correlations) if avg_correlations else 0.0
            
            print(f"       Overall similarity match rate: {total_matches}/{total_examples} ({overall_match_rate*100:.1f}%)")
            print(f"       Average correlation across dimensions: {avg_correlation:.3f}")
            print(f"       Best performing dimension: {max(dimension_analysis.keys(), key=lambda x: dimension_analysis[x].get('similarity_match_rate', 0))}")
            print(f"       Worst performing dimension: {min(dimension_analysis.keys(), key=lambda x: dimension_analysis[x].get('similarity_match_rate', 0))}")
        
        # Show sample examples with cosine similarities
        examples = cosine_analysis.get('examples', [])
        if examples:
            print(f"\n   📝 SAMPLE EXAMPLES WITH COSINE SIMILARITIES (first 3):")
            for i, example in enumerate(examples[:3]):
                print(f"\n     Example {i+1}:")
                print(f"       User comment: {example['user_comment'][:80]}...")
                print(f"       Avg similarity to A: {example['avg_similarity_a']:.3f}")
                print(f"       Avg similarity to B: {example['avg_similarity_b']:.3f}")
                print(f"       Overall preferred summary: {example['preferred_summary']}")
                print(f"       Overall higher similarity summary: {example['higher_similarity_summary']}")
                print(f"       Overall similarity matches preference: {'✅' if example['similarity_matches_preference'] else '❌'}")
                print(f"       Comparison scores: {example['comparison_scores']}")
                
                # Show dimension-specific results
                print(f"       📊 By Dimension:")
                for dim_name in ['perspective', 'informativeness', 'neutrality', 'policy']:
                    pref_key = f'{dim_name}_preferred_summary'
                    sim_key = f'{dim_name}_higher_similarity_summary'
                    match_key = f'{dim_name}_similarity_matches_preference'
                    
                    if pref_key in example:
                        print(f"         {dim_name.title()}:")
                        print(f"           Preferred: {example[pref_key]}")
                        print(f"           Higher similarity: {example[sim_key]}")
                        print(f"           Matches: {'✅' if example[match_key] else '❌'}")
    
    # Sample examples
    print(f"\n📋 SAMPLE EXAMPLES:")
    for i, example in enumerate(results['examples'][:3]):
        print(f"\nExample {i+1}:")
        print(f"   User: {example['user_comment']}")
        print(f"   Summary A: {example['summary_a']}")
        print(f"   Summary B: {example['summary_b']}")
        print(f"   Scores: {example['scores']}")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Remove examples from saved results to keep file size manageable
    results_to_save = results.copy()
    results_to_save['examples'] = results_to_save['examples'][:10]  # Keep only first 10 examples
    
    # Convert numpy types to Python native types
    results_to_save = convert_numpy_types(results_to_save)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")

def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate comparison dataset quality")
    parser.add_argument("--dataset_path", type=str, 
                       default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_dataset/comparison_sft_dataset.jsonl",
                       help="Path to the comparison dataset")
    parser.add_argument("--output_path", type=str,
                       default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/data_validation_results.json",
                       help="Path to save validation results")
    parser.add_argument("--max_examples", type=int, default=1000,
                       help="Maximum number of examples to process (for faster testing)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for sentence transformers (auto, cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=50,
                       help="Batch size for processing examples")
    
    args = parser.parse_args()
    
    print("🔍 Starting data validation...")
    print(f"Dataset: {args.dataset_path}")
    print(f"Max examples: {args.max_examples}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    data = load_dataset(args.dataset_path)
    print(f"Loaded {len(data)} examples")
    
    # Limit examples for faster processing
    if len(data) > args.max_examples:
        print(f"Limiting to first {args.max_examples} examples for faster processing")
        data = data[:args.max_examples]
    
    # Analyze similarity correlations
    print("\n🔬 Analyzing similarity correlations...")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ GPU not available, using CPU")
    
    results = analyze_similarity_correlation(data, device=args.device, batch_size=args.batch_size)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, args.output_path)
    
    print("\n✅ Data validation completed!")

if __name__ == "__main__":
    main()
