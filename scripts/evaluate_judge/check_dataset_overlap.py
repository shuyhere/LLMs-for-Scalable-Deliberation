#!/usr/bin/env python3
"""
Script to check for overlap between train and test datasets
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    item['_line_number'] = line_num  # Add line number for reference
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                    continue
    return data


def extract_content_hash(item: Dict[str, Any]) -> str:
    """Extract a hashable representation of the item content"""
    # Use instruction and output as the key for comparison
    instruction = item.get('instruction', '')
    output = item.get('output', '')
    
    # Create a normalized string for comparison
    content = f"{instruction.strip()}|{output.strip()}"
    return content


def extract_instruction_hash(item: Dict[str, Any]) -> str:
    """Extract hash for instruction only"""
    instruction = item.get('instruction', '').strip()
    return instruction


def extract_output_hash(item: Dict[str, Any]) -> str:
    """Extract hash for output only"""
    output = item.get('output', '').strip()
    return output


def find_overlaps(train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find overlaps between train and test datasets"""
    
    # Create hash maps for different types of comparisons
    train_content_hashes = {}
    train_instruction_hashes = {}
    train_output_hashes = {}
    
    test_content_hashes = {}
    test_instruction_hashes = {}
    test_output_hashes = {}
    
    # Build hash maps for train data
    for item in train_data:
        content_hash = extract_content_hash(item)
        instruction_hash = extract_instruction_hash(item)
        output_hash = extract_output_hash(item)
        
        if content_hash in train_content_hashes:
            train_content_hashes[content_hash].append(item)
        else:
            train_content_hashes[content_hash] = [item]
            
        if instruction_hash in train_instruction_hashes:
            train_instruction_hashes[instruction_hash].append(item)
        else:
            train_instruction_hashes[instruction_hash] = [item]
            
        if output_hash in train_output_hashes:
            train_output_hashes[output_hash].append(item)
        else:
            train_output_hashes[output_hash] = [item]
    
    # Build hash maps for test data
    for item in test_data:
        content_hash = extract_content_hash(item)
        instruction_hash = extract_instruction_hash(item)
        output_hash = extract_output_hash(item)
        
        if content_hash in test_content_hashes:
            test_content_hashes[content_hash].append(item)
        else:
            test_content_hashes[content_hash] = [item]
            
        if instruction_hash in test_instruction_hashes:
            test_instruction_hashes[instruction_hash].append(item)
        else:
            test_instruction_hashes[instruction_hash] = [item]
            
        if output_hash in test_output_hashes:
            test_output_hashes[output_hash].append(item)
        else:
            test_output_hashes[output_hash] = [item]
    
    # Find overlaps
    results = {
        'exact_content_overlaps': [],
        'instruction_overlaps': [],
        'output_overlaps': [],
        'statistics': {
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'unique_train_content': len(train_content_hashes),
            'unique_test_content': len(test_content_hashes),
            'unique_train_instructions': len(train_instruction_hashes),
            'unique_test_instructions': len(test_instruction_hashes),
            'unique_train_outputs': len(train_output_hashes),
            'unique_test_outputs': len(test_output_hashes),
        }
    }
    
    # Check for exact content overlaps (instruction + output)
    for content_hash in test_content_hashes:
        if content_hash in train_content_hashes:
            train_items = train_content_hashes[content_hash]
            test_items = test_content_hashes[content_hash]
            
            results['exact_content_overlaps'].append({
                'content_hash': content_hash,
                'train_items': [{'line': item['_line_number'], 'instruction': item.get('instruction', '')[:100]} for item in train_items],
                'test_items': [{'line': item['_line_number'], 'instruction': item.get('instruction', '')[:100]} for item in test_items]
            })
    
    # Check for instruction overlaps
    for instruction_hash in test_instruction_hashes:
        if instruction_hash in train_instruction_hashes:
            train_items = train_instruction_hashes[instruction_hash]
            test_items = test_instruction_hashes[instruction_hash]
            
            results['instruction_overlaps'].append({
                'instruction_hash': instruction_hash,
                'instruction': instruction_hash[:200] + '...' if len(instruction_hash) > 200 else instruction_hash,
                'train_items': [{'line': item['_line_number'], 'output': item.get('output', '')[:100]} for item in train_items],
                'test_items': [{'line': item['_line_number'], 'output': item.get('output', '')[:100]} for item in test_items]
            })
    
    # Check for output overlaps
    for output_hash in test_output_hashes:
        if output_hash in train_output_hashes:
            train_items = train_output_hashes[output_hash]
            test_items = test_output_hashes[output_hash]
            
            results['output_overlaps'].append({
                'output_hash': output_hash,
                'output': output_hash[:200] + '...' if len(output_hash) > 200 else output_hash,
                'train_items': [{'line': item['_line_number'], 'instruction': item.get('instruction', '')[:100]} for item in train_items],
                'test_items': [{'line': item['_line_number'], 'instruction': item.get('instruction', '')[:100]} for item in test_items]
            })
    
    # Calculate overlap statistics
    results['statistics']['exact_content_overlaps'] = len(results['exact_content_overlaps'])
    results['statistics']['instruction_overlaps'] = len(results['instruction_overlaps'])
    results['statistics']['output_overlaps'] = len(results['output_overlaps'])
    
    return results


def print_overlap_report(results: Dict[str, Any], output_file: str = None):
    """Print detailed overlap report"""
    
    stats = results['statistics']
    
    print("=" * 80)
    print("DATASET OVERLAP ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"  Train samples: {stats['train_samples']}")
    print(f"  Test samples: {stats['test_samples']}")
    print(f"  Unique train content: {stats['unique_train_content']}")
    print(f"  Unique test content: {stats['unique_test_content']}")
    print(f"  Unique train instructions: {stats['unique_train_instructions']}")
    print(f"  Unique test instructions: {stats['unique_test_instructions']}")
    print(f"  Unique train outputs: {stats['unique_train_outputs']}")
    print(f"  Unique test outputs: {stats['unique_test_outputs']}")
    
    print(f"\n🔄 OVERLAP ANALYSIS:")
    print(f"  Exact content overlaps (instruction + output): {stats['exact_content_overlaps']}")
    print(f"  Instruction overlaps: {stats['instruction_overlaps']}")
    print(f"  Output overlaps: {stats['output_overlaps']}")
    
    # Calculate overlap percentages
    if stats['test_samples'] > 0:
        exact_overlap_pct = (stats['exact_content_overlaps'] / stats['test_samples']) * 100
        instruction_overlap_pct = (stats['instruction_overlaps'] / stats['test_samples']) * 100
        output_overlap_pct = (stats['output_overlaps'] / stats['test_samples']) * 100
        
        print(f"\n📈 OVERLAP PERCENTAGES (of test samples):")
        print(f"  Exact content overlaps: {exact_overlap_pct:.2f}%")
        print(f"  Instruction overlaps: {instruction_overlap_pct:.2f}%")
        print(f"  Output overlaps: {output_overlap_pct:.2f}%")
    
    # Show detailed overlaps if any
    if results['exact_content_overlaps']:
        print(f"\n🚨 EXACT CONTENT OVERLAPS ({len(results['exact_content_overlaps'])} found):")
        for i, overlap in enumerate(results['exact_content_overlaps'][:5], 1):  # Show first 5
            print(f"\n  {i}. Content hash: {overlap['content_hash'][:50]}...")
            print(f"     Train items: {[item['line'] for item in overlap['train_items']]}")
            print(f"     Test items: {[item['line'] for item in overlap['test_items']]}")
        
        if len(results['exact_content_overlaps']) > 5:
            print(f"     ... and {len(results['exact_content_overlaps']) - 5} more")
    
    if results['instruction_overlaps']:
        print(f"\n⚠️  INSTRUCTION OVERLAPS ({len(results['instruction_overlaps'])} found):")
        for i, overlap in enumerate(results['instruction_overlaps'][:3], 1):  # Show first 3
            print(f"\n  {i}. Instruction: {overlap['instruction']}")
            print(f"     Train items: {[item['line'] for item in overlap['train_items']]}")
            print(f"     Test items: {[item['line'] for item in overlap['test_items']]}")
        
        if len(results['instruction_overlaps']) > 3:
            print(f"     ... and {len(results['instruction_overlaps']) - 3} more")
    
    if results['output_overlaps']:
        print(f"\n⚠️  OUTPUT OVERLAPS ({len(results['output_overlaps'])} found):")
        for i, overlap in enumerate(results['output_overlaps'][:3], 1):  # Show first 3
            print(f"\n  {i}. Output: {overlap['output']}")
            print(f"     Train items: {[item['line'] for item in overlap['train_items']]}")
            print(f"     Test items: {[item['line'] for item in overlap['test_items']]}")
        
        if len(results['output_overlaps']) > 3:
            print(f"     ... and {len(results['output_overlaps']) - 3} more")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    if stats['exact_content_overlaps'] == 0:
        print("  ✅ No exact content overlaps found - good data separation!")
    else:
        print(f"  🚨 Found {stats['exact_content_overlaps']} exact content overlaps - data leakage detected!")
    
    if stats['instruction_overlaps'] > stats['exact_content_overlaps']:
        print(f"  ⚠️  Found {stats['instruction_overlaps']} instruction overlaps - potential issues")
    
    if stats['output_overlaps'] > stats['exact_content_overlaps']:
        print(f"  ⚠️  Found {stats['output_overlaps']} output overlaps - potential issues")
    
    # Save detailed results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Check for overlaps between train and test datasets")
    
    parser.add_argument("--train_file", type=str, 
                       default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_annotation_format/alpace/comparison_alpaca/train.jsonl",
                       help="Path to training data JSONL file")
    
    parser.add_argument("--test_file", type=str,
                       default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_annotation_format/alpace/comparison_alpaca/test.jsonl",
                       help="Path to test data JSONL file")
    
    parser.add_argument("--output_file", type=str,
                       help="Optional output file to save detailed results")
    
    args = parser.parse_args()
    
    print("🔍 Checking for dataset overlaps...")
    print(f"Train file: {args.train_file}")
    print(f"Test file: {args.test_file}")
    
    # Check if files exist
    if not Path(args.train_file).exists():
        print(f"❌ Train file not found: {args.train_file}")
        return
    
    if not Path(args.test_file).exists():
        print(f"❌ Test file not found: {args.test_file}")
        return
    
    # Load datasets
    print("\n🔄 Loading datasets...")
    train_data = load_jsonl(args.train_file)
    test_data = load_jsonl(args.test_file)
    
    print(f"✅ Loaded {len(train_data)} train samples")
    print(f"✅ Loaded {len(test_data)} test samples")
    
    # Find overlaps
    print("\n🔄 Analyzing overlaps...")
    results = find_overlaps(train_data, test_data)
    
    # Print report
    print_overlap_report(results, args.output_file)


if __name__ == "__main__":
    main()
