#!/usr/bin/env python3
"""
Script to convert absolute paths to relative paths in visualization scripts
"""

import os
import re
from pathlib import Path

def convert_paths_in_file(file_path):
    """Convert absolute paths to relative paths in a single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Convert PROJECT_ROOT definitions
    content = re.sub(
        r"PROJECT_ROOT\s*=\s*Path\('\/ibex\/project\/c2328\/LLMs-Scalable-Deliberation'\)",
        "PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent",
        content
    )
    
    # Convert absolute paths to relative paths
    content = re.sub(
        r"'\/ibex\/project\/c2328\/LLMs-Scalable-Deliberation\/",
        "'../../../../",
        content
    )
    
    content = re.sub(
        r'"\/ibex\/project\/c2328\/LLMs-Scalable-Deliberation\/',
        '"../../../../',
        content
    )
    
    # Convert f-string paths
    content = re.sub(
        r"f'\/ibex\/project\/c2328\/LLMs-Scalable-Deliberation\/",
        "f'../../../../",
        content
    )
    
    content = re.sub(
        r'f"\/ibex\/project\/c2328\/LLMs-Scalable-Deliberation\/',
        'f"../../../../',
        content
    )
    
    # If content changed, write it back
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {file_path}")
        return True
    else:
        print(f"No changes needed: {file_path}")
        return False

def main():
    """Convert paths in all visualization Python files"""
    script_dir = Path(__file__).parent
    visualization_dir = script_dir / "data_visualization"
    
    updated_files = []
    
    # Find all Python files in visualization directory
    for py_file in visualization_dir.rglob("*.py"):
        if convert_paths_in_file(py_file):
            updated_files.append(py_file)
    
    print(f"\nUpdated {len(updated_files)} files:")
    for file in updated_files:
        print(f"  - {file.relative_to(script_dir)}")

if __name__ == "__main__":
    main()
