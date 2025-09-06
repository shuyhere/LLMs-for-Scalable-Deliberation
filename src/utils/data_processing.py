#!/usr/bin/env python3
"""
Data processing utilities for human annotation data
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class HumanAnnotationDataProcessor:
    """
    Data processor for human annotation data
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the data processor
        
        Args:
            debug: Enable debug mode for detailed logging
        """
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
    
    def load_human_annotations(self, annotation_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Load human annotation data including annotator's answers
        
        Args:
            annotation_path: Path to annotation directory
            
        Returns:
            Tuple of (rating_annotations, comparison_annotations)
        """
        rating_annotations = []
        comparison_annotations = []
        
        if self.debug:
            logger.info(f"Loading annotations from: {annotation_path}")
        
        user_dirs = [d for d in annotation_path.iterdir() if d.is_dir()]
        if self.debug:
            logger.info(f"Found {len(user_dirs)} user directories")
        
        for user_dir in user_dirs:
            jsonl_file = user_dir / "annotated_instances.jsonl"
            assign_file = user_dir / "assigned_user_data.json"
            
            if not (jsonl_file.exists() and assign_file.exists()):
                if self.debug:
                    logger.warning(f"Skipping {user_dir.name}: missing required files")
                continue
            
            if self.debug:
                logger.debug(f"Processing user directory: {user_dir.name}")
            
            try:
                # Load assigned data for metadata
                with open(assign_file, 'r', encoding='utf-8') as f:
                    assigned_data = json.load(f)
                
                # First pass: collect annotator's answers from question annotations
                annotator_answers = {}
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'question' in data['id'] and 'rating' not in data['id'] and 'comparison' not in data['id']:
                                # Extract triplet ID (e.g., "triplet_344" from "triplet_344_question")
                                triplet_id = data['id'].replace('_question', '')
                                # Get annotator's answer
                                if 'label_annotations' in data and 'answer' in data['label_annotations']:
                                    answer_text = data['label_annotations']['answer'].get('text_box', '')
                                    annotator_answers[triplet_id] = answer_text
                        except Exception as e:
                            if self.debug:
                                logger.warning(f"Error parsing question line: {e}")
                            continue
                
                if self.debug:
                    logger.debug(f"Found annotator answers")
                
                # Second pass: load rating and comparison annotations with annotator answers
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            
                            # Add metadata
                            if data['id'] in assigned_data:
                                data['metadata'] = assigned_data[data['id']]
                            
                            # Add user_id for data source tracking
                            data['user_id'] = user_dir.name
                            
                            # Extract triplet ID
                            triplet_id = data['id'].rsplit('_', 1)[0] if '_' in data['id'] else data['id']
                            
                            # Add annotator's answer to data
                            data['annotator_answer'] = annotator_answers.get(triplet_id, '')
                            
                            # Categorize by type
                            if 'rating' in data['id']:
                                rating_annotations.append(data)
                            elif 'comparison' in data['id']:
                                comparison_annotations.append(data)
                        except Exception as e:
                            if self.debug:
                                logger.warning(f"Error parsing annotation line: {e}")
                            continue
                            
            except Exception as e:
                if self.debug:
                    logger.error(f"Error processing user directory {user_dir.name}: {e}")
                continue
        
        if self.debug:
            logger.info(f"Loaded rating and comparison annotations")
        
        return rating_annotations, comparison_annotations
    
    def extract_human_ratings(self, annotation: Dict) -> Dict[str, int]:
        """Extract human rating scores from annotation"""
        ratings = {}
        
        if 'label_annotations' not in annotation:
            return ratings
        
        # Map question text to our standard keys
        question_map = {
            "To what extent is your perspective represented": "perspective_representation",
            "How informative": "informativeness",
            "neutral and balanced": "neutrality_balance",
            "approve": "policy_approval"
        }
        
        for question, scales in annotation['label_annotations'].items():
            for key_part, standard_key in question_map.items():
                if key_part in question:
                    # Get the rating value
                    for scale, value in scales.items():
                        if value and str(value).isdigit():
                            ratings[standard_key] = int(value)
                            break
                    break
        
        return ratings
    
    def extract_human_comparisons(self, annotation: Dict) -> Dict[str, int]:
        """Extract human comparison choices from annotation"""
        comparisons = {}
        
        if 'label_annotations' not in annotation:
            return comparisons
        
        # Map question text to our standard keys
        question_map = {
            "more representative": "perspective_representation",
            "more informative": "informativeness",
            "neutral and balanced": "neutrality_balance",
            "prefer": "policy_approval"
        }
        
        for question, scales in annotation['label_annotations'].items():
            for key_part, standard_key in question_map.items():
                if key_part in question:
                    # Get the choice (1 or 2)
                    for scale, value in scales.items():
                        if value and str(value).isdigit():
                            comparisons[standard_key] = int(value)
                            break
                    break
        
        return comparisons
