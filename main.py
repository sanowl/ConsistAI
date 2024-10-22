import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from collections import defaultdict
import networkx as nx
from scipy.stats import wasserstein_distance
import json
from datetime import datetime

@dataclass
class EditInstance:
    """Represents a knowledge edit with comprehensive metadata"""
    id: str
    subject: str
    relation: str
    original_object: str
    target_object: str
    domain: str
    topic: str
    edit_time: datetime
    context_window: List[str]
    verification_queries: List[str]
    semantic_neighbors: List[Tuple[str, float]]
    confidence_threshold: float = 0.8

class KnowledgeGraph:
    """Maintains knowledge relationships and semantic consistency"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.relation_types = set()
        self.entity_embeddings = {}
        
    def add_edit(self, edit: EditInstance):
        """Add edit to knowledge graph and update relationships"""
        self.graph.add_edge(edit.subject, edit.target_object, 
                           relation=edit.relation,
                           confidence=edit.confidence_threshold,
                           edit_time=edit.edit_time)
        
        # Track original relationship for consistency checking
        self.graph.add_edge(edit.subject, edit.original_object,
                           relation=edit.relation,
                           original=True,
                           deprecated_time=edit.edit_time)
        
        self.relation_types.add(edit.relation)
        
    def check_consistency(self, entity: str) -> Dict[str, List[str]]:
        """Verify temporal and relational consistency for an entity"""
        inconsistencies = defaultdict(list)
        
        # Check temporal consistency
        edges = self.graph.out_edges(entity, data=True)
        for _, target, data in edges:
            if data.get('original'):
                # Find any conflicting current edges
                current_edges = [e for e in edges 
                               if e[2]['relation'] == data['relation'] and
                               not e[2].get('original')]
                               
                if current_edges:
                    inconsistencies['temporal'].append(
                        f"Conflict: {entity} -> {target} vs {current_edges[0][1]}"
                    )
                    
        # Check relational consistency
        relations = [d['relation'] for _, _, d in edges]
        for r1, r2 in zip(relations, relations[1:]):
            if self._are_mutually_exclusive(r1, r2):
                inconsistencies['relational'].append(f"Mutually exclusive: {r1}, {r2}")
                
        return inconsistencies

    def _are_mutually_exclusive(self, rel1: str, rel2: str) -> bool:
        """Check if relations are mutually exclusive"""
        exclusivity_rules = {
            'birthPlace': {'deathPlace'},
            'employer': {'founder'},
            'parent': {'child'}
        }
        return rel2 in exclusivity_rules.get(rel1, set())

class SemanticDriftTracker:
    """Tracks semantic changes in edited knowledge"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.baseline_embeddings = {}
        self.drift_threshold = 0.15
        
    def compute_semantic_embedding(self, text: str) -> torch.Tensor:
        """Generate semantic embedding for text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)
    
    def track_drift(self, edit: EditInstance) -> Dict[str, float]:
        """Measure semantic drift caused by edit"""
        key = f"{edit.subject}_{edit.relation}"
        
        if key not in self.baseline_embeddings:
            self.baseline_embeddings[key] = self.compute_semantic_embedding(
                f"{edit.subject} {edit.relation} {edit.original_object}"
            )
            
        current_embedding = self.compute_semantic_embedding(
            f"{edit.subject} {edit.relation} {edit.target_object}"
        )
        
        drift_metrics = {
            'cosine_drift': float(F.cosine_similarity(
                self.baseline_embeddings[key], 
                current_embedding
            )),
            'euclidean_drift': float(torch.norm(
                self.baseline_embeddings[key] - current_embedding
            ))
        }
        
        # Check semantic neighbors for collective drift
        neighbor_drifts = []
        for neighbor, _ in edit.semantic_neighbors:
            neighbor_emb = self.compute_semantic_embedding(neighbor)
            neighbor_drifts.append(float(F.cosine_similarity(
                self.baseline_embeddings[key],
                neighbor_emb
            )))
            
        drift_metrics['neighbor_drift'] = np.mean(neighbor_drifts)
        drift_metrics['drift_severity'] = 'high' if any(
            d < self.drift_threshold for d in neighbor_drifts
        ) else 'low'
        
        return drift_metrics

class ChainOfThoughtVerifier:
    """Verifies edits using chain-of-thought reasoning"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reasoning_templates = self._load_reasoning_templates()
        
    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load templates for different relation types"""
        return {
            "birthPlace": """
                Let's verify this step by step:
                1. Check if {subject} was a real person
                2. Verify if {target} existed during {subject}'s birth
                3. Look for historical records connecting {subject} to {target}
                4. Check for any conflicting birth locations
                Therefore, the statement that {subject} was born in {target} is:
            """,
            "employer": """
                Let's analyze this employment relationship:
                1. Confirm {subject}'s career timeline
                2. Verify {target}'s operational dates
                3. Check for overlapping time periods
                4. Look for official documentation
                Based on this, the employment relationship between {subject} and {target} is:
            """
        }
        
    def verify_edit(self, edit: EditInstance) -> Dict[str, Any]:
        """Perform chain-of-thought verification of edit"""
        template = self.reasoning_templates.get(
            edit.relation, 
            "Let's verify if {subject} has relation {relation} with {target}:"
        )
        
        reasoning_prompt = template.format(
            subject=edit.subject,
            target=edit.target_object,
            relation=edit.relation
        )
        
        inputs = self.tokenizer(reasoning_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_beams=3,
                temperature=0.7,
                do_sample=True
            )
            
        reasoning = self.tokenizer.decode(outputs[0])
        
        # Extract key verification points
        verification_result = {
            'reasoning_chain': reasoning,
            'verification_queries': [
                self._generate_verification_query(edit, point)
                for point in self._extract_reasoning_points(reasoning)
            ],
            'confidence_score': self._calculate_verification_confidence(reasoning),
            'potential_conflicts': self._identify_conflicts(edit, reasoning)
        }
        
        return verification_result
    
    def _generate_verification_query(self, edit: EditInstance, point: str) -> str:
        """Generate specific verification query based on reasoning point"""
        return f"Verify: {point} regarding {edit.subject} {edit.relation} {edit.target_object}"
    
    def _extract_reasoning_points(self, reasoning: str) -> List[str]:
        """Extract key points from reasoning chain"""
        points = []
        for line in reasoning.split('\n'):
            if line.strip().startswith(('1.', '2.', '3.', '4.')):
                points.append(line.strip())
        return points
    
    def _calculate_verification_confidence(self, reasoning: str) -> float:
        """Calculate confidence score based on reasoning chain"""
        positive_indicators = ['confirmed', 'verified', 'documented', 'proven']
        negative_indicators = ['uncertain', 'conflicting', 'unclear', 'disputed']
        
        confidence = 0.5  # baseline
        for indicator in positive_indicators:
            if indicator in reasoning.lower():
                confidence += 0.1
        for indicator in negative_indicators:
            if indicator in reasoning.lower():
                confidence -= 0.1
                
        return max(0.0, min(1.0, confidence))
    
    def _identify_conflicts(self, edit: EditInstance, reasoning: str) -> List[str]:
        """Identify potential conflicts in reasoning"""
        conflicts = []
        conflict_indicators = ['however', 'but', 'although', 'contrary']
        
        for indicator in conflict_indicators:
            idx = reasoning.lower().find(indicator)
            if idx != -1:
                # Extract the conflicting statement
                end_idx = reasoning.find('.', idx)
                if end_idx != -1:
                    conflicts.append(reasoning[idx:end_idx + 1].strip())
                    
        return conflicts

class EditEvaluator:
    """Main class for evaluating knowledge edits"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.knowledge_graph = KnowledgeGraph()
        self.drift_tracker = SemanticDriftTracker(model, tokenizer)
        self.thought_verifier = ChainOfThoughtVerifier(model, tokenizer)
        self.edit_history = []
        
    def evaluate_edit(self, edit: EditInstance) -> Dict[str, Any]:
        """Comprehensive evaluation of a knowledge edit"""
        # Track edit in knowledge graph
        self.knowledge_graph.add_edit(edit)
        
        # Perform evaluations
        consistency_check = self.knowledge_graph.check_consistency(edit.subject)
        semantic_drift = self.drift_tracker.track_drift(edit)
        verification = self.thought_verifier.verify_edit(edit)
        
        # Compile evaluation results
        evaluation = {
            'edit_id': edit.id,
            'timestamp': edit.edit_time.isoformat(),
            'consistency': consistency_check,
            'semantic_drift': semantic_drift,
            'verification': verification,
            'confidence_score': verification['confidence_score'],
            'status': self._determine_edit_status(
                consistency_check,
                semantic_drift,
                verification
            )
        }
        
        self.edit_history.append(evaluation)
        return evaluation
    
    def _determine_edit_status(
        self,
        consistency: Dict[str, List[str]],
        drift: Dict[str, float],
        verification: Dict[str, Any]
    ) -> str:
        """Determine overall status of edit"""
        if consistency['temporal'] or consistency['relational']:
            return 'rejected_consistency'
        if drift['drift_severity'] == 'high':
            return 'rejected_drift'
        if verification['confidence_score'] < 0.6:
            return 'rejected_verification'
        return 'accepted'
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        total_edits = len(self.edit_history)
        accepted_edits = sum(1 for e in self.edit_history 
                           if e['status'] == 'accepted')
        
        report = {
            'summary': {
                'total_edits': total_edits,
                'accepted_rate': accepted_edits / total_edits if total_edits > 0 else 0,
                'rejection_reasons': self._analyze_rejections(),
                'drift_analysis': self._analyze_drift(),
                'consistency_analysis': self._analyze_consistency()
            },
            'detailed_history': self.edit_history
        }
        
        return report
    
    def _analyze_rejections(self) -> Dict[str, int]:
        """Analyze reasons for edit rejections"""
        rejection_counts = defaultdict(int)
        for edit in self.edit_history:
            if edit['status'].startswith('rejected'):
                rejection_counts[edit['status']] += 1
        return dict(rejection_counts)
    
    def _analyze_drift(self) -> Dict[str, float]:
        """Analyze semantic drift patterns"""
        drifts = [e['semantic_drift']['cosine_drift'] 
                 for e in self.edit_history]
        return {
            'mean_drift': np.mean(drifts),
            'max_drift': np.max(drifts),
            'drift_std': np.std(drifts)
        }
    
    def _analyze_consistency(self) -> Dict[str, int]:
        """Analyze consistency violations"""
        temporal_violations = sum(
            1 for e in self.edit_history 
            if e['consistency']['temporal']
        )
        relational_violations = sum(
            1 for e in self.edit_history 
            if e['consistency']['relational']
        )
        return {
            'temporal_violations': temporal_violations,
            'relational_violations': relational_violations
        }