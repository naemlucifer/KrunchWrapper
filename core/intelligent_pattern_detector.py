#!/usr/bin/env python3
"""
Intelligent Pattern Detection for Dynamic Compression
Uses multiple approaches to identify meaningful patterns without hardcoding.
"""

import ast
import re
import math
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

# Optional dependencies - gracefully handle if missing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PatternCandidate:
    """Represents a potential compression pattern with metadata."""
    text: str
    frequency: int
    contexts: List[str]  # Surrounding text contexts
    pattern_type: str   # 'structural', 'semantic', 'repeated', etc.
    confidence: float   # 0.0 to 1.0
    entropy: float      # Information content
    structural_score: float  # How structural vs content-specific
    
class IntelligentPatternDetector:
    """Advanced pattern detection using multiple approaches."""
    
    def __init__(self, min_frequency: int = 3, min_length: int = 5):
        self.min_frequency = min_frequency
        self.min_length = min_length
        self.nlp = self._load_spacy_model()
        
    def _load_spacy_model(self):
        """Load spaCy model if available."""
        if not SPACY_AVAILABLE:
            return None
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            return None
    
    def detect_patterns(self, text: str) -> List[PatternCandidate]:
        """Main entry point - detect patterns using multiple approaches."""
        candidates = []
        
        # Approach 1: Statistical N-gram Analysis
        candidates.extend(self._detect_statistical_patterns(text))
        
        # Approach 2: AST-based Structural Analysis (for code)
        if self._looks_like_code(text):
            candidates.extend(self._detect_ast_patterns(text))
        
        # Approach 3: Semantic Clustering (if sklearn available)
        if SKLEARN_AVAILABLE:
            candidates.extend(self._detect_semantic_clusters(text))
        
        # Approach 4: Linguistic Structure Analysis (if spaCy available)
        if self.nlp:
            candidates.extend(self._detect_linguistic_patterns(text))
        
        # Approach 5: Entropy-based Significance Testing
        candidates = self._filter_by_entropy(candidates, text)
        
        # Approach 6: Context-based Structural Scoring
        candidates = self._score_structural_significance(candidates, text)
        
        # Deduplicate and rank
        return self._deduplicate_and_rank(candidates)
    
    def _detect_statistical_patterns(self, text: str) -> List[PatternCandidate]:
        """Detect patterns using statistical n-gram analysis."""
        candidates = []
        
        # Extract n-grams of various lengths
        for n in range(2, 6):  # 2-grams to 5-grams
            ngrams = self._extract_ngrams(text, n)
            
            for ngram, frequency in ngrams.items():
                if frequency >= self.min_frequency and len(ngram) >= self.min_length:
                    # Calculate entropy (information content)
                    entropy = self._calculate_entropy(ngram, text)
                    
                    # Get contexts where this pattern appears
                    contexts = self._get_contexts(ngram, text)
                    
                    candidates.append(PatternCandidate(
                        text=ngram,
                        frequency=frequency,
                        contexts=contexts,
                        pattern_type='statistical',
                        confidence=min(1.0, frequency / 10.0),  # Basic confidence
                        entropy=entropy,
                        structural_score=0.0  # Will be calculated later
                    ))
        
        return candidates
    
    def _detect_ast_patterns(self, text: str) -> List[PatternCandidate]:
        """Detect structural patterns using AST parsing."""
        candidates = []
        
        try:
            # Try to parse as Python code
            tree = ast.parse(text)
            
            # Extract structural patterns
            patterns = self._extract_ast_patterns(tree)
            
            for pattern, frequency in patterns.items():
                if frequency >= self.min_frequency:
                    candidates.append(PatternCandidate(
                        text=pattern,
                        frequency=frequency,
                        contexts=[],  # AST patterns don't have text contexts
                        pattern_type='structural',
                        confidence=0.9,  # High confidence for AST patterns
                        entropy=self._calculate_entropy(pattern, text),
                        structural_score=0.8  # AST patterns are inherently structural
                    ))
                    
        except SyntaxError:
            # Not valid Python code, try other approaches
            candidates.extend(self._detect_code_like_patterns(text))
        
        return candidates
    
    def _detect_semantic_clusters(self, text: str) -> List[PatternCandidate]:
        """Use machine learning to find semantically similar patterns."""
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            return []
        
        candidates = []
        
        # Extract all potential patterns
        potential_patterns = []
        for length in range(self.min_length, 50):  # Up to 50 chars
            for start in range(len(text) - length + 1):
                pattern = text[start:start + length]
                if self._is_meaningful_pattern(pattern):
                    potential_patterns.append(pattern)
        
        if len(potential_patterns) < 10:  # Need enough patterns for clustering
            return candidates
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                min_df=2,
                max_features=1000
            )
            
            # Count pattern frequencies
            pattern_counts = Counter(potential_patterns)
            unique_patterns = list(pattern_counts.keys())
            
            if len(unique_patterns) < 5:
                return candidates
            
            # Vectorize patterns
            tfidf_matrix = vectorizer.fit_transform(unique_patterns)
            
            # Cluster similar patterns
            clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(tfidf_matrix)
            
            # Extract patterns from clusters
            for cluster_idx in set(cluster_labels):
                if cluster_idx == -1:  # Noise cluster
                    continue
                
                cluster_patterns = [unique_patterns[i] for i, label in enumerate(cluster_labels) if label == cluster_idx]
                
                # Find the most representative pattern in the cluster
                if cluster_patterns:
                    # Use the most frequent pattern as representative
                    representative = max(cluster_patterns, key=lambda p: pattern_counts[p])
                    frequency = pattern_counts[representative]
                    
                    if frequency >= self.min_frequency:
                        candidates.append(PatternCandidate(
                            text=representative,
                            frequency=frequency,
                            contexts=self._get_contexts(representative, text),
                            pattern_type='semantic_cluster',
                            confidence=len(cluster_patterns) / len(unique_patterns),
                            entropy=self._calculate_entropy(representative, text),
                            structural_score=0.5  # Medium structural score
                        ))
        
        except Exception as e:
            logger.debug(f"Semantic clustering failed: {e}")
        
        return candidates
    
    def _detect_linguistic_patterns(self, text: str) -> List[PatternCandidate]:
        """Use NLP to detect linguistic patterns."""
        if not self.nlp:
            return []
        
        candidates = []
        
        try:
            doc = self.nlp(text)
            
            # Extract patterns based on linguistic features
            patterns = defaultdict(int)
            
            # Named entity patterns
            for ent in doc.ents:
                if len(ent.text) >= self.min_length:
                    patterns[ent.text] += 1
            
            # Noun phrase patterns
            for chunk in doc.noun_chunks:
                if len(chunk.text) >= self.min_length:
                    patterns[chunk.text] += 1
            
            # Verb phrase patterns
            for token in doc:
                if token.pos_ == "VERB" and token.head == token:
                    phrase = self._extract_verb_phrase(token)
                    if len(phrase) >= self.min_length:
                        patterns[phrase] += 1
            
            # Convert to candidates
            for pattern, frequency in patterns.items():
                if frequency >= self.min_frequency:
                    candidates.append(PatternCandidate(
                        text=pattern,
                        frequency=frequency,
                        contexts=self._get_contexts(pattern, text),
                        pattern_type='linguistic',
                        confidence=0.7,
                        entropy=self._calculate_entropy(pattern, text),
                        structural_score=0.3  # Linguistic patterns are less structural
                    ))
        
        except Exception as e:
            logger.debug(f"Linguistic analysis failed: {e}")
        
        return candidates
    
    def _filter_by_entropy(self, candidates: List[PatternCandidate], text: str) -> List[PatternCandidate]:
        """Filter candidates based on information entropy."""
        if not candidates:
            return candidates
        
        # Calculate threshold based on text characteristics
        text_entropy = self._calculate_entropy(text[:1000], text)  # Sample for efficiency
        entropy_threshold = text_entropy * 0.3  # Patterns should have reasonable information content
        
        return [c for c in candidates if c.entropy >= entropy_threshold]
    
    def _score_structural_significance(self, candidates: List[PatternCandidate], text: str) -> List[PatternCandidate]:
        """Score how structural vs content-specific each pattern is."""
        for candidate in candidates:
            structural_score = 0.0
            
            # Boost score for programming constructs
            if self._contains_programming_constructs(candidate.text):
                structural_score += 0.4
            
            # Boost score for repeated formatting patterns
            if self._is_formatting_pattern(candidate.text):
                structural_score += 0.3
            
            # Boost score for function/method calls
            if self._is_function_call_pattern(candidate.text):
                structural_score += 0.3
            
            # Penalize content-specific patterns
            if self._is_content_specific(candidate.text):
                structural_score -= 0.3
            
            candidate.structural_score = max(0.0, min(1.0, structural_score))
        
        return candidates
    
    def _deduplicate_and_rank(self, candidates: List[PatternCandidate]) -> List[PatternCandidate]:
        """Remove duplicates and rank by overall quality."""
        # Remove exact duplicates
        seen = set()
        unique_candidates = []
        
        for candidate in candidates:
            if candidate.text not in seen:
                seen.add(candidate.text)
                unique_candidates.append(candidate)
        
        # Remove substring conflicts (keep longer patterns)
        filtered_candidates = []
        unique_candidates.sort(key=lambda x: len(x.text), reverse=True)
        
        for candidate in unique_candidates:
            conflicts = False
            for existing in filtered_candidates:
                if candidate.text in existing.text or existing.text in candidate.text:
                    conflicts = True
                    break
            if not conflicts:
                filtered_candidates.append(candidate)
        
        # Calculate overall quality score and rank
        for candidate in filtered_candidates:
            quality_score = (
                candidate.confidence * 0.3 +
                candidate.structural_score * 0.4 +
                min(1.0, candidate.frequency / 10.0) * 0.2 +
                min(1.0, candidate.entropy / 5.0) * 0.1
            )
            candidate.confidence = quality_score  # Reuse confidence field for final score
        
        # Sort by quality score
        filtered_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered_candidates
    
    # Helper methods
    def _extract_ngrams(self, text: str, n: int) -> Dict[str, int]:
        """Extract n-grams from text."""
        ngrams = Counter()
        words = text.split()
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            if len(ngram) >= self.min_length:
                ngrams[ngram] += 1
        
        return dict(ngrams)
    
    def _calculate_entropy(self, pattern: str, text: str) -> float:
        """Calculate information entropy of a pattern."""
        if not pattern:
            return 0.0
        
        # Count character frequencies in the pattern
        char_counts = Counter(pattern)
        pattern_length = len(pattern)
        
        entropy = 0.0
        for count in char_counts.values():
            probability = count / pattern_length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _get_contexts(self, pattern: str, text: str, context_size: int = 20) -> List[str]:
        """Get surrounding contexts where pattern appears."""
        contexts = []
        start = 0
        
        while start < len(text):
            pos = text.find(pattern, start)
            if pos == -1:
                break
            
            # Extract context
            context_start = max(0, pos - context_size)
            context_end = min(len(text), pos + len(pattern) + context_size)
            context = text[context_start:context_end]
            contexts.append(context)
            
            start = pos + 1
        
        return contexts[:10]  # Limit number of contexts
    
    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to determine if text looks like code."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
            '()', '{}', '[]', '=>', '===', '!==', '&&', '||',
            'function', 'var ', 'let ', 'const ', 'return'
        ]
        
        indicator_count = sum(1 for indicator in code_indicators if indicator in text)
        return indicator_count >= 3
    
    def _extract_ast_patterns(self, tree: ast.AST) -> Dict[str, int]:
        """Extract patterns from AST."""
        patterns = Counter()
        
        for node in ast.walk(tree):
            # Function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    patterns[f"{node.func.id}()"] += 1
                elif isinstance(node.func, ast.Attribute):
                    patterns[f".{node.func.attr}()"] += 1
            
            # Attribute access
            elif isinstance(node, ast.Attribute):
                patterns[f".{node.attr}"] += 1
            
            # Control flow
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                patterns[node.__class__.__name__.lower()] += 1
        
        return dict(patterns)
    
    def _detect_code_like_patterns(self, text: str) -> List[PatternCandidate]:
        """Detect code-like patterns using regex when AST fails."""
        candidates = []
        
        # Common code patterns
        patterns = {
            r'\w+\([^)]*\)': 'function_call',
            r'\w+\.\w+': 'attribute_access',
            r'def\s+\w+': 'function_def',
            r'class\s+\w+': 'class_def',
        }
        
        for pattern, pattern_type in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                match_counts = Counter(matches)
                for match, count in match_counts.items():
                    if count >= self.min_frequency and len(match) >= self.min_length:
                        candidates.append(PatternCandidate(
                            text=match,
                            frequency=count,
                            contexts=self._get_contexts(match, text),
                            pattern_type=f'code_{pattern_type}',
                            confidence=0.8,
                            entropy=self._calculate_entropy(match, text),
                            structural_score=0.7
                        ))
        
        return candidates
    
    def _is_meaningful_pattern(self, pattern: str) -> bool:
        """Check if a pattern is potentially meaningful."""
        # Skip patterns that are too short or too long
        if len(pattern) < self.min_length or len(pattern) > 100:
            return False
        
        # Skip patterns that are mostly whitespace or punctuation
        alpha_ratio = sum(1 for c in pattern if c.isalnum()) / len(pattern)
        if alpha_ratio < 0.3:
            return False
        
        return True
    
    def _extract_verb_phrase(self, verb_token) -> str:
        """Extract verb phrase from spaCy token."""
        phrase_parts = [verb_token.text]
        
        # Add direct objects and modifiers
        for child in verb_token.children:
            if child.dep_ in ('dobj', 'pobj', 'advmod'):
                phrase_parts.append(child.text)
        
        return ' '.join(phrase_parts)
    
    def _contains_programming_constructs(self, text: str) -> bool:
        """Check if text contains programming constructs."""
        constructs = [
            'def ', 'class ', 'import ', 'from ', 'return ', 'yield ',
            'if ', 'else', 'elif', 'for ', 'while ', 'try:', 'except',
            'lambda', 'async ', 'await ', 'with ', 'as '
        ]
        return any(construct in text.lower() for construct in constructs)
    
    def _is_formatting_pattern(self, text: str) -> bool:
        """Check if text is a formatting pattern."""
        # Check for emoji/symbol patterns
        if re.search(r'[🚨🗜️📝📊⚖️🔄🧠📈✅⚠️➡️❌]', text):
            return True
        
        # Check for bracket patterns
        if re.search(r'\[[\w\s]+\]', text):
            return True
        
        return False
    
    def _is_function_call_pattern(self, text: str) -> bool:
        """Check if text is a function call pattern."""
        return bool(re.search(r'\w+\s*\(.*\)', text))
    
    def _is_content_specific(self, text: str) -> bool:
        """Check if pattern is content-specific rather than structural."""
        content_indicators = [
            'payload[', 'data[', 'result[', 'message[', 'content[',
            'enumerate(', 'zip(', 'range(', 'len(',
            '== "', '!= "', "== '", "!= '",
        ]
        
        return any(indicator in text.lower() for indicator in content_indicators)

# Integration with existing system
def get_intelligent_patterns(text: str, min_frequency: int = 5, min_length: int = 6) -> List[Dict]:
    """Get intelligent patterns compatible with existing system."""
    detector = IntelligentPatternDetector(min_frequency=min_frequency, min_length=min_length)
    candidates = detector.detect_patterns(text)
    
    # Convert to format expected by existing system
    patterns = []
    for candidate in candidates:
        patterns.append({
            'token': candidate.text,
            'count': candidate.frequency,
            'pattern_type': candidate.pattern_type,
            'confidence': candidate.confidence,
            'structural_score': candidate.structural_score,
            'entropy': candidate.entropy
        })
    
    return patterns 