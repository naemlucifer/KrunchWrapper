#!/usr/bin/env python3
"""
Parameterized Pattern Detection for Advanced Compression

This module detects patterns with variable parts (parameters) that can be
compressed more efficiently by abstracting the common structure.
"""

import re
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParameterizedPattern:
    """Represents a pattern with parameterized parts."""
    template: str  # Pattern template with placeholders
    instances: List[str]  # Actual instances of the pattern
    parameters: List[List[str]]  # Parameters for each instance
    frequency: int  # Total frequency
    savings_potential: float  # Estimated compression savings

class ParameterizedPatternDetector:
    """Detect patterns with variable parts for enhanced compression."""
    
    def __init__(self, min_instances: int = 3, min_template_length: int = 8):
        self.min_instances = min_instances
        self.min_template_length = min_template_length
        
        # Pre-compiled regex patterns for efficiency
        self.function_call_pattern = re.compile(r'(\w+(?:\.\w+)*)\s*\(([^)]*)\)')
        self.array_access_pattern = re.compile(r'(\w+(?:\.\w+)*)\s*\[([^\]]+)\]')
        self.assignment_pattern = re.compile(r'(\w+(?:\.\w+)*)\s*=\s*([^=\n;]+)')
        self.method_chain_pattern = re.compile(r'(\w+(?:\.\w+)*)\s*\.\s*(\w+)\s*\(([^)]*)\)')
        self.conditional_pattern = re.compile(r'(if|elif|while)\s+([^:]+):')
        self.loop_pattern = re.compile(r'for\s+(\w+)\s+in\s+([^:]+):')
        
    def detect_parameterized_patterns(self, text: str) -> List[ParameterizedPattern]:
        """Main entry point to detect all parameterized patterns."""
        patterns = []
        
        # Detect different types of parameterized patterns
        patterns.extend(self._detect_function_call_patterns(text))
        patterns.extend(self._detect_array_access_patterns(text))
        patterns.extend(self._detect_logging_patterns(text))
        
        # Filter and rank patterns
        return self._filter_and_rank_patterns(patterns)
    
    def _detect_function_call_patterns(self, text: str) -> List[ParameterizedPattern]:
        """Detect function calls with different arguments."""
        function_calls = defaultdict(list)
        
        for match in self.function_call_pattern.finditer(text):
            function_name = match.group(1)
            args = match.group(2).strip()
            full_call = match.group(0)
            
            # Skip very simple calls or very complex ones
            if not args or len(args) > 100:
                continue
                
            function_calls[function_name].append((full_call, args))
        
        patterns = []
        for func_name, calls in function_calls.items():
            if len(calls) >= self.min_instances:
                # Create template and extract parameters
                template = f"{func_name}({{}})"
                instances = [call[0] for call in calls]
                parameters = [[call[1]] for call in calls]
                
                pattern = ParameterizedPattern(
                    template=template,
                    instances=instances,
                    parameters=parameters,
                    frequency=len(calls),
                    savings_potential=self._calculate_savings_potential(template, instances)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_array_access_patterns(self, text: str) -> List[ParameterizedPattern]:
        """Detect array/dict access patterns with different keys."""
        access_patterns = defaultdict(list)
        
        for match in self.array_access_pattern.finditer(text):
            var_name = match.group(1)
            index = match.group(2).strip()
            full_access = match.group(0)
            
            if not index or len(index) > 50:
                continue
                
            access_patterns[var_name].append((full_access, index))
        
        patterns = []
        for var_name, accesses in access_patterns.items():
            if len(accesses) >= self.min_instances:
                template = f"{var_name}[{{}}]"
                instances = [access[0] for access in accesses]
                parameters = [[access[1]] for access in accesses]
                
                pattern = ParameterizedPattern(
                    template=template,
                    instances=instances,
                    parameters=parameters,
                    frequency=len(accesses),
                    savings_potential=self._calculate_savings_potential(template, instances)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_logging_patterns(self, text: str) -> List[ParameterizedPattern]:
        """Detect logging patterns with different messages."""
        logging_patterns = defaultdict(list)
        
        log_methods = ['debug', 'info', 'warning', 'error', 'critical']
        
        for method in log_methods:
            # logger.method() patterns
            pattern = re.compile(rf'(\w*logger\w*)\s*\.\s*{method}\s*\(([^)]+)\)')
            for match in pattern.finditer(text):
                logger_name = match.group(1)
                message = match.group(2).strip()
                full_log = match.group(0)
                
                if len(message) > 10 and len(message) < 120:
                    key = f'{logger_name}.{method}'
                    logging_patterns[key].append((full_log, message))
        
        patterns = []
        for log_key, logs in logging_patterns.items():
            if len(logs) >= self.min_instances:
                template = f"{log_key}({{}})"
                instances = [log[0] for log in logs]
                parameters = [[log[1]] for log in logs]
                
                pattern = ParameterizedPattern(
                    template=template,
                    instances=instances,
                    parameters=parameters,
                    frequency=len(logs),
                    savings_potential=self._calculate_savings_potential(template, instances)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_savings_potential(self, template: str, instances: List[str]) -> float:
        """Calculate the potential compression savings for a parameterized pattern."""
        if not instances:
            return 0.0
        
        # Calculate average instance length
        avg_instance_length = sum(len(instance) for instance in instances) / len(instances)
        
        # Template length is the compressed size
        template_length = len(template)
        
        # Savings per instance
        savings_per_instance = max(0, avg_instance_length - template_length)
        
        # Total savings considering frequency
        total_savings = savings_per_instance * len(instances)
        
        # Account for dictionary overhead (approximate)
        dictionary_overhead = len(f"{template}=SYMBOL, ")
        
        # Net savings
        net_savings = total_savings - dictionary_overhead
        
        return max(0, net_savings)
    
    def _filter_and_rank_patterns(self, patterns: List[ParameterizedPattern]) -> List[ParameterizedPattern]:
        """Filter and rank patterns by their compression potential."""
        # Filter patterns that meet minimum requirements
        filtered_patterns = []
        
        for pattern in patterns:
            # Must have enough instances
            if pattern.frequency < self.min_instances:
                continue
            
            # Template must be long enough to be worthwhile
            if len(pattern.template) < self.min_template_length:
                continue
            
            # Must have positive savings potential
            if pattern.savings_potential <= 0:
                continue
            
            filtered_patterns.append(pattern)
        
        # Sort by savings potential (highest first)
        filtered_patterns.sort(key=lambda p: p.savings_potential, reverse=True)
        
        return filtered_patterns
    
    def convert_to_standard_patterns(self, parameterized_patterns: List[ParameterizedPattern]) -> List[Tuple[str, int]]:
        """Convert parameterized patterns to standard (pattern, frequency) format."""
        standard_patterns = []
        
        for param_pattern in parameterized_patterns:
            if param_pattern.instances:
                # Use the template itself as the pattern for compression
                standard_patterns.append((param_pattern.template, param_pattern.frequency))
                
                # Also add the most frequent instance variations if they're common enough
                instance_counts = Counter(param_pattern.instances)
                for instance, count in instance_counts.most_common(3):  # Top 3 instances
                    if count >= max(2, param_pattern.frequency // 4):  # At least 25% of occurrences or 2+
                        standard_patterns.append((instance, count))
        
        return standard_patterns


# Global instance
_parameterized_detector = None

def get_parameterized_pattern_detector() -> ParameterizedPatternDetector:
    """Get singleton instance of ParameterizedPatternDetector."""
    global _parameterized_detector
    if _parameterized_detector is None:
        _parameterized_detector = ParameterizedPatternDetector()
    return _parameterized_detector
