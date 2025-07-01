#!/usr/bin/env python3
"""
Adaptive Multi-Pass Compression with Diminishing Returns Analysis

This module implements intelligent multi-pass compression that dynamically
adjusts parameters based on previous passes and stops when benefits diminish.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class AdaptiveMultiPassCompressor:
    """Intelligent multi-pass compression with diminishing returns analysis."""
    
    def __init__(self, analyzer=None):
        """Initialize with reference to the dynamic dictionary analyzer."""
        self.analyzer = analyzer
        self.compression_history = []
        
    def compress_adaptive(self, text: str) -> Tuple[str, Dict[str, str], Dict]:
        """
        Apply adaptive multi-pass compression with diminishing returns analysis.
        
        Returns:
            Tuple of (compressed_text, all_substitutions, compression_metrics)
        """
        start_time = time.time()
        
        compressed = text
        all_substitutions = {}
        pass_num = 0
        total_original_length = len(text)
        
        logger.debug(f"🔄 Starting adaptive multi-pass compression on {len(text)} characters")
        
        compression_history = []
        
        while pass_num < 10:  # Safety limit
            pass_num += 1
            pass_start_time = time.time()
            
            # Balanced parameters that don't select too many low-value patterns
            min_freq = max(2, 3 - pass_num // 3)  # Start at 3, reduce to 2 slowly
            min_length = max(3, 8 - pass_num)      # Start at 8, reduce to 3
            
            logger.debug(f"Pass {pass_num}: min_freq={min_freq}, min_length={min_length}")
            
            # Analyze current state with adapted parameters
            if not self.analyzer:
                logger.warning("No analyzer available for adaptive compression")
                break
            
            # Temporarily adjust analyzer configuration
            original_min_freq = self.analyzer.config.get("min_frequency", 2)
            original_min_length = self.analyzer.config.get("min_token_length", 3)
            
            self.analyzer.config["min_frequency"] = min_freq
            self.analyzer.config["min_token_length"] = min_length
            
            try:
                # Analyze current compressed text
                analysis = self.analyzer.analyze_prompt(compressed)
                
                # Restore original configuration
                self.analyzer.config["min_frequency"] = original_min_freq
                self.analyzer.config["min_token_length"] = original_min_length
                
                # Check if compression is beneficial
                compression_ratio = analysis.get("compression_ratio", 0)
                
                if not analysis.get("assignments"):
                    logger.debug(f"Pass {pass_num}: No patterns found, stopping")
                    break
                
                # Apply compression
                compressed_before = compressed
                compressed, used_substitutions = self._apply_substitutions(
                    compressed, analysis.get("assignments", {})
                )
                
                # Track pass metrics
                pass_metrics = {
                    "pass_number": pass_num,
                    "before_length": len(compressed_before),
                    "after_length": len(compressed),
                    "patterns_found": len(analysis.get("assignments", {})),
                    "compression_ratio": compression_ratio,
                    "time_seconds": time.time() - pass_start_time
                }
                compression_history.append(pass_metrics)
                
                # Update all substitutions
                all_substitutions.update(used_substitutions)
                
                logger.debug(f"Pass {pass_num}: {len(compressed_before)} -> {len(compressed)} chars "
                           f"({len(used_substitutions)} patterns, ratio: {compression_ratio:.4f})")
                
                # Calculate diminishing returns if we have previous passes
                if pass_num > 1:
                    improvement_rate = self._calculate_improvement_rate(
                        compression_history[-2], compression_history[-1], total_original_length
                    )
                    
                    logger.debug(f"Pass {pass_num}: Improvement rate: {improvement_rate:.4f}")
                    
                    # Stop if improvement rate is too low
                    if improvement_rate < 0.005:  # Less than 0.5% additional improvement
                        logger.debug(f"Pass {pass_num}: Diminishing returns detected, stopping")
                        break
                    
                    # Also stop if no actual character reduction happened
                    if len(compressed) >= len(compressed_before):
                        logger.debug(f"Pass {pass_num}: No character reduction, stopping")
                        break
                
                # Safety check for minimum compression benefit
                if pass_num > 3 and compression_ratio < 0.001:
                    logger.debug(f"Pass {pass_num}: Very low compression ratio, stopping")
                    break
                    
            except Exception as e:
                logger.error(f"Pass {pass_num} failed: {e}")
                # Restore original configuration
                self.analyzer.config["min_frequency"] = original_min_freq
                self.analyzer.config["min_token_length"] = original_min_length
                break
        
        total_time = time.time() - start_time
        
        # Calculate final metrics
        final_metrics = {
            "total_passes": pass_num,
            "original_length": total_original_length,
            "final_length": len(compressed),
            "total_compression_ratio": 1 - (len(compressed) / total_original_length),
            "total_patterns": len(all_substitutions),
            "total_time_seconds": total_time,
            "passes_per_second": pass_num / total_time if total_time > 0 else 0,
            "compression_history": compression_history
        }
        
        logger.info(f"🎯 Adaptive compression completed in {pass_num} passes ({total_time:.2f}s)")
        logger.info(f"   {total_original_length} -> {len(compressed)} chars "
                   f"({final_metrics['total_compression_ratio']*100:.1f}% reduction)")
        logger.info(f"   Used {len(all_substitutions)} patterns total")
        
        return compressed, all_substitutions, final_metrics
    
    def _calculate_improvement_rate(self, prev_pass: Dict, curr_pass: Dict, original_length: int) -> float:
        """Calculate the improvement rate between two passes."""
        # Previous compression ratio
        prev_ratio = 1 - (prev_pass["after_length"] / original_length)
        
        # Current compression ratio
        curr_ratio = 1 - (curr_pass["after_length"] / original_length)
        
        # Improvement rate is the additional compression gained
        improvement = curr_ratio - prev_ratio
        
        return improvement
    
    def _apply_substitutions(self, text: str, substitutions: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        """Apply substitutions to text and return both result and actually used substitutions."""
        if not substitutions:
            return text, {}
        
        used_substitutions = {}
        modified_text = text
        
        # Sort substitutions by pattern length (longest first) to avoid conflicts
        sorted_substitutions = sorted(
            substitutions.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for pattern, symbol in sorted_substitutions:
            if pattern in modified_text:
                # Count occurrences before replacement
                count_before = modified_text.count(pattern)
                
                # Apply replacement
                modified_text = modified_text.replace(pattern, symbol)
                
                # Verify replacement actually happened
                count_after = modified_text.count(pattern)
                actual_replacements = count_before - count_after
                
                if actual_replacements > 0:
                    # CRITICAL FIX: Return symbol -> pattern mapping for decompression
                    used_substitutions[symbol] = pattern
                    logger.debug(f"Applied substitution: '{pattern}' -> '{symbol}' ({actual_replacements} times)")
        
        return modified_text, used_substitutions
    



# Global instance
_adaptive_compressor = None

def get_adaptive_multipass_compressor(analyzer=None) -> AdaptiveMultiPassCompressor:
    """Get singleton instance of AdaptiveMultiPassCompressor."""
    global _adaptive_compressor
    if _adaptive_compressor is None:
        _adaptive_compressor = AdaptiveMultiPassCompressor(analyzer)
    elif analyzer and not _adaptive_compressor.analyzer:
        _adaptive_compressor.analyzer = analyzer
    return _adaptive_compressor
