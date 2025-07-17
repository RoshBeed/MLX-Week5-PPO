from collections import defaultdict
from typing import List, Dict

class TDError:
    """Computes Temporal Difference errors"""
    
    @staticmethod
    def compute_td_errors(sa_pairs: List[Dict], gamma: float = 0.99) -> List[Dict]:
        """Compute TD error for each (s, a) pair"""
        # Group by prompt_idx
        grouped = defaultdict(list)
        for pair in sa_pairs:
            grouped[pair['prompt_idx']].append(pair)
        
        # For each prompt, sort by step and compute TD error
        for prompt_idx, pairs in grouped.items():
            pairs = sorted(pairs, key=lambda x: x['step'])
            for t, pair in enumerate(pairs):
                reward = pair['adjusted_reward']
                value = pair['value']
                # Next value: value of next step, or 0 if last
                if t < len(pairs) - 1:
                    next_value = pairs[t + 1]['value']
                else:
                    next_value = 0.0
                pair['td_error'] = reward + gamma * next_value - value
        
        return sa_pairs 