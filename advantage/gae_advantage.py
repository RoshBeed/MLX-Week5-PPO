from collections import defaultdict
from typing import List, Dict

class GAEAdvantage:
    """Computes GAE advantages"""
    
    @staticmethod
    def compute_gae_advantages(sa_pairs: List[Dict], gamma: float = 0.99, 
                             lam: float = 0.95) -> List[Dict]:
        """Compute GAE advantage for each (s, a) pair"""
        # Group by prompt_idx
        grouped = defaultdict(list)
        for pair in sa_pairs:
            grouped[pair['prompt_idx']].append(pair)
        
        for prompt_idx, pairs in grouped.items():
            # Sort by step
            pairs = sorted(pairs, key=lambda x: x['step'])
            num_steps = len(pairs)
            advantages = [0.0] * num_steps
            gae = 0.0
            
            # Go backwards through the sequence
            for t in reversed(range(num_steps)):
                td_error = pairs[t]['td_error']
                gae = td_error + gamma * lam * gae
                advantages[t] = gae
            
            # Assign to sa_pairs
            for t in range(num_steps):
                pairs[t]['advantage'] = advantages[t]
        
        return sa_pairs 