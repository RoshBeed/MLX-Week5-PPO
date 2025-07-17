from typing import List, Dict

class ReturnCalculator:
    """Calculates returns (reward-to-go)"""
    
    @staticmethod
    def add_reward_to_go(sa_pairs: List[Dict]) -> List[Dict]:
        """Add reward-to-go (return) to each (s, a) pair"""
        for pair in sa_pairs:
            pair['reward_to_go'] = pair['advantage'] + pair['value']
        return sa_pairs