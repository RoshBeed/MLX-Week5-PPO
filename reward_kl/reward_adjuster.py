from typing import List, Dict

class RewardAdjuster:
    """Adjusts rewards with KL penalty"""
    
    @staticmethod
    def adjust_rewards_with_kl(sa_pairs: List[Dict], rewards: List[float], 
                             kl_coef: float = 0.1) -> List[Dict]:
        """Adjust rewards with KL penalty"""
        for pair in sa_pairs:
            reward = rewards[pair['prompt_idx']]
            kl = pair['kl_div']
            pair['adjusted_reward'] = reward - kl_coef * kl
        return sa_pairs 