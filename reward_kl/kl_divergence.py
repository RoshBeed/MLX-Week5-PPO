from typing import List, Dict

class KLDivergence:
    """Computes KL divergence between policy and reference model"""
    
    @staticmethod
    def compute_kl_per_token(logprobs_policy: List[float], 
                           logprobs_ref: List[float]) -> List[float]:
        """Compute per-token KL divergence"""
        kl_per_token = [pol - ref for pol, ref in zip(logprobs_policy, logprobs_ref)]
        return kl_per_token
    
    @staticmethod
    def add_kl_to_sa_pairs(sa_pairs: List[Dict]) -> List[Dict]:
        """Add KL divergence to each (s, a) pair"""
        for pair in sa_pairs:
            logprob_policy = pair["logprob"]
            logprob_ref = pair["ref_logprob"]
            pair["kl_div"] = logprob_policy - logprob_ref
        return sa_pairs 