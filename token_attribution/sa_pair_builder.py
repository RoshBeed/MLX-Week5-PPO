from typing import List, Dict

class SAPairBuilder:
    """Builds (state, action) pairs from tokens"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def build_sa_pairs(self, prompt: str, gen_tokens: List[str], 
                      gen_token_ids: List[int], policy_logprobs: List[float],
                      ref_logprobs: List[float], prompt_idx: int) -> List[Dict]:
        """Build (s, a) pairs for a single prompt"""
        sa_pairs = []
        current_state = prompt
        
        for step, (token, token_id, pol_logprob, ref_logprob) in enumerate(
            zip(gen_tokens, gen_token_ids, policy_logprobs, ref_logprobs)
        ):
            sa_pair = {
                'state_text': current_state,
                'action_token': token,
                'action_token_id': token_id,
                'logprob': pol_logprob,
                'ref_logprob': ref_logprob,
                'prompt_idx': prompt_idx,
                'step': step
            }
            sa_pairs.append(sa_pair)
            
            # Update state for next step
            current_state += token
        
        return sa_pairs 