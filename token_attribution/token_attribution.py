from typing import List, Dict
from .extractor import TokenExtractor
from .sa_pair_builder import SAPairBuilder

class TokenAttribution:
    """Divides sequences into (state, action) pairs"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.extractor = TokenExtractor(tokenizer)
        self.builder = SAPairBuilder(tokenizer)
    
    def extract_sa_pairs(self, prompts: List[str], generated_texts: List[str], 
                        policy_logprobs: List[List[float]], 
                        ref_logprobs: List[List[float]]) -> List[Dict]:
        """Extract (s, a) pairs from generated sequences"""
        all_sa_pairs = []
        
        for prompt_idx, (prompt, gen_text, pol_logprobs, ref_logprobs) in enumerate(
            zip(prompts, generated_texts, policy_logprobs, ref_logprobs)
        ):
            # Extract tokens
            gen_tokens, gen_token_ids = self.extractor.extract_tokens(gen_text)
            
            # Build SA pairs
            sa_pairs = self.builder.build_sa_pairs(
                prompt, gen_tokens, gen_token_ids, pol_logprobs, ref_logprobs, prompt_idx
            )
            all_sa_pairs.extend(sa_pairs)
        
        return all_sa_pairs 