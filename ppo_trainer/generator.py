import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

class SequenceGenerator:
    """Handles sequence generation and extraction of (s, a) pairs"""
    
    def __init__(self, policy_model, sft_model, tokenizer, config, device):
        self.policy_model = policy_model
        self.sft_model = sft_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
    
    def generate_sequences(self, prompts: List[str]) -> Tuple[List[Dict], List[str]]:
        """Generate sequences and extract (s, a) pairs"""
        # Generate sequences
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        sequences = outputs.sequences
        scores = outputs.scores
        batch_size = sequences.shape[0]
        prompt_len = inputs["input_ids"].shape[1]
        
        # Extract (s, a) pairs
        all_sa_pairs = []
        generated_texts = []
        
        for i in range(batch_size):
            gen_tokens = []
            policy_logprobs = []
            ref_logprobs = []
            
            for t, score_t in enumerate(scores):
                # State: prompt + previously generated tokens
                state_ids = sequences[i, :prompt_len + t]
                state_text = self.tokenizer.decode(
                    state_ids, skip_special_tokens=True
                )
                
                # Action: next token
                action_id = sequences[i, prompt_len + t].item()
                action_token = self.tokenizer.decode([action_id])
                gen_tokens.append(action_token)
                
                # Policy logprob
                log_probs = F.log_softmax(score_t[i], dim=-1)
                logprob = log_probs[action_id].item()
                policy_logprobs.append(logprob)
                
                # Reference logprob
                with torch.no_grad():
                    ref_inputs = state_ids.unsqueeze(0).to(self.device)
                    ref_outputs = self.sft_model(ref_inputs)
                    ref_logits = ref_outputs.logits[0, -1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = ref_log_probs[action_id].item()
                    ref_logprobs.append(ref_logprob)
                
                all_sa_pairs.append({
                    "state_text": state_text,
                    "action_token": action_token,
                    "action_token_id": action_id,
                    "logprob": logprob,
                    "ref_logprob": ref_logprob,
                    "prompt_idx": i,
                    "step": t,
                })
            
            generated_texts.append("".join(gen_tokens))
        
        return all_sa_pairs, generated_texts 