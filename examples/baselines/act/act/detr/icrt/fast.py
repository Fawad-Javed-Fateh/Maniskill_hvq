import torch
import torch.nn as nn
import clip
from transformers import AutoProcessor

class FAST_Tokenizer(nn.Module):
    def __init__(self, action_output_shape):
        super().__init__()
        
        # Load action tokenizer
        self.action_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        
        # Load CLIP model
        self.clip_model, _ = clip.load("ViT-B/32")
        
        # Define action network
        self.action_network = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, action_output_shape),
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, -1)
        aggregated_vector_list = []
        
        for idx in range(batch_size):
            prompt_actions_idx = x[idx]
            tokens = self.action_tokenizer(prompt_actions_idx.cpu().numpy())
            clip_tokens = clip.tokenize(list(map(str, tokens[0]))).to(x.device)
            
            with torch.no_grad():
                latent_vector = self.clip_model.encode_text(clip_tokens)
            
            # Normalize latent vector
            latent_vector = latent_vector / latent_vector.norm(dim=-1, keepdim=True)
            D, dim = latent_vector.shape
            
            if D >= seq_len:
                indices = torch.linspace(0, D - 1, steps=seq_len).long()
                aggregated_vector = latent_vector[indices]
            else:
                aggregated_vector = torch.zeros(seq_len, dim, device=latent_vector.device)
                aggregated_vector[:D] = latent_vector
            
            aggregated_vector_list.append(aggregated_vector)
        
        context_actions = torch.cat(aggregated_vector_list, dim=0)
        context_actions = self.action_network(context_actions).squeeze(0)
        
        return context_actions


if __name__ == "__main__":
    fast_tokenizer = FAST_Tokenizer(action_output_shape=256)

    x = torch.rand(32, 256, 256)
    out = fast_tokenizer(x)
    print(out.data.shape)