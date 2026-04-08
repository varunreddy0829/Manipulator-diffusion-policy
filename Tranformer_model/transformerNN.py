import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """Creates the 'Timestep t -> sin/cos waves' embedding (Step 2)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerDiffusionPolicy(nn.Module):
    def __init__(self, 
                 action_dim=9, 
                 state_dim=18, 
                 embed_dim=256, 
                 num_layers=6, 
                 nhead=8, 
                 max_action_steps=16):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_action_steps = max_action_steps

        # ==========================================
        # PHASE 1: Observation & Context Encoders
        # ==========================================
        
        # 1. Image Token: (Assuming LeRobot's ResNet outputs a flat 512d vector, we project to 256)
        self.image_proj = nn.Linear(512, embed_dim) 
        
        # 2. State Token: Projecting 18d -> 256d
        self.state_proj = nn.Linear(state_dim, embed_dim)
        
        # 3. Time Token: Sinusoidal -> MLP -> 256d
        self.time_emb = SinusoidalPosEmb(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # ==========================================
        # PHASE 2: Action Encoders (The Noisy Blueprint)
        # ==========================================
        
        # Projecting 9d noisy actions -> 256d
        self.action_proj = nn.Linear(action_dim, embed_dim)
        
        # Positional encoding so the model knows the sequence order (Step 1, 2... 16)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_action_steps, embed_dim))

        # ==========================================
        # PHASE 3: The Transformer Decoder
        # ==========================================
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ==========================================
        # PHASE 4: Output Head
        # ==========================================
        
        # Compressing back from 256d -> 9d predicted noise
        self.noise_pred_head = nn.Linear(embed_dim, action_dim)

    def forward(self, noisy_actions, timestep, image_features, robot_state):
        """
        noisy_actions: (Batch, 16, 9)
        timestep: (Batch,)
        image_features: (Batch, 512) - Extracted by frozen ResNet outside this module
        robot_state: (Batch, 18)
        """
        B = noisy_actions.shape[0]

        # --- Encode Context (Keys & Values) ---
        img_token = self.image_proj(image_features).unsqueeze(1)    # (B, 1, 256)
        state_token = self.state_proj(robot_state).unsqueeze(1)     # (B, 1, 256)
        
        t_emb = self.time_emb(timestep)
        time_token = self.time_mlp(t_emb).unsqueeze(1)              # (B, 1, 256)

        # Combine into read-only context sequence
        # Shape: (B, 3, 256)
        context_sequence = torch.cat([img_token, state_token, time_token], dim=1)

        # --- Encode Actions (Queries) ---
        # Shape: (B, 16, 256)
        action_tokens = self.action_proj(noisy_actions) 
        
        # Add positional embedding
        action_tokens = action_tokens + self.pos_embedding[:, :self.max_action_steps, :]

        # --- Run Transformer ---
        # Note: PyTorch TransformerDecoder takes (Target, Memory) which translates to (Queries, Context)
        # Action tokens attend to each other (Self-Attention) and to Context (Cross-Attention)
        processed_tokens = self.transformer(tgt=action_tokens, memory=context_sequence)

        # --- Predict Noise ---
        # Shape: (B, 16, 9)
        predicted_noise = self.noise_pred_head(processed_tokens)

        return predicted_noise