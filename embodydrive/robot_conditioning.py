"""Robot-specific conditioning layers for the Wan DiT backbone."""

import torch
import torch.nn as nn
from diffusers.models.controlnet import zero_module

from videox_fun.models.wan_transformer3d_unified_6v.embeddings import (
    sinusoidal_embedding_batchwise,
)


class RobotActionConditioningProj(nn.Module):
    """Project per-latent-step robot actions into Wan's six modulation vectors.

    The driving implementation assumes three action channels. DROID actions are
    seven-dimensional, so this module keeps all seven channels and normalizes
    them before the sinusoidal projection. Rotational action channels are
    scaled by pi so they are numerically comparable to the other channels.
    """

    def __init__(self, in_dim=7, freq_dim=256, dim=1536, patch_size=None, **kwargs):
        super().__init__()
        self.in_dim = int(in_dim)
        self.freq_dim = int(freq_dim)
        self.dim = int(dim)
        scale = torch.ones(self.in_dim)
        if self.in_dim >= 4:
            scale[3] = torch.pi
        self.register_buffer("action_scale", scale, persistent=False)
        self.action_embedding = nn.Sequential(
            nn.Linear(self.freq_dim * self.in_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.circular_embedding = nn.Linear(6, self.dim)
        self.action_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 6),
        )
        # The original adapter only modulated the six AdaLN vectors.  That
        # signal is easy for the strong visual-history prior to ignore.  A
        # separate per-token residual gives the action a direct path into the
        # video representation while keeping the old modulation path intact.
        self.token_projection = nn.Linear(self.dim, self.dim)
        nn.init.normal_(self.action_projection[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.action_projection[-1].bias)
        nn.init.zeros_(self.token_projection.weight)
        nn.init.zeros_(self.token_projection.bias)
        nn.init.zeros_(self.circular_embedding.weight)
        nn.init.zeros_(self.circular_embedding.bias)

    def forward(self, actions, **kwargs):
        if actions.ndim != 3 or actions.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected actions [B,F,{self.in_dim}], got {tuple(actions.shape)}"
            )
        circular = torch.cat(
            [torch.sin(torch.pi * actions[..., 3:6]),
             torch.cos(torch.pi * actions[..., 3:6])], dim=-1
        )
        scaled_actions = actions / self.action_scale.to(actions)
        a_emb = sinusoidal_embedding_batchwise(self.freq_dim, scaled_actions).flatten(-2).float()
        a = self.action_embedding(a_emb) + self.circular_embedding(circular.float())
        return self.action_projection(a).unflatten(-1, (6, self.dim))

    def token_residual(self, actions):
        if actions.ndim != 3 or actions.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected actions [B,F,{self.in_dim}], got {tuple(actions.shape)}"
            )
        circular = torch.cat(
            [torch.sin(torch.pi * actions[..., 3:6]),
             torch.cos(torch.pi * actions[..., 3:6])], dim=-1
        )
        scaled_actions = actions / self.action_scale.to(actions)
        a_emb = sinusoidal_embedding_batchwise(self.freq_dim, scaled_actions).flatten(-2).float()
        encoded = self.action_embedding(a_emb) + self.circular_embedding(circular.float())
        return self.token_projection(encoded).to(actions.dtype)


class RobotProprioConditioningProj(nn.Module):
    """Broadcast low-dimensional proprioception over latent spatial tokens."""

    def __init__(self, in_dim=14, dim=1536, patch_size=None, **kwargs):
        super().__init__()
        self.in_dim = int(in_dim)
        self.dim = int(dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        # A direct AdaLN path makes future robot state visible to every DiT
        # block.  The token path alone can be ignored by a strong visual
        # history prior, which is especially harmful during long rollouts.
        self.modulation_projection = nn.Linear(self.dim, self.dim * 6)
        self.circular_embedding = nn.Linear(6, self.dim)
        nn.init.zeros_(self.modulation_projection.weight)
        nn.init.zeros_(self.modulation_projection.bias)
        nn.init.zeros_(self.circular_embedding.weight)
        nn.init.zeros_(self.circular_embedding.bias)
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def encode(self, proprio):
        if proprio.ndim != 3 or proprio.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected proprio [B,F,{self.in_dim}], got {tuple(proprio.shape)}"
            )
        circular = torch.cat(
            [torch.sin(torch.pi * proprio[..., 10:13]),
             torch.cos(torch.pi * proprio[..., 10:13])], dim=-1
        )
        return self.encoder(proprio.float()) + self.circular_embedding(circular.float())

    def modulation(self, proprio):
        encoded = self.encode(proprio)
        return self.modulation_projection(encoded).unflatten(-1, (6, self.dim))

    def forward(self, x, proprio, **kwargs):
        encoded = self.encode(proprio).to(x[0].dtype)
        output = []
        for batch_index, tokens in enumerate(x):
            frames = encoded.shape[1]
            if tokens.shape[1] % frames != 0:
                raise ValueError(
                    f"Token count {tokens.shape[1]} is not divisible by proprio frames {frames}"
                )
            tokens_per_frame = tokens.shape[1] // frames
            bias = encoded[batch_index].unsqueeze(1).expand(-1, tokens_per_frame, -1)
            output.append(tokens + bias.reshape(1, tokens.shape[1], self.dim))
        return output


class RobotHistoryConditioningProj(nn.Module):
    """Inject clean history latents and a history mask into Wan token features."""

    def __init__(self, in_dim=17, dim=1536, patch_size=(1, 2, 2), **kwargs):
        super().__init__()
        self.in_dim = int(in_dim)
        self.dim = int(dim)
        self.conv = nn.Conv3d(
            self.in_dim,
            self.dim,
            kernel_size=tuple(patch_size),
            stride=tuple(patch_size),
        )
        self.proj = zero_module(nn.Linear(self.dim, self.dim))

    def forward(self, x, history, **kwargs):
        if history.ndim != 5 or history.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected history [B,{self.in_dim},F,H,W], got {tuple(history.shape)}"
            )
        encoded = self.proj(self.conv(history).flatten(2).transpose(1, 2))
        return [tokens + encoded[index : index + 1].to(tokens.dtype) for index, tokens in enumerate(x)]
