import torch
import torch.nn as nn
from transformers import AutoModel


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()
        self.config = config

        self.random_range = 0.5
        init_value = torch.FloatTensor(1, self.config.model.llm_embedding_channels).uniform_(-self.random_range, self.random_range)
        self.eos_embedding = torch.nn.parameter.Parameter(init_value)

        self.encoder = AutoModel.from_pretrained(self.config.model.audio_encoder.type)

        self.downsample_method = self.config.model.audio_encoder.downsample_method
        self.downsample_factor = self.config.model.audio_encoder.downsample_factor

        if self.downsample_method == "pool":
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.config.model.audio_encoder.pooling.kernel_size,
                stride=self.config.model.audio_encoder.pooling.stride,
            )
            self.embed_projection = nn.Linear(
                self.encoder.config.hidden_size,
                self.config.model.llm_embedding_channels,
            )
        else:
            raise Exception("Invalid downsampling method for audio encoder.")

    def update_embed_projection(self, update_dim=4096):
        print(f"Updating embed projection from {self.config.model.llm_embedding_channels} to {update_dim}")
        self.embed_projection = nn.Linear(
            self.encoder.config.hidden_size,
            4096,
        )    
          
    def forward(self, audio_input, audio_mask = None, ctc_pool_ranges=None):
        encoder_out = self.encoder(audio_input, audio_mask).last_hidden_state  # (B, N, 1024)

        if self.downsample_method == "pool":
            # (B, N, 1024) -> (B, N/4, 1024) -> (B, N/4, 3072)
            audio_embeds = self.pooling_layer(
                encoder_out.transpose(1, 2)
            ).transpose(1, 2)
        else:
            raise Exception("Invalid downsampling method for audio encoder.")

        audio_embeds = self.embed_projection(audio_embeds)
        return audio_embeds
