import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class Patching(nn.Module):
    """
    Divides input time series into patches.
    """
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, n_vars, seq_len]
        Returns:
            Patched tensor of shape [batch_size, n_vars, num_patches, patch_len]
        """
        batch_size, n_vars, seq_len = x.shape
        
        # Calculate the number of patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        
        # Unfold the tensor to create patches
        # x.unfold(dimension, size, step)
        # dimension 2 is seq_len
        patches = x.unfold(2, self.patch_len, self.stride)
        # patches shape: [batch_size, n_vars, num_patches, patch_len]
        
        return patches

class PatchEmbedding(nn.Module):
    """
    Embeds patches using a linear layer.
    """
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, n_vars, num_patches, patch_len]
        Returns:
            Embedded tensor of shape [batch_size, n_vars, num_patches, d_model]
        """
        # x shape: [batch_size, n_vars, num_patches, patch_len]
        # Output shape: [batch_size, n_vars, num_patches, d_model]
        x = self.projection(x)
        return x

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to patch embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000): # max_len is max_num_patches
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, n_vars, num_patches, d_model]
        Returns:
            Tensor with added positional encoding, shape [batch_size, n_vars, num_patches, d_model]
        """
        # x shape: [batch_size, n_vars, num_patches, d_model]
        # self.pe shape: [1, 1, max_num_patches, d_model]
        # We need to select the relevant part of pe for num_patches
        x = x + self.pe[:, :, :x.size(2), :] 
        return x

class LearnedPositionalEncoding(nn.Module):
    """
    Adds learned positional encoding to patch embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000): # max_len is max_num_patches
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, 1, max_len, d_model)) # learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, n_vars, num_patches, d_model]
        Returns:
            Tensor with added positional encoding, shape [batch_size, n_vars, num_patches, d_model]
        """
        # x shape: [batch_size, n_vars, num_patches, d_model]
        # self.pe shape: [1, 1, max_num_patches, d_model]
        x = x + self.pe[:, :, :x.size(2), :]
        return x


class PatchTST(nn.Module):
    """
    Patch Time Series Transformer model.
    """
    def __init__(self, c_in: int, seq_len: int, forecast_horizon: int, 
                 patch_len: int, stride: int, d_model: int, n_heads: int, 
                 n_encoder_layers: int, d_ff: int, dropout: float = 0.1, 
                 attn_dropout: float = 0.0, use_learned_pe: bool = True,
                 norm_first: bool = True): # norm_first=True is common in modern transformers
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # 1. Patching
        self.patching = Patching(patch_len, stride)
        
        # Calculate num_patches after patching
        # This is needed for positional encoding and transformer
        # (seq_len - patch_len) // stride + 1
        self.num_patches = (seq_len - patch_len) // stride + 1

        # 2. Patch Embedding
        self.embedding = PatchEmbedding(patch_len, d_model)

        # 3. Positional Encoding
        if use_learned_pe:
            self.pos_encoder = LearnedPositionalEncoding(d_model, max_len=self.num_patches)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)
        
        # Dropout after embedding and positional encoding
        self.embed_dropout = nn.Dropout(dropout)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff, 
            dropout=attn_dropout, # Dropout for MHA and FFN (different from overall dropout)
            activation=F.gelu, 
            batch_first=True, # Expected input: (N, S, E) N=batch_size, S=seq_len (num_patches), E=embed_dim (d_model)
            norm_first=norm_first # Pre-LN norm if True, Post-LN otherwise
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None # Final norm if norm_first
        )
        
        # 5. Forecasting Head
        # The input to the head will be the output of the transformer encoder, 
        # which has shape [batch_size * n_vars, num_patches, d_model].
        # We need to project this to [batch_size, n_vars, forecast_horizon].
        # A simple way is to flatten the num_patches and d_model dimensions.
        self.head_flatten = nn.Flatten(start_dim=2) # Flattens [num_patches, d_model]
        self.head_linear = nn.Linear(self.num_patches * d_model, forecast_horizon)
        
        # Optional overall dropout before the head
        self.head_dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, n_vars, seq_len]
        Returns:
            Output tensor of shape [batch_size, n_vars, forecast_horizon]
        """
        batch_size, n_vars, seq_len = x.shape

        # 1. Patching
        # x: [batch_size, n_vars, seq_len]
        x_patched = self.patching(x)
        # x_patched: [batch_size, n_vars, num_patches, patch_len]
        
        # 2. Patch Embedding
        x_embedded = self.embedding(x_patched)
        # x_embedded: [batch_size, n_vars, num_patches, d_model]

        # 3. Positional Encoding
        x_pos_encoded = self.pos_encoder(x_embedded)
        # x_pos_encoded: [batch_size, n_vars, num_patches, d_model]
        
        x_pos_encoded = self.embed_dropout(x_pos_encoded)

        # 4. Transformer Encoder (Channel Independent)
        # Reshape for channel-independent processing:
        # Each variable's patch sequence is treated as a separate item in a batch.
        # Input to TransformerEncoderLayer: (S, N, E) or (N, S, E) if batch_first=True
        # S = num_patches, N = batch_size * n_vars, E = d_model
        
        # Reshape from [batch_size, n_vars, num_patches, d_model] to [batch_size * n_vars, num_patches, d_model]
        x_transformed = x_pos_encoded.reshape(batch_size * n_vars, self.num_patches, self.d_model)
        
        # Transformer encoder expects (S, N, E) if batch_first=False, or (N, S, E) if batch_first=True
        # Our encoder_layer is batch_first=True, so input shape is correct.
        x_transformed = self.transformer_encoder(x_transformed)
        # x_transformed: [batch_size * n_vars, num_patches, d_model]
        
        # 5. Forecasting Head
        x_transformed = self.head_dropout(x_transformed)
        # Flatten the patch and model dimensions
        # x_transformed_flat: [batch_size * n_vars, num_patches * d_model]
        x_transformed_flat = self.head_flatten(x_transformed)
        
        # Linear projection to forecast_horizon
        # y_pred: [batch_size * n_vars, forecast_horizon]
        y_pred = self.head_linear(x_transformed_flat)
        
        # Reshape back to [batch_size, n_vars, forecast_horizon]
        y_pred = y_pred.reshape(batch_size, n_vars, self.forecast_horizon)
        
        return y_pred

if __name__ == '__main__':
    # Example Usage
    bs = 16
    nvars = 7 
    seq_len = 104 
    patch_len = 24
    stride = 2
    forecast_horizon = 60
    
    d_model = 16
    n_heads = 4
    n_encoder_layers = 3
    d_ff = 128
    dropout = 0.1
    attn_dropout = 0.0

    # Create dummy input tensor
    dummy_x = torch.randn(bs, nvars, seq_len)

    # Instantiate the model
    model = PatchTST(
        c_in=nvars, 
        seq_len=seq_len, 
        forecast_horizon=forecast_horizon,
        patch_len=patch_len, 
        stride=stride, 
        d_model=d_model, 
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers, 
        d_ff=d_ff, 
        dropout=dropout, 
        attn_dropout=attn_dropout,
        use_learned_pe=True
    )

    # Forward pass
    output = model(dummy_x)
    print(f"Input shape: {dummy_x.shape}")
    print(f"Output shape: {output.shape}") # Expected: [bs, nvars, forecast_horizon]

    # Test with sinusoidal positional encoding
    model_sin_pe = PatchTST(
        c_in=nvars, 
        seq_len=seq_len, 
        forecast_horizon=forecast_horizon,
        patch_len=patch_len, 
        stride=stride, 
        d_model=d_model, 
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers, 
        d_ff=d_ff, 
        dropout=dropout, 
        attn_dropout=attn_dropout,
        use_learned_pe=False
    )
    output_sin_pe = model_sin_pe(dummy_x)
    print(f"Output shape (Sinusoidal PE): {output_sin_pe.shape}")

    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        device = torch.device("cuda")
        model.to(device)
        dummy_x_gpu = dummy_x.to(device)
        output_gpu = model(dummy_x_gpu)
        print(f"Output shape (GPU): {output_gpu.shape}")
        assert output_gpu.device.type == 'cuda'
        print("GPU test successful.")
        model.to('cpu') # Move back to CPU for further potential tests
    else:
        print("\nCUDA not available. Skipping GPU test.")

    # Test patching module
    patching_layer = Patching(patch_len=16, stride=8)
    test_patch_input = torch.randn(2, 3, 64) # bs, nvars, seq_len
    patched_output = patching_layer(test_patch_input)
    # Expected num_patches = (64 - 16) // 8 + 1 = 48 // 8 + 1 = 6 + 1 = 7
    print(f"Patching test: Input shape {test_patch_input.shape}, Output shape {patched_output.shape}")
    assert patched_output.shape == (2, 3, 7, 16)

    # Test embedding module
    embedding_layer = PatchEmbedding(patch_len=16, d_model=32)
    test_embed_input = patched_output
    embedded_output = embedding_layer(test_embed_input)
    print(f"Embedding test: Input shape {test_embed_input.shape}, Output shape {embedded_output.shape}")
    assert embedded_output.shape == (2, 3, 7, 32)

    print("\nAll basic tests passed.")
