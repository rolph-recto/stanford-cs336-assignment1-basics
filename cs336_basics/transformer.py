import einops
import math
import torch

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        std: float = math.sqrt(2 / (out_features + in_features))
        weight_data: torch.Tensor = \
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (out_features, in_features),
                    device=device, dtype=dtype, requires_grad=True
                ),
                mean = 0,
                std = std,
                a = -3.0 * std,
                b = 3.0 * std
            )

        self.weights: torch.nn.Parameter = torch.nn.Parameter(data=weight_data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            x, self.weights,
            '... d_in, d_out d_in -> ... d_out'
        )

class Embedding(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        embedding_weights: torch.Tensor = \
            torch.nn.init.trunc_normal_(
                torch.empty(vocab_size, d_model, device=device, dtype=dtype),
                mean = 0.0,
                std = 1.0,
                a = -3.0,
                b = 3.0
            )

        self.weights = torch.nn.Parameter(data=embedding_weights)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size: int = token_ids.size(0)
        seq_length: int = token_ids.size(1)
        return einops.rearrange(
            self.weights.index_select(0, token_ids.reshape(-1)),
            "... (batch_size seq_length) d_model -> ... batch_size seq_length d_model",
            batch_size=batch_size, seq_length=seq_length
        )
        
class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None =None,
        dtype: torch.dtype | None =None
    ):
        super().__init__()

        self.weights: torch.nn.Parameter =\
            torch.nn.Parameter(
                data=torch.ones(d_model, device=device, dtype=dtype)
            )
        self.eps = eps

    # x: (batch_size, sequence_length, d_model)
    # returns tensor of shape (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast input to float32 to prevent overflow
        x_dtype: torch.dtype = x.dtype
        x = x.to(torch.float32)

        # rms: (batch_size, seq_length, 1)
        rms: torch.Tensor = \
            einops.reduce(x * x,'... batch seq d_model -> ... batch seq', 'mean') \
                .add(self.eps) \
                .sqrt() \
                .unsqueeze(-1)

        result: torch.Tensor = x.div(rms).mul(self.weights)
        return result.to(x_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        std: float = math.sqrt(2 / (d_model + d_model))
        init: dict = {
            "mean": 0.0,
            "std": math.sqrt(2 / (d_model + d_model)),
            "a": -3.0 * std,
            "b": 3.0 * std
        }
        w1_weights: torch.Tensor = \
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (d_ff, d_model),
                    device=device, dtype=dtype, requires_grad=True
                ),
                **init
            )
        self.w1 = torch.nn.Parameter(data=w1_weights)

        w2_weights: torch.Tensor = \
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (d_model, d_ff),
                    device=device, dtype=dtype, requires_grad=True
                ),
                **init
            )
        self.w2 = torch.nn.Parameter(data=w2_weights)

        w3_weights: torch.Tensor = \
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (d_ff, d_model),
                    device=device, dtype=dtype, requires_grad=True
                ),
                **init
            )
        self.w3 = torch.nn.Parameter(data=w3_weights)
    
    # x: (..., d_model)
    # returns (..., d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out: torch.Tensor = \
            einops.einsum(self.w1, x, 'd_ff d_model, ... d_model -> ... d_ff')

        silu_out: torch.Tensor = w1_out * torch.sigmoid(w1_out)

        w3_out: torch.Tensor = \
            einops.einsum(self.w3, x, 'd_ff d_model, ... d_model -> ... d_ff')

        gated_out: torch.Tensor = silu_out * w3_out
        out: torch.Tensor = \
            einops.einsum(self.w2, gated_out, 'd_model d_ff, ... d_ff -> ... d_model')

        return out

class RoPE(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        k_indices = torch.arange(0, d_k, 2, device=device, dtype=dtype).float()
        freqs = 1.0 / (theta ** (k_indices / d_k))
        
        # 2. Compute the outer product of positions and frequencies
        pos = torch.arange(max_seq_len, device=device, dtype=dtype).float()
        angles = torch.outer(pos, freqs) # Shape: (max_seq_len, d_k // 2)

        # 3. Create the 2x2 rotation matrices
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Build matrices: [[cos, -sin], [sin, cos]]
        rot = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1)
        ], dim=-2) # Shape: (max_seq_len, d_k // 2, 2, 2)

        self.register_buffer("rot", rot, persistent=False)

    # x: (..., seq_len, d_k)
    # token_positions: (..., seq_len)
    # output: (..., seq_len, d_k)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # d_k: int = x.size(-1)

        tokens_rot: torch.Tensor = self.rot[token_positions] # type: ignore

        x_batched: torch.Tensor = \
            einops.rearrange(x, '... seq_len (k w) -> ... seq_len k w', w=2)

        result_batched: torch.Tensor = \
            einops.einsum(
                tokens_rot, x_batched,
                '... seq_len k h w, ... seq_len k w -> ... seq_len k h'
            )
        
        # flatten vectors back into original shape
        return einops.rearrange(
            result_batched,
            '... seq_len k h -> ... seq_len (k h)'
        )

def softmax(x: torch.Tensor, d: int) -> torch.Tensor:
    d_max = x.amax(d, keepdim=True)
    x2: torch.Tensor = x - d_max
    x2_exp: torch.Tensor = x2.exp()
    total: torch.Tensor = x2_exp.sum(dim=d, keepdim=True)
    return x2_exp / total

# Q: (... seq d_k)
# K: (... seq d_k)
# V: (... seq d_v)
# mask: (... seq ks)
# output: (... seq d_v)
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
):
    d_k: int = K.size(-1)
    qk: torch.Tensor = \
        einops.einsum(
            Q, K,
            '... qs d_k, ... ks d_k -> ... qs ks'
        )

    qk /= math.sqrt(d_k)

    if mask is not None:
        # is mask is false, that means we SHOULD fill with -inf
        qk.masked_fill_(mask == False, -torch.inf) 

    score: torch.Tensor = softmax(qk, -1)
    return einops.einsum(
        score, V,
        '... qs ks, ... ks d_v -> ... qs d_v'
    )

class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None, # for RoPE
        theta: float | None = None, # for RoPE
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.heads: int = num_heads 
        self.d_e: int = d_model // num_heads

        # concat WQ, WK, and WV together so forward pass is efficient
        std: float = math.sqrt(2 / (self.heads * self.d_e + d_model))
        weights = torch.nn.init.trunc_normal_(
            torch.empty(3 * num_heads * self.d_e, d_model, device=device, dtype=dtype, requires_grad=True),
            mean = 0, std = std, a = -3.0 * std, b = 3.0 * std
        )
        self.WQKV: torch.Tensor = torch.nn.Parameter(data=weights, requires_grad=True)

        self.WO = \
            torch.nn.Parameter(data=torch.nn.init.trunc_normal_(
                torch.empty(d_model, num_heads * self.d_e, device=device, dtype=dtype, requires_grad=True),
                mean = 0, std = std, a = -3.0 * std, b = 3.0 * std
            ))

        self.rope: torch.nn.Module | None = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_e, max_seq_len, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_length: int = x.size(-2)

        # batch the Q, K, V linear layers computations together for efficiency
        QKV: torch.Tensor = einops.einsum(
            self.WQKV, x,
            'qkv__heads__d_e d_model, ... seq d_model -> ... seq qkv__heads__d_e',
        )

        Q, K, V = \
            einops.rearrange(
                QKV,
                '... seq (qkv heads d_e) -> qkv ... heads seq d_e',
                qkv=3, heads=self.heads
            )

        if self.rope is not None:
            assert token_positions is not None
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        seq_range = torch.arange(seq_length)
        causal_mask: torch.Tensor = seq_range.unsqueeze(0) <= seq_range.unsqueeze(1)

        # heads: (... heads seq d_e)
        heads = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        concat_heads = einops.rearrange(heads, '... heads seq d_e -> ... seq (heads d_e)')

        return einops.einsum(
            self.WO, concat_heads,
            "d_model heads__d_e, ... seq heads__d_e -> ... seq d_model"
        )

# Pre-norm causal multi-head self-attention block followed by a FFN layer
# Note that this is the standard attention block today (2025),
# but is different from Vaswani 2017, which has a "post-norm" attention block
class PreNormTransformerBlock(torch.nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.prenorm_attn = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = \
            CausalMultiHeadSelfAttention(
                d_model, num_heads,
                max_seq_len=max_seq_len, theta=theta,
                device=device, dtype=dtype
            )

        self.prenorm_ffn = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length: int = x.size(-2)
        token_positions: torch.Tensor = \
            torch.arange(seq_length).expand(x.shape[:-1])

        x += self.attention(self.prenorm_attn(x), token_positions)
        x += self.ffn(self.prenorm_ffn(x))
        return x

class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            Embedding(vocab_size, d_model, device=device, dtype=dtype),
            torch.nn.Sequential(*[
                PreNormTransformerBlock(
                    d_model, num_heads, d_ff,
                    max_seq_len=context_length, theta=theta
                ) for _ in range(num_layers)
            ]),
            RMSNorm(d_model),
            Linear(d_model, vocab_size),
        ])

    # this returns raw logits, not normalized to a distribution by softmax
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x