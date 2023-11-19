"""Perceiver model."""
import torch
from torch import nn


class Attention(nn.Module):
    """Multi-head attention in transformer using flash attention https://arxiv.org/abs/2205.14135."""
    def __init__(self, query_dim: int, value_dim: int, n_heads: int, dropout_p: float = 0.0):
        """Initialize.

        In this case we do not need the sequence length since the pytorch flash attention
        implementation is dynamic w.r.t. the sequence length.
        """
        super().__init__()
        assert query_dim % n_heads == 0
        assert value_dim % n_heads == 0

        self._n_heads = n_heads

        self._query_transf = nn.Linear(query_dim, query_dim, bias=False)
        self._query_head_dim = query_dim//n_heads
        self._query_dim = query_dim

        self._key_transf = nn.Linear(value_dim, query_dim, bias=False)
        # Same dimensionality as queries

        self._value_transf = nn.Linear(value_dim, query_dim, bias=False)
        #self._value_head_dim = value_dim//n_heads
        #self._value_dim = value_dim

        # The original transformer paper (https://arxiv.org/abs/1706.03762) also does a projection
        # as a part of the attention block.
        self._out_transf = nn.Linear(query_dim, query_dim, bias=False)

        self._dropout_p = dropout_p

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        """Forward pass."""
        # We need to transform the keys queries and values to be able to apply multiple attention
        # heads.
        B, NQ, _ = q.shape

        queries = self._query_transf(q)
        queries = queries.view(B, NQ, self._n_heads, self._query_head_dim).transpose(1, 2)

        _, NKV, _ = kv.shape
        keys = self._key_transf(kv)
        values = self._value_transf(kv)
        keys = keys.view(B, NKV, self._n_heads, self._query_head_dim).transpose(1, 2)
        values = values.view(B, NKV, self._n_heads, self._query_head_dim).transpose(1, 2)

        # Torch implementation of flash attention.
        # is_causal is crucial here, since otherwise the model will be able to look at tokens into
        # the future and cheat.
        out = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self._dropout_p if self.training else 0, is_causal=False)

        # Concatenate the channels from the multiple heads before applying the final linear
        # projection.
        out = out.transpose(1, 2).contiguous().view(B, NQ, self._query_dim)
        return self._out_transf(out)


def _ln(x: torch.Tensor):
    """Making layernorm easier to call."""
    return torch.nn.functional.layer_norm(x, [x.shape[-1]])



class TransformerBlock(nn.Module):
    """Implementation of a transformer block, which contains an attention layer."""
    def __init__(self, in_channels: int, n_heads: int, dropout_p: float = 0.0, mlp_channels=None):
        """Initialize the transformer block.

        After the attention layer a two-layer MLP is applied. Both the attention operation and the
        MLP have a parallel residual connection.

        Layernorm will be applied before attention and before the MLP that follows. This is not how
        it is done in the original transformer paper, but it makes the training procedure of the
        transformative much less sensitive. Using a normal optimizer without a specially finetuned
        learning rate schedule will converge with this architecture. Achieving congergence with the
        original layernorm placement is much harder. See https://arxiv.org/abs/2002.04745 for more
        information.

        Args:
            ...mostly obvious...
            mlp_channels: This determines the number of output channels of the first projection in
                          the two layer MLP. By default this is 4 times the dimensionality of the
                          internal state of the attention layers since this is what GPT uses. My
                          speculation is that the motivation for this is that making the
                          dimensionality higher before applying ReLU prevents information loss.
                          This mechanism is explained in a lot of detail in the paper introducing
                          MobileNetV2: https://arxiv.org/abs/1801.04381.
        """
        super().__init__()
        if mlp_channels is None:
            mlp_channels = in_channels*4
        self._att = Attention(query_dim=in_channels, value_dim=in_channels, n_heads=n_heads, dropout_p=dropout_p)
        self._linear_1 = nn.Linear(in_channels, mlp_channels, bias=True)
        self._linear_2 = nn.Linear(mlp_channels, in_channels, bias=True)

        def _do(x: torch.Tensor):
            """Dropout."""
            return torch.nn.functional.dropout(x, p=dropout_p, training=self.training)
        self._do = _do

    def forward(self, x):
        """Forward pass."""
        normed = _ln(x)
        x = x + self._att(normed, normed)

        # TODO Does it matter to apply GeLU here instead of ReLU like in GPT papers?
        # I expect ReLU to be faster, but should test it explicitly.
        x = x + self._do(self._linear_2(torch.nn.functional.relu(self._linear_1(_ln(x)))))
        return x


class CrossAttentionBlock(nn.Module):
    """Implementation of a transformer block, which contains an attention layer."""
    def __init__(self, latent_channels: int, data_channels: int, n_heads: int, dropout_p: float = 0.0, mlp_channels=None):
        """Initialize the transformer block.

        After the attention layer a two-layer MLP is applied. Both the attention operation and the
        MLP have a parallel residual connection.

        Layernorm will be applied before attention and before the MLP that follows. This is not how
        it is done in the original transformer paper, but it makes the training procedure of the
        transformative much less sensitive. Using a normal optimizer without a specially finetuned
        learning rate schedule will converge with this architecture. Achieving congergence with the
        original layernorm placement is much harder. See https://arxiv.org/abs/2002.04745 for more
        information.

        Args:
            ...mostly obvious...
            mlp_channels: This determines the number of output channels of the first projection in
                          the two layer MLP. By default this is 4 times the dimensionality of the
                          internal state of the attention layers since this is what GPT uses. My
                          speculation is that the motivation for this is that making the
                          dimensionality higher before applying ReLU prevents information loss.
                          This mechanism is explained in a lot of detail in the paper introducing
                          MobileNetV2: https://arxiv.org/abs/1801.04381.
        """
        super().__init__()
        if mlp_channels is None:
            mlp_channels = latent_channels*4
        self._att = Attention(query_dim=latent_channels, value_dim=data_channels, n_heads=n_heads, dropout_p=dropout_p)
        self._linear_1 = nn.Linear(latent_channels, mlp_channels, bias=True)
        self._linear_2 = nn.Linear(mlp_channels, latent_channels, bias=True)

        def _do(x: torch.Tensor):
            """Dropout."""
            return torch.nn.functional.dropout(x, p=dropout_p, training=self.training)
        self._do = _do

    def forward(self, latent, data):
        """Forward pass."""
        normed = _ln(latent)
        latent = latent + self._att(normed, data)

        # TODO Does it matter to apply GeLU here instead of ReLU like in GPT papers?
        # I expect ReLU to be faster, but should test it explicitly.
        latent = latent + self._do(self._linear_2(torch.nn.functional.relu(self._linear_1(_ln(latent)))))
        return latent




class Perceiver(nn.Module):

    def __init__(self, in_channels, n_latent, dim_latent, n_heads_cross = 1, n_heads_self = 8, n_self_per_cross=6):
        super().__init__()
        self._latent = nn.Parameter(torch.randn(n_latent, dim_latent))
        self._cross_attend_1 = CrossAttentionBlock(latent_channels=dim_latent, data_channels=in_channels, n_heads=n_heads_cross, dropout_p=0)
        self._transformer_1 = nn.Sequential(*[TransformerBlock(in_channels=dim_latent, n_heads=n_heads_self, dropout_p=0) for _ in range(6)])
        self._cross_attend_2 = CrossAttentionBlock(latent_channels=dim_latent, data_channels=in_channels, n_heads=n_heads_cross, dropout_p=0)
        self._transformer_2 = nn.Sequential(*[TransformerBlock(in_channels=dim_latent, n_heads=n_heads_self, dropout_p=0) for _ in range(6)])


    def forward(self, byte_array):
        batch_size = byte_array.shape[0]
        latent = self._latent.repeat(batch_size, 1, 1)
        x = self._cross_attend_1(latent, byte_array)
        x = self._transformer_1(x)
        for _ in range(7):
            x = self._cross_attend_2(x, byte_array)
            x = self._transformer_2(x)
        return x

class PerceiverClassifier(nn.Module):

    def __init__(self, perceiver_model, n_classes):
        super().__init__()
        self._perceiver = perceiver_model
        self._class_proj = nn.Linear(perceiver_model._latent.shape[-1], n_classes, bias=True)

    def forward(self, x):
        x = self._perceiver(x)
        x = torch.mean(x, dim=1)
        x = self._class_proj(x)
        return torch.nn.functional.softmax(x, dim=-1)


if __name__ == "__main__":
    q = torch.rand(2, 100, 10)
    kv = torch.rand(2, 100, 10)

    p = Perceiver(10, 100, 128, 1, 8, 6)
    pcls = PerceiverClassifier(p, 10)
    print(pcls(kv).shape)
