import pytest
import torch

from model.Transformer import (
    DotProductAttention,
    MultiHeadAttention,
    FinalMultiHeadAttention,
    FFN,
    EncoderBlock,
    FinalEncoderBlock,
    Encoder,
    PositionalEncoding,
)

EMBED_DIM = 512
HEADS = 8
HEAD_DIM = EMBED_DIM // HEADS


# ---------------------------------------------------------------------------
# DotProductAttention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens,d_k", [(10, 64), (1, 64), (7, 64), (20, 64)])
def test_dot_product_attention_shape(tokens, d_k):
    attention = DotProductAttention()
    Q, K, V = torch.randn(tokens, d_k), torch.randn(tokens, d_k), torch.randn(tokens, d_k)

    out = attention(Q, K, V)

    assert out.shape == (tokens, d_k), f"expected {(tokens, d_k)}, got {tuple(out.shape)}"


def test_dot_product_attention_query_len_can_differ_from_key_value_len():
    """Q can have a different token count than K/V (e.g. cross-attention) -
    output should follow Q's token count, not K/V's."""
    attention = DotProductAttention()
    q_tokens, kv_tokens, d_k = 4, 9, 64
    Q = torch.randn(q_tokens, d_k)
    K, V = torch.randn(kv_tokens, d_k), torch.randn(kv_tokens, d_k)

    out = attention(Q, K, V)

    assert out.shape == (q_tokens, d_k)


# ---------------------------------------------------------------------------
# MultiHeadAttention.get_projections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [1, 5, 10])
def test_get_projections_preserves_shape(tokens):
    mha = MultiHeadAttention()
    Q = torch.randn(tokens, HEAD_DIM)
    K, V = torch.randn(tokens, HEAD_DIM), torch.randn(tokens, HEAD_DIM)

    q_proj, k_proj, v_proj = mha.get_projections(Q, K, V)

    assert q_proj.shape == (tokens, HEAD_DIM)
    assert k_proj.shape == (tokens, HEAD_DIM)
    assert v_proj.shape == (tokens, HEAD_DIM)


# ---------------------------------------------------------------------------
# MultiHeadAttention (full forward, self-attention)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [1, 5, 10, 17])
def test_multi_head_attention_output_shape(tokens):
    """Regression test: output shape must track the actual number of tokens
    passed in, not a hardcoded sequence length."""
    mha = MultiHeadAttention()
    X = torch.randn(tokens, EMBED_DIM)

    out = mha(X)

    assert out.shape == (tokens, EMBED_DIM), f"expected {(tokens, EMBED_DIM)}, got {tuple(out.shape)}"


# ---------------------------------------------------------------------------
# FinalMultiHeadAttention - also hands back K, V (per head) for the decoder's
# cross-attention to consume
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [1, 5, 10])
def test_final_multi_head_attention_returns_output_and_kv_shapes(tokens):
    mha = FinalMultiHeadAttention()
    X = torch.randn(tokens, EMBED_DIM)

    out, K, V = mha(X)

    assert out.shape == (tokens, EMBED_DIM), f"expected {(tokens, EMBED_DIM)}, got {tuple(out.shape)}"
    assert K.shape == (HEADS, tokens, HEAD_DIM), f"expected {(HEADS, tokens, HEAD_DIM)}, got {tuple(K.shape)}"
    assert V.shape == (HEADS, tokens, HEAD_DIM), f"expected {(HEADS, tokens, HEAD_DIM)}, got {tuple(V.shape)}"


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [1, 5, 10])
def test_ffn_default_dims(tokens):
    ffn = FFN()
    out = ffn(torch.randn(tokens, 512))

    assert out.shape == (tokens, 512)


def test_ffn_custom_dims_change_output_width():
    ffn = FFN(input_dim=512, hidden_dim=2048, out_dim=256)
    out = ffn(torch.randn(3, 512))

    assert out.shape == (3, 256)


# ---------------------------------------------------------------------------
# EncoderBlock
# ---------------------------------------------------------------------------

def test_add_and_norm_preserves_shape():
    block = EncoderBlock(embedding_dim=EMBED_DIM)
    tokens = 8
    initial, delta = torch.randn(tokens, EMBED_DIM), torch.randn(tokens, EMBED_DIM)

    out = block.AddAndNorm(initial, delta)

    assert out.shape == (tokens, EMBED_DIM)


def test_add_and_norm_rejects_mismatched_shapes():
    """AddAndNorm is an elementwise residual add - mismatched shapes should
    fail loudly, not silently broadcast into something unintended."""
    block = EncoderBlock(embedding_dim=EMBED_DIM)
    initial = torch.randn(8, EMBED_DIM)
    delta = torch.randn(5, EMBED_DIM)

    with pytest.raises(RuntimeError):
        block.AddAndNorm(initial, delta)


@pytest.mark.parametrize("tokens", [1, 5, 10])
def test_encoder_block_output_shape(tokens):
    block = EncoderBlock(embedding_dim=EMBED_DIM)
    X = torch.randn(tokens, EMBED_DIM)

    out = block(X)

    assert out.shape == (tokens, EMBED_DIM)


# ---------------------------------------------------------------------------
# FinalEncoderBlock - unlike EncoderBlock, forward returns (X, K, V): the
# transformed tensor plus the K, V the decoder's cross-attention will use.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [1, 5, 10])
def test_final_encoder_block_output_shape(tokens):
    block = FinalEncoderBlock(embedding_dim=EMBED_DIM)
    X = torch.randn(tokens, EMBED_DIM)

    out, K, V = block(X)

    assert out.shape == (tokens, EMBED_DIM), f"expected {(tokens, EMBED_DIM)}, got {tuple(out.shape)}"
    assert K.shape == (HEADS, tokens, HEAD_DIM), f"expected {(HEADS, tokens, HEAD_DIM)}, got {tuple(K.shape)}"
    assert V.shape == (HEADS, tokens, HEAD_DIM), f"expected {(HEADS, tokens, HEAD_DIM)}, got {tuple(V.shape)}"


# ---------------------------------------------------------------------------
# Encoder - full stack. forward returns (X, K, V): the stack's final output
# plus the K, V handed off by the final block for the decoder to consume.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [1, 5, 10])
def test_encoder_output_shape(tokens):
    encoder = Encoder(n_blocks=3)
    X = torch.randn(tokens, EMBED_DIM)

    out, K, V = encoder(X)

    assert out.shape == (tokens, EMBED_DIM), f"expected {(tokens, EMBED_DIM)}, got {tuple(out.shape)}"
    assert K.shape == (HEADS, tokens, HEAD_DIM), f"expected {(HEADS, tokens, HEAD_DIM)}, got {tuple(K.shape)}"
    assert V.shape == (HEADS, tokens, HEAD_DIM), f"expected {(HEADS, tokens, HEAD_DIM)}, got {tuple(V.shape)}"


def test_encoder_has_n_blocks_minus_one_in_sequential():
    """The final block is called separately in Encoder.forward (so it can
    return its (X, K, V) tuple without nn.Sequential trying to chain that
    tuple into the next module) - only the other n_blocks - 1 blocks live
    inside self.EncoderBlocks."""
    encoder = Encoder(n_blocks=3)

    assert len(list(encoder.EncoderBlocks)) == 2


def test_encoder_blocks_have_independent_weights():
    """Each block should learn its own weights. If two blocks are literally
    the same Python object, they'll share one set of weights instead of
    being independently trainable layers."""
    encoder = Encoder(n_blocks=3)
    blocks = list(encoder.EncoderBlocks)

    assert blocks[0] is not blocks[1], "block 0 and block 1 are the same object - they share one set of weights"


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------

def test_add_positional_encodings_preserves_shape():
    pe = PositionalEncoding()
    tokens, dim = 6, pe.d_model
    embeddings = [[0.0] * dim for _ in range(tokens)]

    out = pe.AddPositionalEncodings(embeddings, pos=0)

    assert len(out) == tokens
    assert all(len(word) == dim for word in out)


# ---------------------------------------------------------------------------
# Still under construction - documented but not asserted yet
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="CrossAttention.forward is still a stub (`pass`, and its signature is missing "
                          "`self`) - write the real assertion once it takes shape.")
def test_cross_attention_output_shape():
    pass


@pytest.mark.skip(reason="MaskedMultiHeadAttention's constructor still expects the pre-refactor "
                          "MultiHeadAttention(W, Wo) signature, and forward references "
                          "self.MultiHeadAttention which isn't the attribute name it assigned "
                          "(self.multiheadAttention) - contract not settled yet.")
def test_masked_multi_head_attention_output_shape():
    pass
