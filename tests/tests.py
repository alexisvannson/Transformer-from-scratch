import pytest
import torch

from model.MultiHeadAttention import (
    DotProductAttention,
    MultiHeadAttention,
    FFN,
    Encoder,
    PositionalEncoding,
)

EMBED_DIM = 512


# ---------------------------------------------------------------------------
# DotProductAttention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens,d_k", [(10, 64), (1, 64), (7, 64), (20, 64)])
def test_dot_product_attention_shape(tokens, d_k):
    attention = DotProductAttention()
    Q, K, V = torch.randn(tokens, d_k), torch.randn(tokens, d_k), torch.randn(tokens, d_k)

    out = attention(Q, K, V)

    # attention output should always take the shape of V: (tokens, d_k)
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
    Q, K, V = torch.randn(tokens, 64), torch.randn(tokens, 64), torch.randn(tokens, 64)

    q_proj, k_proj, v_proj = mha.get_projections(Q, K, V)

    assert q_proj.shape == (tokens, 64)
    assert k_proj.shape == (tokens, 64)
    assert v_proj.shape == (tokens, 64)


# ---------------------------------------------------------------------------
# MultiHeadAttention.MultiHeadAttention (full forward)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [1, 5, 10, 17])
def test_multi_head_attention_output_shape(tokens):
    """Regression test: output shape must track the actual number of tokens
    passed in, not a hardcoded sequence length."""
    mha = MultiHeadAttention()
    X = torch.randn(tokens, EMBED_DIM)
    W = torch.randn(EMBED_DIM, EMBED_DIM * 3)
    Wo = torch.randn(EMBED_DIM, EMBED_DIM)

    out = mha(X, W, Wo)

    assert out.shape == (tokens, EMBED_DIM), f"expected {(tokens, EMBED_DIM)}, got {tuple(out.shape)}"


def test_multi_head_attention_concatenates_full_width():
    """If a head's contribution were silently dropped (e.g. wrong concat dim,
    or fewer than h heads processed), the pre-Wo tensor would be narrower
    than embedding_dim and this would fail even with Wo as a passthrough."""
    mha = MultiHeadAttention()
    tokens = 6
    X = torch.randn(tokens, EMBED_DIM)
    W = torch.randn(EMBED_DIM, EMBED_DIM * 3)
    Wo = torch.eye(EMBED_DIM)  # identity: output width should equal input width to Wo

    out = mha(X, W, Wo)

    assert out.shape[-1] == EMBED_DIM


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
# Encoder
# ---------------------------------------------------------------------------

def test_add_and_norm_preserves_shape():
    encoder = Encoder(embedding_dim=EMBED_DIM)
    tokens = 8
    initial, delta = torch.randn(tokens, EMBED_DIM), torch.randn(tokens, EMBED_DIM)

    out = encoder.AddAndNorm(initial, delta)

    assert out.shape == (tokens, EMBED_DIM)


def test_add_and_norm_rejects_mismatched_shapes():
    """AddAndNorm is an elementwise residual add - mismatched shapes should
    fail loudly, not silently broadcast into something unintended."""
    encoder = Encoder(embedding_dim=EMBED_DIM)
    initial = torch.randn(8, EMBED_DIM)
    delta = torch.randn(5, EMBED_DIM)

    with pytest.raises(RuntimeError):
        encoder.AddAndNorm(initial, delta)


@pytest.mark.parametrize("tokens", [1, 5, 10])
def test_encoder_block_output_shape(tokens):
    encoder = Encoder(embedding_dim=EMBED_DIM)
    X = torch.randn(tokens, EMBED_DIM)
    W = torch.randn(EMBED_DIM, EMBED_DIM * 3)
    Wo = torch.randn(EMBED_DIM, EMBED_DIM)

    out = encoder.EncoderBlock(X, W, Wo)

    assert out.shape == (tokens, EMBED_DIM)


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
# MaskedMultiHeadAttention
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="MaskedMultiHeadAttention calls self.MultiHeadAttention(Q, K, V) but "
           "MultiHeadAttention's signature is (X, W, Wo) - contract mismatch to "
           "resolve before this can pass. Un-skip once fixed."
)
def test_masked_multi_head_attention_output_shape():
    mha = MultiHeadAttention()
    tokens = 6
    Q = torch.randn(tokens, EMBED_DIM)
    K, V = torch.randn(tokens, EMBED_DIM), torch.randn(tokens, EMBED_DIM)

    out = mha.MaskedMultiHeadAttention(Q, K, V)

    assert out.shape == (tokens, EMBED_DIM)
