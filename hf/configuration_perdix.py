"""Perdix 설정 (transformers 연동용)."""
from transformers import PretrainedConfig


class PerdixConfig(PretrainedConfig):
    model_type = "perdix"
    attribute_map = {"hidden_size": "dim", "num_hidden_layers": "n_layers",
                     "num_attention_heads": "n_heads",
                     "max_position_embeddings": "max_seq_len"}

    def __init__(self, vocab_size=49152, dim=2048, n_layers=20, n_heads=16,
                 ffn_dim=8192, max_seq_len=2048, rope_theta=10000.0,
                 norm_eps=1e-5, init_std=0.02, bos_token_id=0, eos_token_id=0,
                 pad_token_id=1, tie_word_embeddings=True, **kwargs):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads          # differential attention은 head_dim을 반으로 나눠 2쌍 사용
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.init_std = init_std
        kwargs.setdefault("use_cache", False)  # KV 캐시 미구현
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         pad_token_id=pad_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)
