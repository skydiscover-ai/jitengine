"""Qwen3-1.7B's decode matmul shapes: (name, K, output-row segments of the original projections)."""

HIDDEN = 2048
INTERMEDIATE = 6144
VOCAB = 151936
SHAPES = [
    ("qkv", HIDDEN, (2048, 1024, 1024)),
    ("o", HIDDEN, (2048,)),
    ("gate_up", HIDDEN, (INTERMEDIATE, INTERMEDIATE)),
    ("down", INTERMEDIATE, (HIDDEN,)),
    ("lm_head", HIDDEN, (VOCAB,)),
]
