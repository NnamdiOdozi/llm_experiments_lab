"""Template registry — maps template keys to their model/data modules."""

from backend.training.templates.transformer.model import (
    TinyTransformerLM,
    build_model_from_config as build_transformer,
)
from backend.training.templates.rnn.model import (
    CharRNN,
    build_model_from_config as build_rnn,
)
from backend.training.templates.moe.model import (
    TinyMoeLM,
    build_model_from_config as build_moe,
)

TEMPLATE_REGISTRY = {
    "transformer": {
        "build_model": build_transformer,
        "model_class": TinyTransformerLM,
    },
    "rnn": {
        "build_model": build_rnn,
        "model_class": CharRNN,
    },
    "moe": {
        "build_model": build_moe,
        "model_class": TinyMoeLM,
    },
}
