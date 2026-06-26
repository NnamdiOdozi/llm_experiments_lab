"""Template registry — maps template keys to their model/data modules."""

from backend.training.templates.transformer.model import (
    TinyTransformerLM,
    build_model_from_config as build_transformer,
)
from backend.training.templates.rnn.model import (
    CharRNN,
    build_model_from_config as build_rnn,
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
}
