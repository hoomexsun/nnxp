from .attn import AttnSeq2Seq
from .lstm import LSTMSeq2Seq
from .cnn import CNNSeq2Seq
from .transformers import TransformerSeq2Seq

MODEL_REGISTRY = dict(
    attention=AttnSeq2Seq,
    lstm=LSTMSeq2Seq,
    cnn=CNNSeq2Seq,
    transformers=TransformerSeq2Seq,
)


def load_model(model_name, model_conf, device):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not supported.")
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(model_conf, device)
    return model
