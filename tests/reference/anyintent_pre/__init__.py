"""AnyIntent: zero-shot intent classification with a label set chosen at run time (retrieve the top-k labels by
embedding, rerank them with a cross-encoder, fuse the two scores). Inference only."""
from ._version import __version__
from .hub import AccessError, AnyIntentError
from .model import AnyIntent, Prediction

__all__ = ["AnyIntent", "Prediction", "AnyIntentError", "AccessError", "__version__"]
