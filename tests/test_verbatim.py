"""The copied blocks of the library stay identical to the registered code."""
import inspect

from anyintent import fast, labels
from reference import v11_latency_ref as LR
from reference import v11_lib_ref as R


def test_fast_bodies_verbatim():
    for name in ("FastXLMR", "Head", "FastXLMRLoRA", "bucket_len"):
        assert inspect.getsource(getattr(fast, name)) == inspect.getsource(getattr(LR, name)), name
    assert fast.V11_LORA_TARGETS == LR.V11_LORA_TARGETS == R.LORA_TARGETS
    assert fast.BUCKETS16 == LR.BUCKETS16


def test_label_text_helpers_verbatim():
    for name in ("strip_examples", "tokens", "norm_text", "label_text_of"):
        assert inspect.getsource(getattr(labels, name)) == inspect.getsource(getattr(R, name)), name
    assert labels.QUOTE_RE.pattern == R.QUOTE_RE.pattern and labels.LEAD == R.LEAD
