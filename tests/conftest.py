"""Test fixtures: an offline tokenizer (trained here, XLM-R special tokens and pair format) and tiny random XLM-R models
saved in every layout the library loads, with the R2Intent names (PROTOCOL E5 (10f), decision D6):
r2intent_meta.json (name key r2intent), the adapter tensors under r2intent_embed_lora.* and the safetensors metadata
key r2intent, as the P job and the PACK build write them. No network: the Hugging Face hub is set offline and outbound
socket connections are refused for the whole test session."""
import json
import os
import socket
import sys

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _no_network(self, address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    if host in ("127.0.0.1", "::1", "localhost") or isinstance(address, (str, bytes)):
        return _real_connect(self, address, *args, **kwargs)
    raise OSError(f"network disabled in the anyintent tests (connect to {address})")


_real_connect = socket.socket.connect
socket.socket.connect = _no_network

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch         # noqa: E402

from reference import v11_lib_ref as R   # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

LONG = ("Hello, I am writing because my order from last month still has not arrived and nobody answers my emails. " * 14
        ).strip()
MESSAGES = [
    "where is my parcel, it was supposed to arrive yesterday",
    "I want to cancel my subscription right now",
    "how do I change the password of my account",
    "my card was blocked after I typed the wrong pin",
    "can I get a refund for a damaged item",
    "what time does the store open on sunday",
    "please send me the invoice for my last order",
    "I lost my card, what should I do",
    "the app keeps crashing when I log in",
    "book a table for two at seven",
    "Kur yra mano siuntinys? Jis turėjo atvykti vakar.",
    "Noriu atšaukti prenumeratą ir susigrąžinti pinigus.",
    "Pamiršau slaptažodį, kaip jį pakeisti?",
    "Gdzie jest moja paczka? Miała przyjść wczoraj.",
    "Chcę zwrócić uszkodzony towar i odzyskać pieniądze.",
    "Zgubiłem kartę, co mam zrobić? Żółć, gęś, źdźbło.",
    "",
    "   ",
    "?",
    LONG,
    "refund",
    "card",
]
LABELS = [
    {"name": "order status", "description": "The customer asks where an order or parcel is, for example \"where is my parcel\"."},
    {"name": "cancel subscription", "description": "The customer wants to end a subscription or membership."},
    {"name": "change password", "description": "The customer wants to change or reset the account password, e.g. \"I forgot my password\"."},
    {"name": "card blocked", "description": "A payment card was blocked or declined."},
    {"name": "refund request", "description": "The customer asks for money back for a product or service."},
    {"name": "opening hours", "description": "Questions about when a store or office is open."},
    {"name": "invoice request", "description": "The customer asks for an invoice or a receipt."},
    {"name": "lost card", "description": "A card was lost or stolen; users might say \"I lost my card\" or \"my card is gone\"."},
    {"name": "app crash", "description": "The mobile app or website crashes or does not load."},
    {"name": "restaurant booking", "description": "Reserve a table at a restaurant."},
    {"name": "delivery address change", "description": "Change the address an order will be delivered to."},
    {"name": "payment methods", "description": "Which ways of paying are accepted."},
    {"name": "account deletion", "description": "The customer wants to delete the account and all data."},
    {"name": "speak to human", "description": "The customer asks to talk to a person instead of a bot."},
    {"name": "greeting", "description": ""},
    {"name": "thanks"},
    "goodbye",
    {"name": "siuntos būsena", "description": "Klientas klausia, kur yra jo užsakymas."},
    {"name": "zwrot pieniędzy", "description": "Klient prosi o zwrot pieniędzy za towar."},
    {"name": "loan interest rate", "description": "Questions about the interest rate of a loan or a mortgage."},
]
CORPUS = MESSAGES + [l if isinstance(l, str) else (l["name"] + ": " + (l.get("description") or "")) for l in LABELS] + [
    "the quick brown fox jumps over the lazy dog", "užsakymas prenumerata slaptažodis kortelė pinigai grąžinti",
    "zamówienie subskrypcja hasło karta pieniądze zwrot", "customer account order delivery payment invoice",
    "ąčęėįšųūž ĄČĘĖĮŠŲŪŽ ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ"]


def build_tokenizer(d):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import AutoTokenizer
    try:
        from transformers import PreTrainedTokenizerFast as TokFast
    except ImportError:   # transformers 5 name
        from transformers import TokenizersBackend as TokFast
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tk = Tokenizer(models.Unigram())
    tk.normalizer = normalizers.NFKC()
    tk.pre_tokenizer = pre_tokenizers.Metaspace()
    tk.decoder = decoders.Metaspace()
    tk.train_from_iterator(CORPUS * 3, trainer=trainers.UnigramTrainer(vocab_size=700, special_tokens=specials,
                                                                         unk_token="<unk>", show_progress=False))
    assert [tk.token_to_id(t) for t in specials] == [0, 1, 2, 3, 4]
    tk.post_processor = processors.TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
                                                      special_tokens=[("<s>", 0), ("</s>", 2)])
    hf = TokFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="</s>",
                 cls_token="<s>", pad_token="<pad>", mask_token="<mask>", model_max_length=514)
    hf.save_pretrained(d)
    tok = AutoTokenizer.from_pretrained(d)
    ids = tok("order status", "lost card")["input_ids"]
    assert ids[0] == 0 and ids[-1] == 2 and tok.pad_token_id == 1 and ids.count(2) == 3, ids
    return tok


def tiny_config(tok, num_labels=None):
    from transformers import XLMRobertaConfig
    kw = dict(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
              max_position_embeddings=520, type_vocab_size=1, pad_token_id=1, bos_token_id=0, eos_token_id=2,
              initializer_range=0.1)
    if num_labels:
        kw["num_labels"] = num_labels
    return XLMRobertaConfig(**kw)


def tiny_cross(tok, seed):
    from transformers import XLMRobertaForSequenceClassification
    torch.manual_seed(seed)
    m = XLMRobertaForSequenceClassification(tiny_config(tok, 1)).eval()
    with torch.no_grad():   # logits of order 1, so that score differences are meaningful at 1e-4
        m.classifier.out_proj.weight.mul_(30.0)
    return m


def v11_meta(lora, w, k, **extra):
    meta = {"r2intent": "v1.1", "arm": "A1", "run_id": "smoke", "seed": 0, "smoke": True, "shipped": True,
            "license": "apache-2.0", "languages": ["en", "lt", "pl"],
            "label_text_format": "name: description (name only without a description)",
            "passes": {"embedding": {"adapter": "on", "vector": "<s> of the last layer of model.roberta, L2-normalised"},
                       "cross": {"adapter": "off", "score": "logits[:, 0] of the (message, label text) pair",
                                 "pair_truncation": R.PAIR_TRUNCATION}},
            "fusion": {"w": w, "k": k, "formula": "cross logit + w x cosine over the top-k candidates",
                       "w_grid": [0, 5, 10, 15, 20, 30, 40, 60]},
            "msg_max_len": 128, "label_max_len": 192, "pair_max_len": 160,
            "adapter": {**lora.spec(), "dtype_in_file": "float32"}}
    meta.update(extra)
    return meta


def make_one_file(d, tok, seed=0, w=10.0, k=8, final=False, tau=None, mode=None):
    m = tiny_cross(tok, seed)
    lora = R.EmbedLoRA(m.roberta, 4, 8, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for b in lora.B:
            b.copy_(torch.randn(b.shape, generator=g) * 0.3)
    meta = v11_meta(lora, w, k)
    if final:
        meta.update({"r2intent": "1.2", "license": "cc-by-nc-4.0", "final_data": "pub"})
        meta["fusion"]["tau"] = tau
        if mode:
            meta["label_text_mode"] = mode
    R.save_one_file(m, lora, tok, d, meta)
    return d


def make_two_file(root, tok, seed=3, w=5.0, k=8, with_json=True):
    from transformers import XLMRobertaModel
    os.makedirs(root, exist_ok=True)
    emb_dir, cross_dir = os.path.join(root, "embedder"), os.path.join(root, "cross")
    torch.manual_seed(seed)
    XLMRobertaModel(tiny_config(tok), add_pooling_layer=True).save_pretrained(emb_dir)
    tok.save_pretrained(emb_dir)
    m = tiny_cross(tok, seed + 1)
    sd = {kk: v.detach().to("cpu", torch.float16).contiguous() for kk, v in m.state_dict().items()}
    m.save_pretrained(cross_dir, state_dict=sd)
    tok.save_pretrained(cross_dir)
    json.dump({"variant": "a", "seed": 0, "k": k, "pair_max_len": 160, "pair_truncation": "longest_first",
               "msg_max_len": 128, "label_max_len": 192, "license": "apache-2.0",
               "label_text_format": "name: description (name only without a description)"},
              open(os.path.join(cross_dir, "r2intent_meta.json"), "w", encoding="utf-8"))
    if with_json:
        json.dump({"format": "anyintent-two-file", "version": 1,
                   "embedder": {"path": "embedder", "max_len": 512}, "cross": {"path": "cross"},
                   "fusion": {"w": w, "k": k, "tau": None}, "label_text_mode": "v2"},
                  open(os.path.join(root, "anyintent.json"), "w", encoding="utf-8"))
    return root


@pytest.fixture(scope="session")
def tok(tmp_path_factory):
    return build_tokenizer(str(tmp_path_factory.mktemp("tok")))


@pytest.fixture(scope="session")
def one_file(tmp_path_factory, tok):
    return make_one_file(str(tmp_path_factory.mktemp("m") / "one_file"), tok)


@pytest.fixture(scope="session")
def final_file(tmp_path_factory, tok):
    return make_one_file(str(tmp_path_factory.mktemp("m") / "final_dom"), tok, seed=5, w=15.0, final=True, tau=0.5,
                         mode="noex")


@pytest.fixture(scope="session")
def two_file(tmp_path_factory, tok):
    return make_two_file(str(tmp_path_factory.mktemp("m") / "two_file"), tok)


def label_texts_ref(mode="v2"):
    out = []
    for l in LABELS:
        n = l if isinstance(l, str) else l["name"]
        d = "" if isinstance(l, str) else (l.get("description") or "")
        if mode == "noex":
            d = R.strip_examples(d, n) if d else ""
        out.append(n if mode == "name" else R.label_text_of(n, d))
    return out
