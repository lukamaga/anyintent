# GENERATED from the project's registered v1.1 runtime (not in this repository): byte for byte but for the
# R2Intent names in code and comments; used only as a reference of the tests. Do not edit.
# R2Intent v1.1 runtime: load the one-file model and classify a message against label texts ("name: description").
# One backbone, two passes: the embedding pass (the LoRA adapter on) retrieves the top-k labels by cosine, the
# cross-encoder pass (the adapter off, exactly the trained cross-encoder) rescores those k pairs, and the final score is
# cross logit + w x cosine (w, k and the token lengths are in r2intent_meta.json). Eager PyTorch reference code. On a GPU
# it computes what the DEV scores were computed with: fp16 weights under torch.autocast(float16), the adapter matrices
# kept in fp32 and cast by autocast, as the v1.1 training library. The latency numbers of the model card come from the optimised
# runtime (torch.compile, CUDA graphs, length buckets).
import contextlib, json, os
import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _dtype_kw(dtype):
    # transformers >= 4.56 takes `dtype`, older versions `torch_dtype` (an unknown keyword is not rejected there, it
    # would be passed to the config and the weights would load in fp32)
    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    return {("dtype" if major_minor >= (4, 56) else "torch_dtype"): dtype}


class R2IntentV11:
    def __init__(self, path, device=None, dtype=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda = str(self.dev).startswith("cuda")
        dt = dtype or (torch.float16 if self.cuda else torch.float32)
        self.meta = json.load(open(os.path.join(path, "r2intent_meta.json"), encoding="utf-8"))
        self.tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForSequenceClassification.from_pretrained(path, **_dtype_kw(dt))
        self.model = m.to(self.dev).eval()
        ad = self.meta["adapter"]
        self.scale = float(ad["alpha"]) / int(ad["rank"])
        pre = ad["prefix"] + "roberta."
        self.lora = {}
        with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
            for li, layer in enumerate(self.model.roberta.encoder.layer):
                for t in ad["targets"]:
                    mod = layer.get_submodule(t)
                    a = f.get_tensor(f"{pre}encoder.layer.{li}.{t}.lora_A").to(self.dev, torch.float32)
                    b = f.get_tensor(f"{pre}encoder.layer.{li}.{t}.lora_B").to(self.dev, torch.float32)
                    self.lora[mod] = (a, b)
        self.on = False
        for mod in self.lora:
            mod.register_forward_hook(self._hook)
        self.labels, self.L = [], None

    def _amp(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.cuda else contextlib.nullcontext()

    def _hook(self, mod, inp, out):
        if not self.on:
            return None
        a, b = self.lora[mod]
        x = inp[0]
        return out + (F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * self.scale).to(out.dtype)

    @torch.no_grad()
    def embed(self, texts, max_len):
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.dev)
        self.on = True
        try:
            with self._amp():
                h = self.model.roberta(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
        finally:
            self.on = False
        return F.normalize(h.float(), dim=-1)

    @torch.no_grad()
    def cross(self, msgs, labels):
        enc = self.tok(list(msgs), list(labels), padding=True, truncation="longest_first",
                       max_length=int(self.meta["pair_max_len"]), return_tensors="pt").to(self.dev)
        with self._amp():
            return self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.view(-1).float()

    def set_labels(self, label_texts):
        self.labels = list(label_texts)
        self.L = self.embed(self.labels, int(self.meta["label_max_len"]))

    @torch.no_grad()
    def predict(self, msg, k=None, w=None, return_index=False):
        k = int(k or self.meta["fusion"]["k"])
        w = float(self.meta["fusion"]["w"] if w is None else w)
        q = self.embed([msg], int(self.meta["msg_max_len"]))
        cos, top = (q @ self.L.T)[0].topk(min(k, len(self.labels)))
        c = self.cross([msg] * len(top), [self.labels[i] for i in top.tolist()])
        j = int(top[int((c + w * cos).argmax())])
        return j if return_index else self.labels[j]


if __name__ == "__main__":
    import sys
    clf = R2IntentV11(sys.argv[1])
    clf.set_labels([line.strip() for line in open(sys.argv[2], encoding="utf-8") if line.strip()])   # one label text per line
    print(clf.predict(sys.argv[3]))
