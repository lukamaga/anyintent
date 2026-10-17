"""The fast inference bodies of the v1.1 latency bench (method 3b): FastXLMR, FastXLMRLoRA and Head are verbatim
copies of the bench's code (same math as the Hugging Face XLM-R forward, written with scaled_dot_product_attention;
the last layer computes only the <s> row), with the R2Intent adapter prefix as the default of FastXLMRLoRA, plus pair
assembly from pre-tokenised ids and optional length buckets for torch.compile."""
import torch
import torch.nn.functional as F

BUCKETS = (32, 64, 96, 128, 160, 192)


# ---- verbatim copy (the v1.1 latency bench) ----
BUCKETS16 = tuple(range(32, 193, 16))     # finer buckets: less padding, more recorded graphs


def bucket_len(L, buckets=BUCKETS):
    for b in buckets:
        if L <= b:
            return b
    return int(L)


class FastXLMR(torch.nn.Module):
    """XLM-R / bge-m3 encoder forward written with F.scaled_dot_product_attention and the model's own modules (no
    copy of the weights). Same math as the Hugging Face forward; no data-dependent Python branches, so it can be
    captured in CUDA graphs and exported to ONNX. cls_only_last: the last layer computes only the <s> row (exact:
    no other position feeds the <s> output of the last layer)."""

    def __init__(self, rob, cls_only_last=True):
        super().__init__()
        cfg = rob.config
        if getattr(cfg, "position_embedding_type", "absolute") not in (None, "absolute"):
            raise ValueError(f"position_embedding_type {cfg.position_embedding_type} not supported")
        e = rob.embeddings
        self.word, self.pos, self.tte, self.ln = e.word_embeddings, e.position_embeddings, e.token_type_embeddings, e.LayerNorm
        self.pad = int(cfg.pad_token_id)
        self.nh = int(cfg.num_attention_heads)
        self.hd = int(cfg.hidden_size) // self.nh
        self.layers = torch.nn.ModuleList(list(rob.encoder.layer))
        for lay in self.layers:   # fail early if this transformers version names the modules differently
            _ = (lay.attention.self.query, lay.attention.self.key, lay.attention.self.value, lay.attention.output.dense,
                 lay.attention.output.LayerNorm, lay.intermediate.dense, lay.intermediate.intermediate_act_fn,
                 lay.output.dense, lay.output.LayerNorm)
        self.cls_only_last = cls_only_last

    def forward(self, input_ids, attention_mask):
        B = input_ids.shape[0]
        nt = input_ids.ne(self.pad).to(torch.int64)
        pos = torch.cumsum(nt, dim=1) * nt + self.pad          # create_position_ids_from_input_ids
        x = self.word(input_ids) + self.tte.weight[0]
        x = x + self.pos(pos)
        x = self.ln(x)
        am = attention_mask.to(torch.bool)[:, None, None, :]
        n = len(self.layers)
        for i, lay in enumerate(self.layers):
            sa = lay.attention.self
            xq = x[:, :1] if (self.cls_only_last and i == n - 1) else x
            q = sa.query(xq).view(B, -1, self.nh, self.hd).transpose(1, 2)
            k = sa.key(x).view(B, -1, self.nh, self.hd).transpose(1, 2)
            v = sa.value(x).view(B, -1, self.nh, self.hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
            a = a.transpose(1, 2).reshape(B, -1, self.nh * self.hd)
            h = lay.attention.output.LayerNorm(lay.attention.output.dense(a) + xq)
            f = lay.intermediate.intermediate_act_fn(lay.intermediate.dense(h))
            x = lay.output.LayerNorm(lay.output.dense(f) + h)
        return x[:, 0]


class Head(torch.nn.Module):
    # XLMRobertaClassificationHead on the <s> vector (dropout is a no-op in eval): out_proj(tanh(dense(h)))[:, 0]
    def __init__(self, cls_head):
        super().__init__()
        self.dense, self.out_proj = cls_head.dense, cls_head.out_proj

    def forward(self, h):
        return self.out_proj(torch.tanh(self.dense(h)))[:, 0]


V11_LORA_TARGETS = ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense",
                    "intermediate.dense", "output.dense")


class FastXLMRLoRA(FastXLMR):
    """FastXLMR (same math as the Hugging Face forward) with the v1.1 embedding adapter applied explicitly: every target
    linear layer returns mod(x) + scale * B(A(x)), as the training hooks do. The adapter matrices are buffers in the
    backbone dtype, so they are graph constants under torch.compile; the backbone weights are shared with the cross
    body (one backbone in memory)."""

    def __init__(self, rob, lora_sd, scale, targets=V11_LORA_TARGETS, cls_only_last=True,
                 prefix="r2intent_embed_lora.roberta."):
        super().__init__(rob, cls_only_last)
        if tuple(targets) != V11_LORA_TARGETS:
            raise ValueError(f"FastXLMRLoRA supports the targets {V11_LORA_TARGETS}, got {targets}")
        self.scale = float(scale)
        w = rob.embeddings.word_embeddings.weight
        for li in range(len(self.layers)):
            for ti, t in enumerate(V11_LORA_TARGETS):
                a = lora_sd[f"{prefix}encoder.layer.{li}.{t}.lora_A"].to(w.device, w.dtype)
                b = lora_sd[f"{prefix}encoder.layer.{li}.{t}.lora_B"].to(w.device, w.dtype)
                self.register_buffer(f"lA_{li}_{ti}", a, persistent=False)
                self.register_buffer(f"lB_{li}_{ti}", b, persistent=False)

    def _lin(self, mod, x, li, ti):
        return mod(x) + F.linear(F.linear(x, getattr(self, f"lA_{li}_{ti}")), getattr(self, f"lB_{li}_{ti}")) * self.scale

    def forward(self, input_ids, attention_mask):
        B = input_ids.shape[0]
        nt = input_ids.ne(self.pad).to(torch.int64)
        pos = torch.cumsum(nt, dim=1) * nt + self.pad
        x = self.word(input_ids) + self.tte.weight[0]
        x = x + self.pos(pos)
        x = self.ln(x)
        am = attention_mask.to(torch.bool)[:, None, None, :]
        n = len(self.layers)
        for i, lay in enumerate(self.layers):
            sa = lay.attention.self
            xq = x[:, :1] if (self.cls_only_last and i == n - 1) else x
            q = self._lin(sa.query, xq, i, 0).view(B, -1, self.nh, self.hd).transpose(1, 2)
            k = self._lin(sa.key, x, i, 1).view(B, -1, self.nh, self.hd).transpose(1, 2)
            v = self._lin(sa.value, x, i, 2).view(B, -1, self.nh, self.hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
            a = a.transpose(1, 2).reshape(B, -1, self.nh * self.hd)
            h = lay.attention.output.LayerNorm(self._lin(lay.attention.output.dense, a, i, 3) + xq)
            f = lay.intermediate.intermediate_act_fn(self._lin(lay.intermediate.dense, h, i, 4))
            x = lay.output.LayerNorm(self._lin(lay.output.dense, f, i, 5) + h)
        return x[:, 0]
# ---- end of the verbatim copy ----


def bucket_len_any(L):
    # 16-token buckets up to 192, then multiples of 64 (the embedder of a two-file model reads up to 512 tokens)
    if L <= BUCKETS16[-1]:
        return bucket_len(L, BUCKETS16)
    return int(-(-L // 64) * 64)


class PairAssembler:
    """(message, label text) pairs as <s> msg </s></s> label </s> from ids tokenised once (the XLM-R pair format); a
    pair longer than max_len falls back to the tokenizer's own longest_first truncation (the latency bench's Pipe).
    check() verifies the assembly against the tokenizer; if it differs, every pair goes through the tokenizer."""

    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, int(max_len)
        self.cls, self.sep, self.pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
        self.ok = None not in (self.cls, self.sep, self.pad)
        self.lab_ids, self.lab_texts = [], []
        self.n_fallback = 0

    def ids(self, texts):
        return [list(x) for x in self.tok(list(texts), add_special_tokens=False)["input_ids"]]

    def set_labels(self, label_texts):
        self.lab_texts = list(label_texts)
        self.lab_ids = self.ids(self.lab_texts)

    def check(self, probe_msgs):
        if not self.ok or not self.lab_texts:
            return self.ok
        for i, m in enumerate(probe_msgs):
            j = i % len(self.lab_texts)
            want = list(self.tok(m, self.lab_texts[j], truncation="longest_first", max_length=self.max_len)["input_ids"])
            got = self.pairs([m], [0], [j], self.ids([m]))[0]
            if want != got:
                self.ok = False
                break
        return self.ok

    def pairs(self, msgs, mi, lj, msg_ids=None):
        if msg_ids is None:
            msg_ids = self.ids(msgs)
        out = []
        for i, j in zip(mi, lj):
            q, la = msg_ids[i], self.lab_ids[j]
            if self.ok and len(q) + len(la) + 4 <= self.max_len:
                out.append([self.cls] + q + [self.sep, self.sep] + la + [self.sep])
            else:
                self.n_fallback += 1
                out.append(list(self.tok(msgs[i], self.lab_texts[j], truncation="longest_first",
                                         max_length=self.max_len)["input_ids"]))
        return out


def pad_batch(seqs, pad_id, device, bucket=False, min_b=1):
    # rows beyond len(seqs) (min_b) repeat row 0 and are dropped after the forward
    n = len(seqs)
    L = max(len(s) for s in seqs)
    if bucket:
        L = bucket_len_any(L)
    B = max(n, min_b)
    ids = torch.full((B, L), int(pad_id), dtype=torch.long)
    m = torch.zeros((B, L), dtype=torch.long)
    for i in range(B):
        s = seqs[i if i < n else 0]
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        m[i, :len(s)] = 1
    return ids.to(device), m.to(device)
