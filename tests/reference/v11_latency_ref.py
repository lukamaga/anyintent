# GENERATED from the project's registered v1.1 latency bench (not in this repository): verbatim copies, with the
# R2Intent names in code and comments; used only as the reference of the equivalence tests. Do not edit.
import os, gc, json, time, math, traceback
import numpy as np, torch
import torch.nn.functional as F


PAIR_MAX_LEN, MSG_MAX_LEN, LABEL_MAX_LEN = 192, 128, 192


PAIR_TRUNCATION = "longest_first"


BUCKETS = (32, 64, 96, 128, 160, 192)     # padded lengths for the static-shape runtimes (CUDA graphs)


BUCKETS16 = tuple(range(32, 193, 16))     # finer buckets: less padding, more recorded graphs


CHUNK = 128                               # pairs per cross-encoder forward in batch mode (pairs sorted by length)


EMB_CHUNK = 64                            # texts per embedding forward (label precompute, batch-64 queries)


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def bucket_len(L, buckets=BUCKETS):
    for b in buckets:
        if L <= b:
            return b
    return int(L)


class StageTimer:
    # Accumulates wall time per stage (with a device sync at every mark); a no-op when st is None
    def __init__(self, st, dev):
        self.st, self.dev = st, dev
        if st is not None:
            sync(dev)
            self.t = time.perf_counter()

    def __call__(self, name):
        if self.st is None:
            return
        sync(self.dev)
        t = time.perf_counter()
        self.st[name] = self.st.get(name, 0.0) + (t - self.t)
        self.t = t


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


class HFBody(torch.nn.Module):
    # Fallback body: the Hugging Face forward itself
    def __init__(self, rob):
        super().__init__()
        self.rob = rob

    def forward(self, input_ids, attention_mask):
        return self.rob(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]


class Head(torch.nn.Module):
    # XLMRobertaClassificationHead on the <s> vector (dropout is a no-op in eval): out_proj(tanh(dense(h)))[:, 0]
    def __init__(self, cls_head):
        super().__init__()
        self.dense, self.out_proj = cls_head.dense, cls_head.out_proj

    def forward(self, h):
        return self.out_proj(torch.tanh(self.dense(h)))[:, 0]


class Engine:
    # emb(ids, mask) -> normalised fp32 [B, H]; cross(ids, mask) -> fp32 [B]
    def __init__(self, name, emb, cross, bucketed=False, min_q=1, buckets=BUCKETS):
        self.name, self.emb, self.cross, self.bucketed, self.min_q = name, emb, cross, bucketed, min_q
        self.buckets = tuple(buckets)


class Pipe:
    """Two-stage pipeline for one engine and one label set. Label token ids and label embeddings are computed once
    (as in production). Pairs are assembled from pre-tokenised ids as <s> msg </s></s> label </s> (the XLM-R pair
    format); a pair longer than PAIR_MAX_LEN falls back to the tokenizer's own longest_first truncation."""

    def __init__(self, eng, tok, label_texts, dev, tq=None):
        self.eng, self.tok, self.dev = eng, tok, dev
        self.tq = tq if tq is not None else tok
        self.same_tok = self.tq is self.tok
        self.cls, self.sep, self.pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
        self.qcls, self.qsep, self.qpad = self.tq.cls_token_id, self.tq.sep_token_id, self.tq.pad_token_id
        self.label_texts = list(label_texts)
        self.lab_ids = [list(x) for x in tok(self.label_texts, add_special_tokens=False)["input_ids"]]
        lq = self.lab_ids if self.same_tok else [list(x) for x in self.tq(self.label_texts, add_special_tokens=False)["input_ids"]]
        self.n_label_trunc = int(sum(len(x) + 2 > LABEL_MAX_LEN for x in lq))
        self.n_pair_fallback = 0
        self.Lm = self.embed_seqs([[self.qcls] + x[:LABEL_MAX_LEN - 2] + [self.qsep] for x in lq])

    def _pad(self, seqs, min_b=1, padv=None):
        padv = self.pad if padv is None else padv
        n = len(seqs)
        L = max(len(s) for s in seqs)
        if self.eng.bucketed:
            L = bucket_len(L, self.eng.buckets)
        B = max(n, min_b)
        ids = np.full((B, L), padv, dtype=np.int64)
        m = np.zeros((B, L), dtype=np.int64)
        for i in range(B):
            s = seqs[i if i < n else 0]     # extra rows (min_b) repeat row 0 and are dropped after the forward
            ids[i, :len(s)] = s
            m[i, :len(s)] = 1
        return torch.from_numpy(ids).to(self.dev), torch.from_numpy(m).to(self.dev)

    def embed_seqs(self, seqs):
        n = len(seqs)
        order = np.argsort([len(s) for s in seqs], kind="stable")
        outs = []
        for i in range(0, n, EMB_CHUNK):
            idx = order[i:i + EMB_CHUNK]
            ids, m = self._pad([seqs[j] for j in idx], self.eng.min_q, self.qpad)
            outs.append(self.eng.emb(ids, m)[:len(idx)].float())
        E = torch.cat(outs)
        inv = np.empty(n, dtype=np.int64)
        inv[order] = np.arange(n)
        return E[torch.from_numpy(inv).to(E.device)].contiguous()

    def pair_seqs(self, msgs, praw, cand_np):
        seqs = []
        for i in range(len(msgs)):
            q = praw[i]
            for j in cand_np[i]:
                la = self.lab_ids[int(j)]
                if len(q) + len(la) + 4 <= PAIR_MAX_LEN:
                    seqs.append([self.cls] + q + [self.sep, self.sep] + la + [self.sep])
                else:
                    self.n_pair_fallback += 1
                    seqs.append(list(self.tok(msgs[i], self.label_texts[int(j)], truncation=PAIR_TRUNCATION,
                                              max_length=PAIR_MAX_LEN)["input_ids"]))
        return seqs

    def run(self, msgs, k, full=False, st=None):
        tm = StageTimer(st, self.dev)
        msgs = list(msgs)
        n = len(msgs)
        qraw = self.tq(msgs, add_special_tokens=False)["input_ids"]
        praw = qraw if self.same_tok else self.tok(msgs, add_special_tokens=False)["input_ids"]
        qraw = [list(q) for q in qraw]
        praw = qraw if self.same_tok else [list(q) for q in praw]
        ids, m = self._pad([[self.qcls] + q[:MSG_MAX_LEN - 2] + [self.qsep] for q in qraw], self.eng.min_q, self.qpad)
        tm("tok")
        Q = self.eng.emb(ids, m)[:n]
        tm("qemb")
        cand = torch.topk(Q @ self.Lm.T, k, dim=1).indices
        cand_np = cand.cpu().numpy()
        tm("retr")
        seqs = self.pair_seqs(msgs, praw, cand_np)
        N = len(seqs)
        if N <= CHUNK:
            order, batches = None, [np.arange(N)]
        else:
            order = np.argsort([len(s) for s in seqs], kind="stable")
            batches = [order[i:i + CHUNK] for i in range(0, N, CHUNK)]
        enc = [self._pad([seqs[j] for j in b]) for b in batches]
        tm("tok")
        out = torch.cat([self.eng.cross(ids, m)[:len(b)].float() for (ids, m), b in zip(enc, batches)])
        if order is not None:
            inv = np.empty(N, dtype=np.int64)
            inv[order] = np.arange(N)
            out = out[torch.from_numpy(inv).to(out.device)]
        sc = out.view(n, k)
        if full:
            res = (cand_np, sc.cpu().numpy())
        else:
            res = cand.gather(1, sc.argmax(1, keepdim=True)).squeeze(1).cpu().numpy()
        tm("cross")
        return res


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


class PipeFused(Pipe):
    """Pipe with score fusion: the prediction is the argmax over the top-k candidates of cross logit + w x cosine."""

    def __init__(self, eng, tok, label_texts, dev, w=0.0, tq=None):
        super().__init__(eng, tok, label_texts, dev, tq)
        self.w = float(w)

    def run(self, msgs, k, full=False, st=None):
        tm = StageTimer(st, self.dev)
        msgs = list(msgs)
        n = len(msgs)
        qraw = [list(q) for q in self.tq(msgs, add_special_tokens=False)["input_ids"]]
        praw = qraw if self.same_tok else [list(q) for q in self.tok(msgs, add_special_tokens=False)["input_ids"]]
        ids, m = self._pad([[self.qcls] + q[:MSG_MAX_LEN - 2] + [self.qsep] for q in qraw], self.eng.min_q, self.qpad)
        tm("tok")
        Q = self.eng.emb(ids, m)[:n]
        tm("qemb")
        vals, cand = torch.topk(Q @ self.Lm.T, k, dim=1)
        cand_np = cand.cpu().numpy()
        tm("retr")
        seqs = self.pair_seqs(msgs, praw, cand_np)
        N = len(seqs)
        if N <= CHUNK:
            order, batches = None, [np.arange(N)]
        else:
            order = np.argsort([len(s) for s in seqs], kind="stable")
            batches = [order[i:i + CHUNK] for i in range(0, N, CHUNK)]
        enc = [self._pad([seqs[j] for j in b]) for b in batches]
        tm("tok")
        out = torch.cat([self.eng.cross(ids, m)[:len(b)].float() for (ids, m), b in zip(enc, batches)])
        if order is not None:
            inv = np.empty(N, dtype=np.int64)
            inv[order] = np.arange(N)
            out = out[torch.from_numpy(inv).to(out.device)]
        sc = out.view(n, k)
        fused = sc + self.w * vals.float()
        if full:
            res = (cand_np, sc.cpu().numpy(), vals.float().cpu().numpy())
        else:
            res = cand.gather(1, fused.argmax(1, keepdim=True)).squeeze(1).cpu().numpy()
        tm("cross")
        return res
