"""
microbot_core.py  -  MicroBot v7  |  ~1.6M params, numpy-only
==============================================================
v7 upgrades vs v6:
  ① SwiGLU FFN  (val*swish(gate)) — better than tanh, same param count
  ② BM25 retrieval  — length-normalised, beats plain TF-IDF
  ③ word_assoc skip-gram — semantic query expansion from curated corpus
  ④ IDF-weighted sentence vectors with L2 normalisation
  ⑤ Fine-tuned magic numbers: temp=0.72, top-k=40, lr=1.5e-3, smooth=0.10
  ⑥ 75 000 training steps  (resumable, saves every 100)
  ⑦ STEPS_PER_CHUNK configurable for slow Android devices
  ⑧ Full fallback chain: BM25→word-facts→topic→web→generate

Architecture: N_LAYER=2  N_EMBD=256  N_HEAD=4  FFN_DIM=680  BLOCK_SIZE=64
              params ≈ 1.60M   (SwiGLU: wg+w1+w2 per layer)
"""
import os, math, random, pickle, re, zlib, json
import urllib.request, urllib.parse
from collections import deque, Counter

import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
random.seed(42)
np.random.seed(42)

# ==============================================================================
#  FILES
# ==============================================================================
MEMORY_MB   = "memory.mb"
WEIGHTS_PKL = "weights.pkl"
PROG_FILE   = "progress.txt"
LOSS_LOG    = "loss_log.txt"

# ==============================================================================
#  HYPER-PARAMS  — v7 fine-tuned
# ==============================================================================
N_LAYER    = 2
N_EMBD     = 256
FFN_DIM    = 680          # SwiGLU: ≈2/3 × 4×N_EMBD  (same total params as 4×N_EMBD tanh)
BLOCK_SIZE = 64
N_HEAD     = 4
HEAD_DIM   = N_EMBD // N_HEAD   # 64

TOTAL_STEPS     = 75_000
STEPS_PER_CHUNK = 75_000        # change to 2-10 for slow Android, does NOT affect quality
LR_BASE         = 1.5e-3        # tuned for 256-dim @ 75k
LR_WARMUP       = 800
LR_FLOOR        = 0.025         # 2.5% of LR_BASE minimum
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
LABEL_SMOOTH    = 0.10          # slightly higher → more generalisation
TEMPERATURE     = 0.72          # confident but not rigid
TOP_K           = 40            # top-k + temperature sampling
BETA1, BETA2, EPS_ADAM = 0.9, 0.999, 1e-8

# BM25 retrieval parameters (standard Robertson 1994)
BM25_K1 = 1.5
BM25_B  = 0.75

# ==============================================================================
#  DATASET
# ==============================================================================
_DATASET = None

def _get_dataset():
    global _DATASET
    if _DATASET is None:
        try:
            from dataset import load_dataset
            _DATASET = load_dataset()
        except ImportError:
            _DATASET = {"pairs": [], "word_facts": {}, "topic_index": {}}
    return _DATASET

def _dataset_pairs():  return _get_dataset().get("pairs", [])
def _word_facts():     return _get_dataset().get("word_facts", {})

def _topic_summary(topic):
    try:
        from dataset import get_topic_summary
        return get_topic_summary(_get_dataset(), topic)
    except Exception:
        return None

# ==============================================================================
#  MEMORY
# ==============================================================================
def _mem_compress(pairs):  return zlib.compress(json.dumps(pairs).encode(), level=9)
def _mem_decompress(data): return json.loads(zlib.decompress(data).decode())

def load_memory():
    if os.path.exists(MEMORY_MB):
        try:
            return _mem_decompress(open(MEMORY_MB, "rb").read())
        except Exception:
            pass
    return []

def save_memory(pairs):
    with open(MEMORY_MB, "wb") as f:
        f.write(_mem_compress(pairs))

_PAIRS_CACHE     = None
_PAIRS_CACHE_KEY = None

_BAD_ANSWERS = {
    "i don't have that in my knowledge yet",
    "i'm still building my understanding",
}

def _is_good_pair(pair):
    if "=>" not in pair:
        return False
    _, a = pair.split("=>", 1)
    a = a.strip().lower()
    if len(a) < 12:
        return False
    for bad in _BAD_ANSWERS:
        if a.startswith(bad[:35].lower()):
            return False
    return True

def _all_pairs():
    global _PAIRS_CACHE, _PAIRS_CACHE_KEY
    mem  = [p for p in load_memory() if _is_good_pair(p)]
    dset = _dataset_pairs()
    key  = (len(mem), len(dset))
    if key == _PAIRS_CACHE_KEY and _PAIRS_CACHE is not None:
        return _PAIRS_CACHE
    seen, out = set(), []
    for p in dset + mem:
        if p not in seen:
            seen.add(p); out.append(p)
    _PAIRS_CACHE     = out
    _PAIRS_CACHE_KEY = key
    return out

# ==============================================================================
#  CHAR VOCAB
# ==============================================================================
CHARS      = sorted(set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?':;()-"))
BOS        = len(CHARS)
VOCAB_SIZE = len(CHARS) + 1
stoi       = {ch: i for i, ch in enumerate(CHARS)}
itos       = {i: ch for i, ch in enumerate(CHARS)}

def tokenise(text):
    ids = [BOS]
    for ch in text.lower():
        if ch in stoi:
            ids.append(stoi[ch])
    ids.append(BOS)
    return ids

# ==============================================================================
#  MODEL  —  v7: SwiGLU FFN (wg=gate, w1=value, w2=out)
# ==============================================================================
def _nm(r, c, std): return np.random.normal(0, std, (r, c)).astype(np.float32)
def _nv(n, val=1.0): return np.full(n, val, dtype=np.float32)

def build_model():
    sd = {
        "wte": _nm(VOCAB_SIZE, N_EMBD, 0.02),
        "wpe": _nm(BLOCK_SIZE, N_EMBD, 0.01),
    }
    for i in range(N_LAYER):
        sd[f"l{i}.wq"]  = _nm(N_EMBD,   N_EMBD,   0.02)
        sd[f"l{i}.wk"]  = _nm(N_EMBD,   N_EMBD,   0.02)
        sd[f"l{i}.wv"]  = _nm(N_EMBD,   N_EMBD,   0.02)
        sd[f"l{i}.wo"]  = _nm(N_EMBD,   N_EMBD,   0.02 / math.sqrt(N_LAYER))
        # SwiGLU FFN: gate + value projections (both [FFN_DIM, N_EMBD])
        sd[f"l{i}.wg"]  = _nm(FFN_DIM,  N_EMBD,   0.02)   # gate
        sd[f"l{i}.w1"]  = _nm(FFN_DIM,  N_EMBD,   0.02)   # value
        sd[f"l{i}.w2"]  = _nm(N_EMBD,   FFN_DIM,  0.02 / math.sqrt(N_LAYER))
        sd[f"l{i}.ns1"] = _nv(N_EMBD)
        sd[f"l{i}.ns2"] = _nv(N_EMBD)
    sd["ns_f"] = _nv(N_EMBD)
    return sd

def zero_grads(sd): return {k: np.zeros_like(v) for k, v in sd.items()}

# ==============================================================================
#  MATH PRIMITIVES
# ==============================================================================
def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(-1, keepdims=True) + 1e-9)

def _rmsnorm(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)

def _rmsnorm_bwd(dy, x):
    rms = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    y   = x / rms
    return (dy - y * (dy * y).mean(-1, keepdims=True)) / rms

_CAUSAL_CACHE = {}
def _causal_mask(T):
    if T not in _CAUSAL_CACHE:
        _CAUSAL_CACHE[T] = np.triu(np.ones((T, T), dtype=bool), k=1)
    return _CAUSAL_CACHE[T]

# ==============================================================================
#  FORWARD + BACKWARD   (SwiGLU edition)
# ==============================================================================
def forward_backward(ids, sd, grads, smooth=LABEL_SMOOTH):
    n = min(len(ids) - 1, BLOCK_SIZE)
    if n < 1:
        return None

    inp  = np.array(ids[:n],    dtype=np.int32)
    tgt  = np.array(ids[1:n+1], dtype=np.int32)
    T    = n
    mask = _causal_mask(T)
    sc   = HEAD_DIM ** -0.5

    x    = sd["wte"][inp] + sd["wpe"][np.arange(T) % BLOCK_SIZE]
    saves = []

    for li in range(N_LAYER):
        x_in  = x
        xn_raw = _rmsnorm(x_in)
        xn_s   = xn_raw * sd[f"l{li}.ns1"]

        Q = xn_s @ sd[f"l{li}.wq"].T
        K = xn_s @ sd[f"l{li}.wk"].T
        V = xn_s @ sd[f"l{li}.wv"].T

        Q3 = Q.reshape(T, N_HEAD, HEAD_DIM)
        K3 = K.reshape(T, N_HEAD, HEAD_DIM)
        V3 = V.reshape(T, N_HEAD, HEAD_DIM)

        attn = np.einsum("thd,shd->hts", Q3, K3) * sc
        attn[:, mask] = -1e9
        aw = _softmax(attn)

        ao  = np.einsum("hts,shd->thd", aw, V3).reshape(T, N_EMBD)
        ap  = ao @ sd[f"l{li}.wo"].T
        x_a = x_in + ap

        xn2_raw = _rmsnorm(x_a)
        xn2_s   = xn2_raw * sd[f"l{li}.ns2"]

        # ── SwiGLU FFN ──────────────────────────────────────────────
        gate_pre = xn2_s @ sd[f"l{li}.wg"].T   # [T, FFN_DIM]
        val_pre  = xn2_s @ sd[f"l{li}.w1"].T   # [T, FFN_DIM]
        sig      = 1.0 / (1.0 + np.exp(-np.clip(gate_pre, -20, 20)))
        h1       = val_pre * (gate_pre * sig)   # SwiGLU(gate) × val
        h2       = h1 @ sd[f"l{li}.w2"].T       # [T, N_EMBD]
        x        = x_a + h2

        saves.append((x_in, xn_raw, xn_s, Q3, K3, V3, aw, ao,
                      x_a, xn2_raw, xn2_s, gate_pre, sig, val_pre, h1))

    xf     = _rmsnorm(x) * sd["ns_f"]
    logits = xf @ sd["wte"].T

    probs  = _softmax(logits)
    tgt_sm = np.full_like(probs, smooth / (VOCAB_SIZE - 1))
    tgt_sm[np.arange(T), tgt] = 1.0 - smooth
    loss   = -(tgt_sm * np.log(probs + 1e-9)).sum(-1).mean()

    # ── Backward ────────────────────────────────────────────────────
    d_logits = (probs - tgt_sm) / T
    grads["wte"] += d_logits.T @ xf

    xf_raw  = _rmsnorm(x)
    d_xf    = (d_logits @ sd["wte"]) * sd["ns_f"]
    grads["ns_f"] += (d_logits @ sd["wte"] * xf_raw).sum(0)
    dx = _rmsnorm_bwd(d_xf, x)

    for li in reversed(range(N_LAYER)):
        (x_in, xn_raw, xn_s, Q3, K3, V3, aw, ao,
         x_a, xn2_raw, xn2_s, gate_pre, sig, val_pre, h1) = saves[li]

        # ── SwiGLU backward ─────────────────────────────────────────
        d_h2 = dx
        grads[f"l{li}.w2"] += d_h2.T @ h1          # [N_EMBD, FFN_DIM]
        d_h1 = d_h2 @ sd[f"l{li}.w2"]              # [T, FFN_DIM]

        swish         = gate_pre * sig              # swish(gate_pre)
        d_val_pre     = d_h1 * swish                # ∂L/∂val_pre
        d_swish_dgate = sig * (1.0 + gate_pre * (1.0 - sig))
        d_gate_pre    = d_h1 * val_pre * d_swish_dgate

        grads[f"l{li}.w1"]  += d_val_pre.T  @ xn2_s
        grads[f"l{li}.wg"]  += d_gate_pre.T @ xn2_s
        d_xn2 = d_val_pre @ sd[f"l{li}.w1"] + d_gate_pre @ sd[f"l{li}.wg"]

        grads[f"l{li}.ns2"] += (d_xn2 * xn2_raw).sum(0)
        d_xa = dx + _rmsnorm_bwd(d_xn2 * sd[f"l{li}.ns2"], x_a)

        # ── Attention backward ──────────────────────────────────────
        grads[f"l{li}.wo"] += d_xa.T @ ao
        d_ao   = d_xa @ sd[f"l{li}.wo"]
        d_out3 = d_ao.reshape(T, N_HEAD, HEAD_DIM)

        d_aw   = np.einsum("thd,shd->hts", d_out3, V3)
        d_attn = aw * (d_aw - (d_aw * aw).sum(-1, keepdims=True)) * sc
        d_attn[:, mask] = 0.0

        dQ = np.einsum("hts,shd->thd", d_attn, K3).reshape(T, N_EMBD)
        dK = np.einsum("hts,thd->shd", d_attn, Q3).reshape(T, N_EMBD)
        dV = np.einsum("hts,thd->shd", aw, d_out3).reshape(T, N_EMBD)

        grads[f"l{li}.wq"] += dQ.T @ xn_s
        grads[f"l{li}.wk"] += dK.T @ xn_s
        grads[f"l{li}.wv"] += dV.T @ xn_s

        d_xn_s = dQ @ sd[f"l{li}.wq"] + dK @ sd[f"l{li}.wk"] + dV @ sd[f"l{li}.wv"]
        grads[f"l{li}.ns1"] += (d_xn_s * xn_raw).sum(0)
        dx = d_xa + _rmsnorm_bwd(d_xn_s * sd[f"l{li}.ns1"], x_in)

    np.add.at(grads["wte"], inp, dx)
    np.add.at(grads["wpe"], np.arange(T) % BLOCK_SIZE, dx)
    return float(loss)

# ==============================================================================
#  OPTIMIZER  — AdamW with cosine LR + warmup
# ==============================================================================
def lr_schedule(step):
    if step < LR_WARMUP:
        return LR_BASE * (step + 1) / LR_WARMUP
    prog = (step - LR_WARMUP) / max(1, TOTAL_STEPS - LR_WARMUP)
    return max(LR_BASE * LR_FLOOR, LR_BASE * 0.5 * (1 + math.cos(math.pi * prog)))

def adamw_step(sd, grads, step, am, av, t_cnt):
    lr = lr_schedule(step)
    t_cnt[0] += 1
    t = t_cnt[0]
    bc1 = 1.0 - BETA1 ** t
    bc2 = 1.0 - BETA2 ** t
    for key in sd:
        g = np.clip(grads[key], -GRAD_CLIP, GRAD_CLIP)
        sd[key]   *= (1.0 - lr * WEIGHT_DECAY)
        am[key]    = BETA1 * am[key] + (1.0 - BETA1) * g
        av[key]    = BETA2 * av[key] + (1.0 - BETA2) * g * g
        sd[key]   -= lr * (am[key] / bc1) / (np.sqrt(av[key] / bc2) + EPS_ADAM)
        grads[key][:] = 0.0

# ==============================================================================
#  WEIGHTS
# ==============================================================================
SD = build_model()
AM = zero_grads(SD)
AV = zero_grads(SD)
AT = [0]

def save_weights():
    pickle.dump({"sd": SD, "am": AM, "av": AV, "at": AT[0]},
                open(WEIGHTS_PKL, "wb"))

def load_weights():
    global SD, AM, AV, AT
    try:
        ck = pickle.load(open(WEIGHTS_PKL, "rb"))
        if isinstance(ck, dict) and "sd" in ck:
            loaded_sd = ck["sd"]
            # Check compatible (SwiGLU wg key present and shapes match)
            if (f"l0.wg" in loaded_sd and
                    loaded_sd["wte"].shape == SD["wte"].shape and
                    loaded_sd[f"l0.wg"].shape == SD[f"l0.wg"].shape):
                SD.update(loaded_sd)
                AM.update(ck.get("am", AM))
                AV.update(ck.get("av", AV))
                AT[0] = ck.get("at", 0)
                return True
    except Exception:
        pass
    return False

def load_progress():
    try:    return int(open(PROG_FILE).read().strip())
    except: return 0

def save_progress(step):
    open(PROG_FILE, "w").write(str(step))

# ==============================================================================
#  TRAINING STATUS
# ==============================================================================
training_status = {
    "running": False, "done": False,
    "step": load_progress(), "total": TOTAL_STEPS,
    "loss": None, "pct": min(100, int(load_progress() / TOTAL_STEPS * 100)),
    "dataset": {},
}

def _build_training_corpus():
    dset = _dataset_pairs()
    mem  = load_memory()
    wf   = _word_facts()
    wf_pairs = [f"what does {w} mean => {m}" for w, m in list(wf.items())[:400]]
    corpus = mem + dset + wf_pairs
    if not corpus:
        corpus = ["hello => hi there, how can i help you today"]
    random.Random(7).shuffle(corpus)
    return corpus

def pretrain_chunk(status_dict=None):
    if status_dict is None:
        status_dict = training_status

    start = load_progress()
    if start >= TOTAL_STEPS:
        ds = _get_dataset()
        status_dict.update(
            running=False, done=True, step=TOTAL_STEPS, pct=100,
            dataset={"pairs": ds.get("pair_count", len(_dataset_pairs())),
                     "topics": ds.get("topic_count", 0),
                     "word_facts": len(_word_facts())},
        )
        return

    corpus = _build_training_corpus()
    ds     = _get_dataset()
    status_dict.update(
        running=True, done=False, total=TOTAL_STEPS,
        dataset={"pairs": ds.get("pair_count", len(_dataset_pairs())),
                 "topics": ds.get("topic_count", 0),
                 "word_facts": len(_word_facts())},
    )

    grads       = zero_grads(SD)
    recent_loss = []
    end         = min(start + STEPS_PER_CHUNK, TOTAL_STEPS)

    for step in range(start, end):
        doc  = corpus[step % len(corpus)]
        loss = forward_backward(tokenise(doc), SD, grads)
        if loss is None:
            save_progress(step + 1)
            continue

        adamw_step(SD, grads, step, AM, AV, AT)
        recent_loss.append(loss)
        if len(recent_loss) > 200:
            recent_loss.pop(0)
        avg = sum(recent_loss) / len(recent_loss)

        status_dict["step"] = step + 1
        status_dict["pct"]  = int((step + 1) / TOTAL_STEPS * 100)
        status_dict["loss"] = round(avg, 4)

        if (step + 1) % 100 == 0:
            save_progress(step + 1)
            save_weights()
            with open(LOSS_LOG, "a") as f:
                f.write(f"{step+1},{avg:.4f}\n")

    save_progress(end)
    save_weights()
    status_dict.update(
        step=end,
        pct=int(end / TOTAL_STEPS * 100),
        running=end < TOTAL_STEPS,
        done=end >= TOTAL_STEPS,
    )

# ==============================================================================
#  STOPWORDS
# ==============================================================================
_STOPWORDS = {
    "i","the","a","an","is","are","it","to","in","of","and","that",
    "you","me","my","we","do","be","this","for","on","at","its","was",
    "were","what","how","why","who","where","when","can","could","will",
    "would","tell","about","please","does","did","explain","describe",
    "define","meaning","give","some","just","very","really","want","know",
    "get","let","make","take","use","has","had","have","with","from","by",
    "or","but","not","no","up","out","so","if","then","than","into","onto",
}

def _content_words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in _STOPWORDS and len(w) > 2]

# ==============================================================================
#  WORD / SENTENCE VECTORS  (L2-normalised for better cosine similarity)
# ==============================================================================
def _word_vector(word):
    ids = tokenise(word)[1:-1]
    if not ids:
        return None
    v = SD["wte"][np.array(ids)].mean(0)
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9) if norm > 1e-9 else v

_IDF_CACHE     = None
_IDF_CACHE_KEY = None

def _get_idf():
    global _IDF_CACHE, _IDF_CACHE_KEY
    pairs = _all_pairs()
    key   = len(pairs)
    if key == _IDF_CACHE_KEY and _IDF_CACHE is not None:
        return _IDF_CACHE
    df = Counter()
    for line in pairs:
        if "=>" not in line:
            continue
        q = line.split("=>", 1)[0]
        for w in set(_content_words(q)):
            df[w] += 1
    N   = len(pairs) + 1
    idf = {w: math.log(N / (c + 1)) for w, c in df.items()}
    _IDF_CACHE     = idf
    _IDF_CACHE_KEY = key
    return idf

def _sentence_vector(text):
    """IDF-weighted, L2-normalised sentence vector."""
    words = _content_words(text)
    if not words:
        ids = tokenise(text)[1:-1]
        if not ids:
            return None
        v = SD["wte"][np.array(ids)].mean(0)
        n = np.linalg.norm(v)
        return v / (n + 1e-9) if n > 1e-9 else None

    idf = _get_idf()
    vecs, weights = [], []
    for w in dict.fromkeys(words):
        v = _word_vector(w)
        if v is not None:
            wt = idf.get(w, 1.0) * min(1.5, len(w) / 5.0)
            vecs.append(v); weights.append(wt)

    if not vecs:
        return None
    w_arr = np.array(weights, dtype=np.float32)
    w_arr /= w_arr.sum() + 1e-9
    agg = (np.stack(vecs) * w_arr[:, None]).sum(0)
    norm = np.linalg.norm(agg)
    return agg / (norm + 1e-9) if norm > 1e-9 else None

def _cosine(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))   # already L2-normalised → dot == cosine

# ==============================================================================
#  BM25 INDEX  — rebuilt only when pairs change
# ==============================================================================
_INDEX_CACHE     = None
_INDEX_CACHE_KEY = None

def _get_index():
    global _INDEX_CACHE, _INDEX_CACHE_KEY
    pairs = _all_pairs()
    key   = len(pairs)
    if key == _INDEX_CACHE_KEY and _INDEX_CACHE is not None:
        return _INDEX_CACHE

    questions, answers, doc_tfs, doc_lens = [], [], [], []
    for line in pairs:
        if "=>" not in line:
            continue
        q, a = line.split("=>", 1)
        q = q.strip().lower(); a = a.strip()
        words = _content_words(q)
        questions.append(q)
        answers.append(a)
        tf = Counter(words)
        doc_tfs.append(tf)
        doc_lens.append(len(words))

    N      = len(questions) + 1
    avgdl  = (sum(doc_lens) / N) if doc_lens else 1.0
    df     = Counter()
    for tf in doc_tfs:
        df.update(tf.keys())
    idf    = {w: math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1.0)
              for w in df}           # Robertson BM25 IDF

    _INDEX_CACHE     = (questions, answers, idf, doc_tfs, doc_lens, avgdl)
    _INDEX_CACHE_KEY = key
    return _INDEX_CACHE

# ==============================================================================
#  RETRIEVAL  — BM25 + semantic + context boost
# ==============================================================================
def _training_ratio():
    return min(1.0, load_progress() / max(1, TOTAL_STEPS))

def retrieve(user_text, threshold=0.38, ctx_words=None):
    questions, answers, idf, doc_tfs, doc_lens, avgdl = _get_index()
    qwords  = _content_words(user_text)
    ctx     = ctx_words or set()
    sem_w   = _training_ratio() * 0.35
    user_vec = _sentence_vector(user_text) if sem_w > 0.05 else None

    # BM25 query term weights
    q_set     = set(qwords)
    ul_words  = set(user_text.lower().split())
    ul_bg     = set(user_text.lower()[j:j+3]
                    for j in range(max(0, len(user_text)-2)))

    best_s, best_a = 0.0, None

    for i, (q, a) in enumerate(zip(questions, answers)):
        if len(a) < 15:
            continue
        dl = doc_lens[i]
        tf = doc_tfs[i]

        # ── BM25 score ──────────────────────────────────────────────
        bm25 = 0.0
        for w in q_set:
            if w in tf and w in idf:
                f    = tf[w]
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / (avgdl + 1e-9))
                bm25 += idf[w] * (f * (BM25_K1 + 1)) / (denom + 1e-9)

        bm25_n = min(1.0, bm25 / (len(qwords) + 1e-9) * 0.8)

        # ── Word overlap ─────────────────────────────────────────────
        q_words_set = set(q.split())
        union_w     = q_words_set | ul_words
        ovlp        = len(q_words_set & ul_words) / (len(union_w) + 1e-9)

        # ── Trigram overlap ──────────────────────────────────────────
        q_bg      = set(q[j:j+3] for j in range(max(0, len(q)-2)))
        union_bg  = q_bg | ul_bg
        ng_s      = len(q_bg & ul_bg) / (len(union_bg) + 1e-9)

        # ── Context boost ────────────────────────────────────────────
        ctx_b = sum(0.05 for cw in ctx if cw in q and len(cw) > 3)

        text_s = 0.55 * bm25_n + 0.27 * ovlp + 0.18 * ng_s

        if sem_w > 0.05 and user_vec is not None:
            q_vec = _sentence_vector(q)
            sem_s = _cosine(user_vec, q_vec)
            # word_assoc similarity bonus
            _wa = _get_wa()
            if _wa is not None:
                try:
                    w2v_s = _wa.semantic_similarity(user_text, q)
                    sem_s = 0.80 * sem_s + 0.20 * max(0.0, w2v_s)
                except Exception:
                    pass
            score = (1.0 - sem_w) * text_s + sem_w * sem_s + ctx_b
        else:
            score = text_s + ctx_b

        if score > best_s:
            best_s, best_a = score, a

    return (best_a, best_s) if best_s >= threshold else (None, best_s)

# ==============================================================================
#  WORD ASSOCIATION  (lazy-load word_assoc.py)
# ==============================================================================
_WA = None

def _get_wa():
    global _WA
    if _WA is None:
        try:
            import word_assoc as _wa_mod
            _WA = _wa_mod
        except ImportError:
            _WA = False
    return _WA if _WA else None

# ==============================================================================
#  WORD-MEANING COMPOSITION
# ==============================================================================
def _find_word_meaning(word):
    wf = _word_facts()
    if word in wf:
        m = wf[word]
        for s in re.split(r"[.;]", m):
            if word in s.lower():
                return s.strip().rstrip(".")
        return m.split(".")[0].strip()
    for line in _all_pairs():
        if "=>" not in line:
            continue
        q, a = line.split("=>", 1)
        if word in q.lower() and len(a.strip()) > 20:
            return a.strip()
    return None

def _compose_answer_from_words(sentence):
    words = _content_words(sentence)
    if not words:
        return None
    for w in words[:5]:
        m = _find_word_meaning(w)
        if m and len(m) > 18:
            return f"{w}: {m}"
    return None

# ==============================================================================
#  INTERNET LEARNING
# ==============================================================================
_WIKI_CACHE: dict = {}

def fetch_wiki(topic: str, max_chars: int = 700) -> str | None:
    topic = topic.strip()
    if topic in _WIKI_CACHE:
        return _WIKI_CACHE[topic]
    try:
        q   = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "MicroBot/7.0 (educational)"})
        with urllib.request.urlopen(req, timeout=8) as r:
            if r.status == 200:
                d = json.loads(r.read().decode("utf-8", errors="replace"))
                extract = d.get("extract", "").strip()
                if extract:
                    _WIKI_CACHE[topic] = extract[:max_chars]
                    return _WIKI_CACHE[topic]
    except Exception:
        pass
    return None

def learn_from_web(topic: str) -> str | None:
    text = fetch_wiki(topic)
    if not text:
        return None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.strip()) > 20]
    q     = f"what is {topic.lower().strip()}"
    first = None
    for sent in sentences[:3]:
        a = sent.lower().rstrip(".!?,;").strip()
        if len(a) > 15:
            learn_pair(q, a, steps=10)
            if first is None:
                first = a
    return first

def extract_topic(text: str) -> str | None:
    text = text.lower().strip().rstrip("?!.,")
    for prefix in [
        "what is", "what are", "what's", "whats",
        "who is", "who are", "who was", "where is",
        "tell me about", "explain", "describe", "define",
        "how does", "how do", "give me information about",
    ]:
        if text.startswith(prefix):
            topic = text[len(prefix):].strip()
            if topic:
                return topic
    stop = {"the","a","an","is","are","do","does","did","have","has",
            "will","could","i","you","me","we","it","this","that","please"}
    words = [w for w in re.findall(r"[a-z]+", text) if w not in stop]
    return " ".join(words[-3:]) if words else None

# ==============================================================================
#  WORD EMBEDDING  (for CLI plot_vectors)
# ==============================================================================
def word_embedding(word):
    ids = tokenise(word)[1:-1]
    if not ids:
        return None
    return SD["wte"][np.array(ids)].mean(0).tolist()

# ==============================================================================
#  GENERATION  — top-K + temperature (more deterministic than top-p)
# ==============================================================================
def _topk_sample(logits, k=TOP_K, temp=TEMPERATURE):
    logits = logits / temp
    if k > 0:
        idx    = np.argpartition(logits, -k)[-k:]
        logits_k = logits[idx]
    else:
        idx      = np.arange(len(logits))
        logits_k = logits
    logits_k -= logits_k.max()
    probs = np.exp(logits_k)
    probs /= probs.sum() + 1e-9
    return int(idx[np.random.choice(len(idx), p=probs)])

def generate(prompt_text, max_new=70):
    tokens = tokenise(prompt_text)
    ck = [[] for _ in range(N_LAYER)]
    cv = [[] for _ in range(N_LAYER)]
    sc = HEAD_DIM ** -0.5

    def _step(tid, pos):
        x = SD["wte"][tid] + SD["wpe"][pos % BLOCK_SIZE]
        for li in range(N_LAYER):
            xn_r = _rmsnorm(x[None, :])[0]
            xn_s = xn_r * SD[f"l{li}.ns1"]
            q, k, v = xn_s @ SD[f"l{li}.wq"].T, xn_s @ SD[f"l{li}.wk"].T, xn_s @ SD[f"l{li}.wv"].T
            ck[li].append(k); cv[li].append(v)
            win = min(len(ck[li]), 8)
            K   = np.stack(ck[li][-win:])
            V   = np.stack(cv[li][-win:])
            q3  = q.reshape(N_HEAD, HEAD_DIM)
            K3  = K.reshape(win, N_HEAD, HEAD_DIM)
            V3  = V.reshape(win, N_HEAD, HEAD_DIM)
            aw  = _softmax(np.einsum("hd,whd->hw", q3, K3) * sc)
            ao  = np.einsum("hw,whd->hd", aw, V3).reshape(N_EMBD)
            x   = x + ao @ SD[f"l{li}.wo"].T
            xn2_r = _rmsnorm(x[None, :])[0]
            xn2_s = xn2_r * SD[f"l{li}.ns2"]
            # SwiGLU
            gp  = xn2_s @ SD[f"l{li}.wg"].T
            vp  = xn2_s @ SD[f"l{li}.w1"].T
            sg  = 1.0 / (1.0 + np.exp(-np.clip(gp, -20, 20)))
            h1  = vp * (gp * sg)
            x   = x + h1 @ SD[f"l{li}.w2"].T
        xf = _rmsnorm(x[None, :])[0] * SD["ns_f"]
        return SD["wte"] @ xf    # raw logits

    for pos, tid in enumerate(tokens[1:-1]):
        _step(tid, pos)

    out, tid = [], BOS
    for s in range(max_new):
        logits = _step(tid, len(tokens) - 1 + s)
        tid    = _topk_sample(logits)
        if tid == BOS:
            break
        out.append(itos[tid])
    return "".join(out).strip()

# ==============================================================================
#  ONLINE LEARNING
# ==============================================================================
REPLAY      = deque(maxlen=128)
ONLINE_STEP = [load_progress()]

def learn_pair(q, a, steps=15):
    global _PAIRS_CACHE, _PAIRS_CACHE_KEY, _INDEX_CACHE, _INDEX_CACHE_KEY, _IDF_CACHE
    pair  = f"{q.strip().lower()} => {a.strip().lower()}"
    pairs = load_memory()
    if pair not in pairs:
        pairs.append(pair)
        save_memory(pairs)
        _PAIRS_CACHE = _INDEX_CACHE = _IDF_CACHE = None

    REPLAY.append(pair)
    grads = zero_grads(SD)
    for s in range(steps):
        forward_backward(tokenise(pair), SD, grads)
        adamw_step(SD, grads, ONLINE_STEP[0] + s, AM, AV, AT)
    ONLINE_STEP[0] += steps

    if len(REPLAY) >= 4:
        for rp in random.sample(list(REPLAY), min(3, len(REPLAY))):
            forward_backward(tokenise(rp), SD, grads)
            adamw_step(SD, grads, ONLINE_STEP[0], AM, AV, AT)
            ONLINE_STEP[0] += 1
    save_weights()

# ==============================================================================
#  EMOTION DETECTION
# ==============================================================================
_EMOTION_LEX = {
    "joy":      {"happy","joyful","glad","excited","love","great","wonderful","smile","laugh","enjoy","amazing","fantastic","brilliant"},
    "sadness":  {"sad","unhappy","cry","pain","lonely","miss","lost","sorry","grief","depressed","hurt","alone","miserable","hopeless"},
    "anger":    {"angry","mad","hate","furious","annoyed","frustrated","rage","upset","irritate"},
    "fear":     {"fear","scared","worry","anxious","nervous","afraid","stress","panic","dread","terrified"},
    "surprise": {"surprised","shocked","amazed","wow","incredible","unexpected","unbelievable"},
    "curiosity":{"curious","wonder","interesting","learn","discover","explore","understand","mystery"},
}
_EMOTION_RSP = {
    "joy":      ["that's wonderful! ","great to hear! ",""],
    "sadness":  ["i understand that feels hard. ","i'm here with you. ",""],
    "anger":    ["i hear you. ","let's work through this. ",""],
    "fear":     ["it's okay to feel this way. ","one step at a time. ",""],
    "surprise": ["wow indeed! ","that's quite something! ",""],
    "curiosity":["great question! ","let's explore that! ",""],
    "neutral":  [""],
}

def detect_emotion(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    scores = {e: len(words & kws) for e, kws in _EMOTION_LEX.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"

def emotion_prefix(e):
    return random.choice(_EMOTION_RSP.get(e, [""]))

# ==============================================================================
#  CONVERSATION CONTEXT
# ==============================================================================
_CTX: deque = deque(maxlen=6)

def push_context(user: str, bot: str):
    _CTX.append((user.lower().strip(), bot.lower().strip()))

def context_words() -> set:
    words = set()
    for u, b in _CTX:
        words.update(re.findall(r"[a-z]+", u + " " + b))
    return words - _STOPWORDS

# ==============================================================================
#  PARAM INFO
# ==============================================================================
def neuron_info() -> dict:
    total = (VOCAB_SIZE * N_EMBD + BLOCK_SIZE * N_EMBD
             + N_LAYER * (4 * N_EMBD * N_EMBD    # wq wk wv wo
                          + 2 * FFN_DIM * N_EMBD  # wg w1
                          + N_EMBD * FFN_DIM       # w2
                          + 2 * N_EMBD)            # ns1 ns2
             + N_EMBD)
    return dict(total_params=total, n_layer=N_LAYER, n_head=N_HEAD,
                embed_dim=N_EMBD, ffn_dim=FFN_DIM,
                block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE)

# ==============================================================================
#  PUBLIC API
# ==============================================================================
def handle_message(user: str) -> str:
    ul = user.lower().strip()

    if ul in ("quit", "exit", "q"):
        return "goodbye! see you next time."

    if ul.startswith("plot"):
        return "plot commands are only available in the CLI (microbot.py)."

    if ul.startswith("teach:"):
        body = user[6:].strip()
        if "=>" in body:
            q, a = body.split("=>", 1)
            learn_pair(q.strip(), a.strip(), steps=30)
            return f"learned! '{q.strip()}' → '{a.strip()}'"
        return "format:  teach: question => answer"

    if ul.startswith("learn:"):
        topic = user[6:].strip()
        if not topic:
            return "format:  learn: topic"
        result = learn_from_web(topic)
        if result:
            return f"i just learned about '{topic}': {result}"
        return (f"couldn't reach wikipedia for '{topic}'. "
                f"teach me:  teach: {topic} => your answer")

    if ul.startswith("assoc:") or ul.startswith("similar:"):
        word = ul.replace("assoc:", "").replace("similar:", "").strip()
        wa   = _get_wa()
        if wa is None:
            return "word association module not loaded (word_assoc.py missing)"
        sims = wa.similar_words(word, n=10)
        if not sims:
            return f"'{word}' not in association vocabulary"
        return "words associated with '{}': {}".format(
            word, ", ".join(f"{w}({s:.2f})" for w, s in sims[:8]))

    if any(k in ul for k in ("how many param","how big","your size","your params","how many neuron")):
        info = neuron_info()
        return (f"i have {info['total_params']:,} parameters across "
                f"{info['n_layer']} transformer layers "
                f"(embd={info['embed_dim']}, ffn_swiglu={info['ffn_dim']}, "
                f"heads={info['n_head']}).")

    if ul.startswith("dataset"):
        ds = _get_dataset()
        p  = ds.get("pair_count", len(_dataset_pairs()))
        t  = ds.get("topic_count", 0)
        wf = len(_word_facts())
        r  = int(_training_ratio() * 100)
        return f"dataset: {p} qa pairs / {t} topics / {wf} word-facts. training: {r}% complete."

    emotion = detect_emotion(ul)
    prefix  = emotion_prefix(emotion)
    ctx     = context_words()

    # word_assoc query expansion
    wa = _get_wa()
    if wa is not None:
        try:
            expanded = wa.expand_query(ul, n=3, threshold=0.25)
            ctx = ctx | expanded
        except Exception:
            pass

    # Primary BM25+semantic retrieval
    answer, score = retrieve(user, ctx_words=ctx)

    # Context-boosted retry
    if not answer and ctx:
        ctx_str = " ".join(w for w in ctx if len(w) > 3)
        answer, score = retrieve(user + " " + ctx_str, threshold=0.30)

    if answer:
        learn_pair(user, answer, steps=4)
        push_context(user, answer)
        return prefix + answer

    # Word-by-word meaning composition
    composed = _compose_answer_from_words(ul)
    if composed and len(composed) > 20:
        learn_pair(user, composed, steps=6)
        push_context(user, composed)
        return prefix + composed

    # Topic index lookup
    topic = extract_topic(ul)
    if topic and len(topic) >= 3:
        summary = _topic_summary(topic)
        if summary and len(summary) > 20:
            ans = summary.lower().rstrip(".")
            learn_pair(user, ans, steps=8)
            push_context(user, ans)
            return prefix + ans

        web_ans = learn_from_web(topic)
        if web_ans:
            push_context(user, web_ans)
            return prefix + web_ans

    # Neural generation (after significant training)
    if _training_ratio() > 0.20:
        reply = generate(user)
        if len(reply) >= 8:
            learn_pair(user, reply, steps=6)
            push_context(user, reply)
            return prefix + reply

    # Word-fact fallback
    words = _content_words(ul)
    for w in words:
        m = _find_word_meaning(w)
        if m and len(m) > 15:
            reply = f"about '{w}': {m}"
            learn_pair(user, reply, steps=4)
            push_context(user, reply)
            return prefix + reply

    # Web last resort
    if words:
        web_ans = learn_from_web(" ".join(words[:2]))
        if web_ans:
            push_context(user, web_ans)
            return prefix + web_ans

    return (prefix + "i don't have that yet. try:\n"
            f"  learn: {' '.join(words[:2]) if words else 'topic'}   "
            "or   teach: your question => your answer")
