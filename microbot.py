"""
microbot.py  -  MicroBot v7  CLI
=================================
Run this file. microbot_core.py, dataset.py, word_assoc.py must be in the same folder.

  python microbot.py

First run: fetches dataset from Wikipedia (~100 topics) — needs internet once.
Subsequent runs: uses cached dataset; no internet unless you use 'learn:'.
"""
import os, sys, re, time

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def _safe_input(prompt=""):
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        line = sys.stdin.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\n")
    except Exception:
        raise EOFError

if not sys.stdin.isatty():
    input = _safe_input  # noqa

try:
    from microbot_core import (
        handle_message, pretrain_chunk, training_status,
        load_progress, save_progress, TOTAL_STEPS,
        load_weights, WEIGHTS_PKL,
        word_embedding, LOSS_LOG, N_EMBD, FFN_DIM,
        _get_dataset, _dataset_pairs, _word_facts,
        neuron_info,
    )
except ImportError as e:
    print(f"ERROR: cannot import microbot_core.py ({e})")
    print("Make sure microbot_core.py and dataset.py are in the same folder.")
    sys.exit(1)

# ==============================================================================
#  LOAD WEIGHTS
# ==============================================================================
import os as _os
if _os.path.exists(WEIGHTS_PKL):
    ok = load_weights()
    if ok:
        print("weights loaded")
    else:
        print("weights format mismatch — starting fresh (SwiGLU v7 model)")

# ==============================================================================
#  PLOTTING
# ==============================================================================
def plot_loss():
    try:
        import matplotlib.pyplot as plt
        if not os.path.exists(LOSS_LOG):
            print("no loss data yet"); return
        steps, losses = [], []
        for line in open(LOSS_LOG):
            parts = line.strip().split(",")
            if len(parts) == 2:
                steps.append(int(parts[0])); losses.append(float(parts[1]))
        if not steps:
            print("no loss data"); return
        w = 10
        smoothed = [sum(losses[max(0,i-w):i+1]) / len(losses[max(0,i-w):i+1])
                    for i in range(len(losses))]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses, alpha=0.3, color="steelblue", label="raw loss")
        plt.plot(steps, smoothed, color="steelblue", lw=2, label="smoothed")
        plt.xlabel("Step"); plt.ylabel("Loss")
        plt.title("MicroBot v7 Training Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig("loss_plot.png", dpi=120)
        print("saved: loss_plot.png")
    except Exception as e:
        print(f"plot failed: {e}")

def plot_vectors(words=None):
    import math as _math, random as _random
    try:
        import matplotlib.pyplot as plt
        if words is None:
            words = ["hello","love","ai","science","earth","food",
                     "robot","happy","sad","learn","water","time",
                     "python","brain","code","star","fast",
                     "neural","language","energy","light","gravity"]
        embs, valid_words = [], []
        for w in words:
            e = word_embedding(w)
            if e:
                embs.append(e); valid_words.append(w)
        if len(embs) < 3:
            print("not enough words for plot"); return

        n, d = len(embs), len(embs[0])
        mean = [sum(embs[i][j] for i in range(n)) / n for j in range(d)]
        X    = [[embs[i][j] - mean[j] for j in range(d)] for i in range(n)]

        def cov_vec(X, v):
            Xv = [sum(X[i][j]*v[j] for j in range(d)) for i in range(n)]
            return [sum(X[i][j]*Xv[i] for i in range(n))/n for j in range(d)]

        def normalize(v):
            norm = _math.sqrt(sum(x*x for x in v)) + 1e-9
            return [x/norm for x in v]

        def power_iter(X, iters=30):
            v = [_random.gauss(0, 1) for _ in range(d)]
            v = normalize(v)
            for _ in range(iters):
                v = cov_vec(X, v); v = normalize(v)
            return v

        pc1 = power_iter(X)
        X2  = [[X[i][j] - sum(X[i][k]*pc1[k] for k in range(d))*pc1[j]
                for j in range(d)] for i in range(n)]
        pc2 = power_iter(X2)
        p1  = [sum(X[i][j]*pc1[j] for j in range(d)) for i in range(n)]
        p2  = [sum(X[i][j]*pc2[j] for j in range(d)) for i in range(n)]

        plt.figure(figsize=(9, 6))
        plt.scatter(p1, p2, s=60, alpha=0.7, color="steelblue")
        for i, w in enumerate(valid_words):
            plt.annotate(w, (p1[i], p2[i]), fontsize=9,
                         xytext=(4,4), textcoords="offset points")
        plt.title("MicroBot v7 Word Embeddings (PCA 2D)")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
        plt.savefig("vectors_plot.png", dpi=120)
        print("saved: vectors_plot.png")
    except Exception as e:
        print(f"vector plot failed: {e}")

# ==============================================================================
#  STARTUP
# ==============================================================================
ds          = _get_dataset()
pair_count  = ds.get("pair_count", len(_dataset_pairs()))
topic_count = ds.get("topic_count", 0)
wf_count    = len(_word_facts())

info = neuron_info()
print(f"\nMicroBot v7  |  {info['total_params']:,} params  "
      f"embd={N_EMBD}  ffn_swiglu={FFN_DIM}  "
      f"dataset: {pair_count} pairs / {topic_count} topics / {wf_count} word-facts")

start = load_progress()
pct   = min(100, int(start / TOTAL_STEPS * 100))

if start < TOTAL_STEPS:
    print(f"training {pct}% done ({start}/{TOTAL_STEPS} steps) — running next chunk...")
    t0 = time.time()
    pretrain_chunk(training_status)
    elapsed = time.time() - t0
    new_pct = training_status["pct"]
    loss    = training_status.get("loss", "?")
    print(f"chunk done [{new_pct}%]  loss={loss}  time={elapsed:.1f}s\n")
else:
    print("fully trained\n")

done   = load_progress()
pct    = min(100, int(done / TOTAL_STEPS * 100))
status = "fully trained" if done >= TOTAL_STEPS else f"{pct}% trained"
print(f"MicroBot ready [{status}]")
print("commands:")
print("  quit                          - exit")
print("  teach: question => answer     - teach a pair manually")
print("  learn: topic                  - fetch wikipedia and learn it")
print("  assoc: word                   - show word associations")
print("  dataset                       - show dataset info")
print("  plot loss                     - training loss curve")
print("  plot vectors [w1 w2 ...]      - word embedding space\n")

while True:
    try:
        user = input("you: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nbye!"); break

    if not user:
        continue

    ul = user.lower()

    if ul in ("quit", "exit", "q"):
        print("bot: goodbye! see you next time."); break

    if ul.startswith("plot loss"):
        plot_loss(); continue

    if ul.startswith("plot vectors"):
        parts = ul.replace("plot vectors", "").strip().split()
        plot_vectors(parts if parts else None)
        continue

    reply = handle_message(user)
    print(f"bot: {reply}")

    if not ul.startswith(("teach:", "learn:", "assoc:", "dataset", "plot")):
        print("     [correction or Enter to skip]")
        try:
            correction = input("     > ").strip()
        except (EOFError, KeyboardInterrupt):
            correction = ""
        if correction:
            from microbot_core import learn_pair
            learn_pair(user, correction, steps=30)
            print("bot: got it, learned!")

    print()
