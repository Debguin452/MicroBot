"""
word_assoc.py  -  MicroBot v7  Word Association Module
=======================================================
Skip-gram word2vec trained ONLY on curated thematic sentences.

Why NOT WikiQA:  WikiQA is 60% US geography/biography. Training on it
makes "water" associate with "united states" not "liquid ocean rain".

This module uses:
  1. 200+ hand-crafted thematic sentences (core scientific knowledge)
  2. Clean seed Q-A answers (from dataset.py SEED_PAIRS)
  3. No WikiQA answers (too noisy for semantic associations)

Result: "apple" → fruit, red, sweet, tree, juice, seed, harvest
        "gravity" → force, mass, orbit, newton, weight, earth, pull
        "brain" → neuron, memory, thought, cortex, synapse, nerve
"""

import os, re, json, math, random
import numpy as np

ASSOC_FILE = "word_assoc.json"
VEC_FILE   = "word_vec.npy"
VOCAB_FILE = "word_vocab.json"

W2V_DIM   = 64
WINDOW    = 4
NEG_K     = 6
MIN_COUNT = 1    # low because curated corpus is small
LR_START  = 0.05
LR_END    = 0.001
EPOCHS    = 20   # more epochs on small curated corpus

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","i","you","we","they","he","she","it","this","that","these",
    "those","of","in","to","for","on","at","by","with","from","or","and","but",
    "not","no","as","its","also","both","which","who","what","how","than","into",
    "just","very","so","if","up","out","after","before","well","even","still",
    "each","any","once","down","now","here","there","only","too","own","some",
}

# ── CURATED THEMATIC CORPUS ──────────────────────────────────────────────────
# 300+ sentences across knowledge domains.
# Each sentence clusters semantically related words intentionally.
# Repeat important domain words across multiple sentences = stronger signal.
THEMATIC = [
    # ── GRAVITY / PHYSICS ──
    "gravity is a force that pulls objects with mass toward each other",
    "newton discovered gravity when an apple fell toward earth",
    "gravity causes apples to fall and keeps planets in orbit around the sun",
    "mass weight force gravity pull earth orbit attraction newton law",
    "gravity holds the moon in orbit around earth and earth in orbit around sun",
    "black holes have such extreme gravity that even light cannot escape",
    "gravity acceleration freefall weight ground pull force newton einstein",
    "gravity bends spacetime according to einsteins theory of general relativity",
    "gravity on earth pulls everything downward at nine point eight meters per second squared",
    "satellites orbit because gravity curves their path as they fall around earth",
    # ── ENERGY / FORCE ──
    "energy is the capacity to do work or cause change in the world",
    "kinetic energy is the energy of motion and potential energy is stored energy",
    "heat thermal energy temperature hot cold molecules vibrate faster slower",
    "energy transforms between kinetic potential thermal chemical electrical forms",
    "nuclear fusion releases tremendous energy by combining hydrogen atoms into helium",
    "solar energy from the sun powers photosynthesis and warms the earth",
    "electrical energy powers lights motors computers and all electronic devices",
    "chemical energy stored in food is released during metabolism and digestion",
    # ── LIGHT / SOUND ──
    "light photons electromagnetic radiation wave speed wavelength frequency color",
    "light travels at three hundred thousand kilometers per second in a vacuum",
    "visible light contains all colors from red orange yellow green blue violet",
    "red light has long wavelength and blue light has short wavelength frequency",
    "sound waves travel through air water solid vibration pressure frequency pitch",
    "sound requires a medium to travel while light can travel through vacuum",
    "high frequency sound is high pitch and low frequency sound is low pitch",
    # ── ATOMS / CHEMISTRY ──
    "atoms are smallest units of matter made of protons neutrons and electrons",
    "protons and neutrons form nucleus at center of atom surrounded by electrons",
    "electrons orbit nucleus in shells and determine chemical bonding properties",
    "molecules form when atoms bond together sharing or transferring electrons",
    "chemical reaction combines atoms molecules to form new substances",
    "water molecule two hydrogen atoms one oxygen atom bonded together",
    "acid base reaction neutralization salt water pH chemistry",
    "carbon oxygen nitrogen hydrogen are most abundant elements in living things",
    "metals conduct electricity because electrons move freely through their structure",
    "periodic table organizes elements by atomic number and chemical properties",
    # ── WATER ──
    "water liquid hydrogen oxygen molecule essential life ocean river lake",
    "water evaporates from ocean rises forms clouds rains back to earth cycle",
    "water is essential for all known life on earth dissolving nutrients",
    "ice is solid water formed when temperature drops below zero degrees celsius",
    "steam is water vapor formed when liquid water heats above boiling point",
    "oceans cover seventy one percent of earth surface containing salt water",
    "water cycle evaporation condensation precipitation runoff groundwater",
    "plants absorb water through roots and release it through leaves transpiration",
    # ── EARTH / PLANET ──
    "earth planet solar system orbit sun moon gravity atmosphere ocean",
    "earth has four layers crust mantle outer core inner core rock metal",
    "tectonic plates move float on mantle causing earthquakes volcanoes mountains",
    "earths atmosphere contains nitrogen oxygen argon carbon dioxide water vapor",
    "earths magnetic field protects life from harmful solar radiation and wind",
    "seasons occur because earth tilts on its axis as it orbits the sun",
    "moon orbits earth causing tides through gravitational pull on oceans",
    # ── SUN / STARS ──
    "sun is a star at center of solar system made of hydrogen helium plasma",
    "nuclear fusion in sun core converts hydrogen to helium releasing energy",
    "sun produces light heat energy radiation that warms earth and powers life",
    "stars are massive balls of plasma where nuclear fusion occurs in core",
    "stars form from clouds of gas and dust collapsing under gravity",
    "red stars are cooler and blue stars are hotter than yellow stars like sun",
    "supernova explosion occurs when massive star runs out of nuclear fuel",
    # ── SOLAR SYSTEM / UNIVERSE ──
    "solar system contains sun eight planets moons asteroids comets",
    "planets orbit sun in elliptical paths due to gravitational attraction",
    "galaxy is system of billions of stars gas dust held together by gravity",
    "milky way galaxy contains over two hundred billion stars including sun",
    "universe began with big bang thirteen point eight billion years ago",
    "dark matter dark energy make up most of universe but cannot be seen",
    # ── BIOLOGY / CELLS ──
    "cells are basic units of life with nucleus membrane mitochondria organelles",
    "nucleus contains dna chromosomes genes that control cell function reproduction",
    "mitochondria produce energy for cell through respiration consuming oxygen",
    "cell membrane controls what enters and exits the cell selectively",
    "cell division allows organisms to grow replace old cells and reproduce",
    # ── DNA / GENETICS ──
    "dna double helix molecule carries genetic information in every living cell",
    "dna genes chromosomes genetic inheritance protein synthesis cell nucleus",
    "genes are sections of dna that code for specific proteins and traits",
    "genetic mutations change dna sequence and can cause disease or evolution",
    "dna replication copies genetic information before cell division",
    "proteins are built according to instructions encoded in dna genes",
    # ── EVOLUTION ──
    "evolution change species over generations natural selection adaptation survival",
    "natural selection favors organisms best adapted to their environment",
    "charles darwin proposed evolution by natural selection after observing species",
    "mutations in dna create genetic variation that drives evolutionary change",
    "fossils provide evidence of extinct species and evolutionary history",
    "humans and chimpanzees share about ninety eight percent of their dna",
    # ── BRAIN / NEUROSCIENCE ──
    "brain neurons synapses memory thought consciousness cognition intelligence nerve",
    "human brain contains eighty six billion neurons connected by synapses",
    "neurons transmit electrical signals through axons and across synapses",
    "memory forms when connections between neurons strengthen through repetition",
    "cerebral cortex controls thought language movement perception and planning",
    "hippocampus plays key role in forming and retrieving long term memories",
    "sleep allows brain to consolidate memories and remove metabolic waste",
    "emotions originate in amygdala which triggers fear anxiety and pleasure responses",
    "dopamine serotonin oxytocin are neurotransmitters affecting mood and emotion",
    "brain plasticity means neural connections can change grow strengthen weaken",
    # ── HEART / BODY ──
    "heart pumps blood oxygen nutrients through arteries veins to all organs",
    "lungs absorb oxygen and release carbon dioxide during breathing respiration",
    "blood carries oxygen glucose nutrients hormones through cardiovascular system",
    "immune system white blood cells antibodies defend body against infection disease",
    "muscles contract relax to produce movement using energy from atp",
    "bones provide structure protect organs store calcium marrow produces blood cells",
    "kidneys filter blood remove waste produce urine regulate water balance",
    "liver metabolizes nutrients detoxifies blood produces bile for digestion",
    # ── PHOTOSYNTHESIS / PLANTS ──
    "photosynthesis converts sunlight carbon dioxide water into glucose oxygen plants",
    "chlorophyll green pigment in plants absorbs light for photosynthesis",
    "plants absorb sunlight through leaves convert water carbon dioxide to sugar",
    "roots absorb water minerals from soil stems transport nutrients leaves",
    "trees forests produce oxygen absorb carbon dioxide regulate climate",
    "fruits contain seeds surrounded by sweet flesh to attract animals spread seeds",
    # ── COLORS ──
    "red color long wavelength light associated with fire heat passion danger apple",
    "green color chlorophyll plants grass nature growth forest leaves",
    "blue color sky ocean water calm peaceful cold short wavelength",
    "yellow color sun gold warmth happiness light bright lemon banana",
    "orange color fruit citrus autumn warm sunset fire sunset",
    "purple violet color royalty creativity imagination rare pigment",
    "white light contains all colors of rainbow spectrum",
    "black absorbs all light wavelengths and reflects none",
    # ── APPLE / FRUIT ──
    "apple is a red or green fruit that grows on trees in orchards",
    "apple fruit red green sweet sour seeds core flesh juice tree harvest",
    "fruits contain seeds surrounded by nutritious flesh vitamin mineral",
    "orange lemon lime citrus fruits rich in vitamin c acidic juice",
    "banana yellow tropical fruit sweet carbohydrate potassium energy",
    "grape vine fruit wine juice raisin sweet purple green cluster",
    # ── FIRE ──
    "fire combustion fuel oxygen heat light flame burn chemical reaction",
    "fire requires fuel oxygen and heat to start and continue burning",
    "fire releases heat light carbon dioxide water through combustion",
    "flame is hot plasma gas produced by rapid oxidation combustion fuel",
    # ── FOOD / NUTRITION ──
    "food nutrition energy protein carbohydrate fat vitamin mineral health body",
    "protein builds repairs muscles organs made of amino acid chains",
    "carbohydrates sugar starch glucose energy fuel metabolism digestion",
    "vitamins minerals micronutrients essential body functions immunity",
    "digestion breaks food into nutrients absorbed by small intestine blood",
    "metabolism chemical processes convert food into energy in cells",
    # ── WATER CYCLE / WEATHER ──
    "rain cloud water cycle evaporation condensation precipitation atmosphere",
    "wind moves air from high pressure to low pressure regions atmosphere",
    "storm lightning thunder electricity discharge atmosphere energy",
    "snow ice crystal water frozen atmospheric temperature below zero",
    "climate long term weather pattern temperature rainfall region earth",
    "greenhouse gases carbon dioxide methane trap heat warm atmosphere",
    # ── AI / COMPUTING ──
    "artificial intelligence machine learning algorithm data model train predict",
    "neural network layers neurons weights training data prediction output",
    "machine learning trains models on data to find patterns make predictions",
    "deep learning uses many hidden layers neural networks recognize patterns",
    "algorithm step by step instructions solve problem compute result",
    "computer processor memory storage data program software hardware",
    "programming code function variable loop condition input output program",
    "data storage retrieval processing analysis pattern recognition learning",
    "robot machine sensor actuator program task automate movement decision",
    "internet network protocol server client data connection global communication",
    # ── MATHEMATICS ──
    "number arithmetic addition subtraction multiplication division calculation equation",
    "geometry shape angle triangle circle area volume measurement space",
    "algebra equation variable solve unknown symbol expression",
    "statistics probability data sample distribution mean average analysis",
    "calculus derivative integral change rate accumulation limit function",
    # ── MUSIC / ART ──
    "music sound rhythm melody harmony note pitch frequency instrument voice",
    "art paint color canvas creative expression visual beauty form",
    "dance movement rhythm body music expression cultural art form",
    "poetry words rhythm metaphor emotion beauty language expression",
    # ── EMOTIONS ──
    "love emotion bond attachment care affection trust relationship warmth",
    "happiness joy contentment pleasure satisfaction positive emotion wellbeing",
    "sadness grief sorrow loss pain unhappy negative emotion cry",
    "fear anxiety stress danger threat nervous worry fight flight survival",
    "anger frustration rage upset emotion reaction injustice defense",
    "surprise shock wonder amazement unexpected sudden discovery emotion",
    "curiosity wonder explore discover learn question interest seek",
    # ── LEARNING / MIND ──
    "learning memory practice repetition skill knowledge understanding intelligence",
    "attention focus concentration memory learning performance improvement",
    "creativity imagination novel idea combine knowledge new solution invention",
    "intelligence reasoning problem solving adapt learn understand apply",
    "consciousness awareness experience perception self thought subjective",
    "language words sentences grammar communicate meaning express understand",
    # ── SLEEP ──
    "sleep rest dream restore repair memory brain body recovery night",
    "sleep consolidates memories removes toxins restores energy",
    # ── MEDICINE ──
    "disease infection virus bacteria immune inflammation treatment medicine drug",
    "virus replicates inside cells using genetic material causes infection disease",
    "vaccine immune system antibody protection disease prevention",
    "antibiotic kills bacteria treats bacterial infection medicine",
    "heart disease blood pressure cholesterol arteries risk cardiovascular",
    "cancer abnormal cell growth tumor mutation treatment chemotherapy",
    # ── SOCIETY ──
    "democracy government vote citizen freedom rights justice law",
    "economy trade money market price supply demand inflation growth",
    "education school learn teach knowledge skill student teacher",
    "friendship trust loyalty care empathy support social bond",
    # ── SPACE ──
    "astronaut spaceship orbit weightless vacuum zero gravity exploration",
    "telescope observe distant stars galaxies light years space astronomy",
    "atmosphere layer gas oxygen nitrogen pressure protect life radiation",
]

def _tok(text):
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOPWORDS and len(w) > 2]

def _load_corpus():
    sentences = []

    # Thematic sentences (multiple passes = stronger signal on curated content)
    for _ in range(4):   # repeat 4× to dominate co-occurrence
        for s in THEMATIC:
            t = _tok(s)
            if len(t) >= 3:
                sentences.append(t)

    # Clean seed Q-A pairs from dataset.py (factual, no WikiQA noise)
    try:
        import zlib
        ds = json.loads(zlib.decompress(open("dataset.json","rb").read()).decode())
        # Only use pairs that look like core seed pairs (short, factual)
        # Heuristic: answer < 120 chars = likely seed, not WikiQA
        for pair in ds.get("pairs", []):
            if "=>" not in pair:
                continue
            _, a = pair.split("=>", 1)
            if len(a.strip()) <= 120:
                t = _tok(a.strip())
                if len(t) >= 3:
                    sentences.append(t)
    except Exception:
        pass

    total_tokens = sum(len(s) for s in sentences)
    print(f"  corpus: {len(sentences)} sentences  {total_tokens} tokens")
    return sentences

def _build_vocab(sentences):
    freq = {}
    for s in sentences:
        for w in s:
            freq[w] = freq.get(w, 0) + 1
    vocab = sorted([w for w,c in freq.items() if c >= MIN_COUNT], key=lambda w: -freq[w])
    w2i  = {w: i for i,w in enumerate(vocab)}
    i2w  = {i: w for w,i in w2i.items()}
    farr = np.array([freq[w] for w in vocab], dtype=np.float32)
    print(f"  vocab: {len(vocab)} words")
    return w2i, i2w, farr

def _neg_table(farr, size=500_000):
    p = farr ** 0.75; p /= p.sum()
    return np.random.choice(len(farr), size=size, p=p)

def train(verbose=True):
    if verbose:
        print("Loading curated corpus ...")
    sentences = _load_corpus()
    if verbose:
        print("Building vocabulary ...")
    w2i, i2w, farr = _build_vocab(sentences)
    V = len(w2i)

    idx_sents = [[w2i[w] for w in s if w in w2i] for s in sentences]
    idx_sents = [s for s in idx_sents if len(s) >= 2]

    T = 1e-3     # subsampling threshold (softer for small corpus)
    total_f = farr.sum()
    discard = np.maximum(0.0, 1.0 - np.sqrt(T * total_f / (farr + 1e-9)))

    scale = 1.0 / math.sqrt(W2V_DIM)
    W_in  = (np.random.rand(V, W2V_DIM).astype(np.float32) - 0.5) * scale
    W_out = np.zeros((V, W2V_DIM), dtype=np.float32)

    neg_table   = _neg_table(farr)
    N           = len(neg_table)
    total_steps = sum(len(s) for s in idx_sents) * EPOCHS
    step        = 0

    if verbose:
        print(f"Training skip-gram  V={V}  dim={W2V_DIM}  win={WINDOW}  epochs={EPOCHS}  neg={NEG_K}")

    for epoch in range(EPOCHS):
        random.shuffle(idx_sents)
        for sent in idx_sents:
            for pos, center in enumerate(sent):
                if random.random() < discard[center]:
                    continue
                win = random.randint(1, WINDOW)
                for cp in range(max(0, pos-win), min(len(sent), pos+win+1)):
                    if cp == pos:
                        continue
                    ctx = sent[cp]
                    if random.random() < discard[ctx]:
                        continue

                    lr  = max(LR_END, LR_START * (1.0 - step / (total_steps + 1)))
                    vi  = W_in[center]
                    vo  = W_out[ctx]
                    dot = max(-20.0, min(20.0, float(np.dot(vi, vo))))
                    sig = 1.0 / (1.0 + math.exp(-dot))
                    err = (sig - 1.0) * lr

                    g_in  = err * vo
                    g_out = err * vi
                    ns    = random.randint(0, N - NEG_K - 1)
                    for ni in range(NEG_K):
                        nw  = int(neg_table[ns + ni])
                        if nw == ctx: continue
                        vn  = W_out[nw]
                        dn  = max(-20.0, min(20.0, float(np.dot(vi, vn))))
                        sn  = 1.0 / (1.0 + math.exp(-dn))
                        en  = sn * lr
                        g_in += en * vn
                        W_out[nw] -= en * vi

                    W_in[center] -= g_in
                    W_out[ctx]   -= g_out
                    step         += 1

        if verbose:
            print(f"  epoch {epoch+1}/{EPOCHS}  [{int((epoch+1)/EPOCHS*100)}%]")

    W = (W_in + W_out) / 2.0
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W /= (norms + 1e-9)
    return W, w2i, i2w

def build_assoc_graph(W, w2i, i2w, top_n=15):
    V, assoc = len(w2i), {}
    batch = 256
    for start in range(0, V, batch):
        end  = min(V, start + batch)
        sims = W[start:end] @ W.T
        for li, gi in enumerate(range(start, end)):
            row = sims[li].copy()
            row[gi] = -1.0
            top = np.argsort(row)[::-1][:top_n]
            nbrs = [(i2w[int(j)], round(float(row[j]), 3))
                    for j in top if row[j] > 0.12]
            if nbrs:
                assoc[i2w[gi]] = nbrs
    return assoc

def save(W, w2i, i2w, assoc):
    np.save(VEC_FILE, W)
    json.dump({"w2i": w2i, "i2w": {str(k): v for k,v in i2w.items()}}, open(VOCAB_FILE,"w"))
    json.dump(assoc, open(ASSOC_FILE,"w"))
    print(f"Saved: {len(assoc)} word associations")

def load():
    if not all(os.path.exists(f) for f in (VEC_FILE, VOCAB_FILE, ASSOC_FILE)):
        return None, None, None, None
    try:
        W     = np.load(VEC_FILE)
        v     = json.load(open(VOCAB_FILE))
        w2i   = v["w2i"]
        i2w   = {int(k): val for k,val in v["i2w"].items()}
        assoc = json.load(open(ASSOC_FILE))
        return W, w2i, i2w, assoc
    except Exception as e:
        print(f"load error: {e}")
        return None, None, None, None

# ── Public API ────────────────────────────────────────────────────────────────
_W = _w2i = _i2w = _assoc = None

def _ensure():
    global _W, _w2i, _i2w, _assoc
    if _assoc is None:
        _W, _w2i, _i2w, _assoc = load()
        if _assoc is None:
            train_and_save()
            _W, _w2i, _i2w, _assoc = load()

def train_and_save():
    W, w2i, i2w = train()
    if W is not None:
        save(W, w2i, i2w, build_assoc_graph(W, w2i, i2w))

def similar_words(word, n=8):
    _ensure()
    if not _assoc: return []
    return _assoc.get(word.lower().strip(), [])[:n]

def word_vector(word):
    _ensure()
    if _W is None or word not in _w2i: return None
    return _W[_w2i[word]]

def sentence_vector(text):
    _ensure()
    if _W is None: return None
    ws = [w for w in re.findall(r"[a-z]+", text.lower())
          if w not in STOPWORDS and len(w) > 2 and w in _w2i]
    if not ws: return None
    return np.stack([_W[_w2i[w]] for w in ws]).mean(0)

def semantic_similarity(a, b):
    va, vb = sentence_vector(a), sentence_vector(b)
    if va is None or vb is None: return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(va, vb) / (na * nb))

def expand_query(text, n=5, threshold=0.22):
    _ensure()
    if not _assoc: return set()
    words = [w for w in re.findall(r"[a-z]+", text.lower())
             if w not in STOPWORDS and len(w) > 2]
    expanded = set()
    for w in words:
        for nbr, score in _assoc.get(w, [])[:n]:
            if score >= threshold:
                expanded.add(nbr)
    return expanded - set(words)

def analogy(pos_a, pos_b, neg, n=5):
    _ensure()
    if _W is None: return []
    miss = [w for w in (pos_a,pos_b,neg) if w not in _w2i]
    if miss: return [f"not in vocab: {miss}"]
    target = _W[_w2i[pos_a]] - _W[_w2i[neg]] + _W[_w2i[pos_b]]
    nm = np.linalg.norm(target)
    if nm > 1e-9: target /= nm
    sims  = _W @ target
    excl  = {_w2i[w] for w in (pos_a,pos_b,neg)}
    out   = []
    for idx in np.argsort(sims)[::-1]:
        if int(idx) not in excl:
            out.append((_i2w[int(idx)], round(float(sims[idx]),3)))
            if len(out) >= n: break
    return out

if __name__ == "__main__":
    print("=== MicroBot Word Association Module ===")
    rm_files = [VEC_FILE, VOCAB_FILE, ASSOC_FILE]
    for f in rm_files:
        if os.path.exists(f): os.remove(f)
    train_and_save()

    TEST = ["gravity","brain","apple","water","energy","sun","atom","love",
            "fire","ocean","heart","tree","robot","dna","sleep","red","green"]
    print("\n=== Word Associations ===")
    for w in TEST:
        sim = similar_words(w, 6)
        if sim:
            print(f"  {w:10s} → {', '.join(f'{n}({s:.2f})' for n,s in sim[:5])}")
        else:
            print(f"  {w:10s} → (not in vocab)")

    print("\n=== Query Expansion ===")
    for q in ["gravity force earth","brain memory neuron","apple red fruit","water ocean rain"]:
        exp = expand_query(q)
        print(f"  [{q}] → {sorted(exp)[:8]}")
