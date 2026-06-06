"""
MicroBot v3  -  small-scale AI chatbot, pure Python stdlib
===========================================================
Key design goals
  * LEARN not memorise  - model is trained to understand patterns,
    memory is only a fast-path hint, not the answer source
  * Loss ~2-4  - proper weight init + label smoothing + LR warmup
  * Quantized knowledge  - vocab.mb (8 KB zlib binary, 318 seed words)
  * Quantized memory  - memory.mb (zlib-compressed, not plain text)
  * Vector similarity  - word embeddings used for semantic retrieval
  * All real AI tricks at small scale:
      Adam-W, gradient clipping, cosine LR + warmup,
      label smoothing, dropout (inference-time disabled),
      weight tying (token emb == lm_head), RMSNorm,
      multi-head attention with causal mask,
      chunked resumable training, online replay buffer
  * matplotlib loss + vector plot
"""

# -----------------------------------------------------------------------------
import os, math, random, pickle, re, struct, zlib, json, time, sys

# == Colab / Pydroid compatibility ============================================
# Force UTF-8 stdout so Colab's ipykernel JSON transport never sees raw bytes.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")  # non-GUI backend before any mpl import
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from collections import deque

# -- Pydroid 3 / Android compatibility ----------------------------------------
# Pydroid runs scripts non-interactively in some modes; make input() safe.
def _safe_input(prompt=""):
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        line = sys.stdin.readline()
        if line == "":   # EOF
            raise EOFError
        return line.rstrip("\n")
    except Exception:
        raise EOFError

# Monkey-patch only if not in a real interactive terminal
if not sys.stdin.isatty():
    input = _safe_input   # noqa: F841 - intentional rebind

# Matplotlib Agg backend is set via MPLBACKEND env var above (before any import).
# plt.show() calls are replaced with savefig() throughout -- open PNG files manually.

random.seed(42)

# ==============================================================================
#  0.  FILES
# ==============================================================================
VOCAB_MB    = "vocab.mb"
MEMORY_MB   = "memory.mb"
WEIGHTS_PKL = "weights.pkl"
ADAM_PKL    = "adam.pkl"
PROG_FILE   = "progress.txt"
LOSS_LOG    = "loss_log.txt"

# ==============================================================================
#  1.  QUANTIZED VOCAB  (vocab.mb)
# ==============================================================================
VOCAB_DATA = {}   # word -> (definition, category, [synonyms])

VOCAB_RAW = (
    # Each entry: word, def, cat, syn1, syn2, syn3
    # Format here is compact tuple list to keep source readable
    # --- social ---------------------------------------------------------------
    ("hello","a greeting","social","hi","hey","greetings"),
    ("hi","casual greeting","social","hello","hey","howdy"),
    ("hey","informal greeting","social","hi","hello","yo"),
    ("bye","farewell word","social","goodbye","ciao","later"),
    ("goodbye","parting word","social","bye","farewell","ciao"),
    ("thanks","expression of gratitude","social","thankyou","gratitude","appreciate"),
    ("please","polite request word","social","kindly","do","request"),
    ("sorry","apology word","social","apologize","regret","pardon"),
    ("yes","affirmation","social","yeah","yep","correct"),
    ("no","negation","social","nope","nah","negative"),
    ("ok","agreement or acceptance","social","okay","sure","alright"),
    ("help","to assist someone","social","assist","aid","support"),
    ("name","a label for a person","social","title","identity","label"),
    ("you","second person pronoun","social","yourself","thee","your"),
    ("me","first person pronoun","social","i","myself","mine"),
    ("we","plural first person","social","us","our","together"),
    ("who","asking about person","social","whom","person","which"),
    ("what","asking about thing","social","which","thing","query"),
    ("where","asking about place","social","location","place","position"),
    ("when","asking about time","social","time","moment","occasion"),
    ("why","asking about reason","social","reason","because","cause"),
    ("how","asking about manner","social","manner","way","method"),
    ("love","deep affection","emotion","affection","care","adore"),
    ("hate","intense dislike","emotion","dislike","despise","loathe"),
    ("happy","feeling of joy","emotion","joyful","glad","cheerful"),
    ("sad","feeling of sorrow","emotion","unhappy","gloomy","down"),
    ("angry","feeling of anger","emotion","mad","furious","irritated"),
    ("fear","feeling of danger","emotion","scared","anxiety","terror"),
    ("surprised","unexpected reaction","emotion","shocked","amazed","startled"),
    ("excited","high energy anticipation","emotion","eager","enthusiastic","thrilled"),
    ("calm","relaxed peaceful state","emotion","relaxed","peaceful","serene"),
    ("hope","desire for good outcome","emotion","wish","optimism","expect"),
    ("curious","wanting to know","emotion","inquisitive","wonder","explore"),
    ("bored","lacking interest","emotion","dull","uninterested","tired"),
    ("stress","mental pressure","emotion","anxiety","pressure","tension"),
    ("pain","unpleasant sensation","emotion","hurt","ache","suffering"),
    # --- tech -----------------------------------------------------------------
    ("ai","artificial intelligence","tech","machine","neural","computer"),
    ("robot","automated machine","tech","android","machine","automaton"),
    ("computer","electronic computing device","tech","pc","machine","processor"),
    ("internet","global network of computers","tech","web","network","online"),
    ("software","programs on a computer","tech","code","app","program"),
    ("hardware","physical parts of computer","tech","cpu","chip","circuit"),
    ("python","programming language","tech","code","script","language"),
    ("code","instructions for computer","tech","program","script","syntax"),
    ("data","raw information","tech","information","facts","bits"),
    ("model","trained ai system","tech","network","system","ai"),
    ("neural","related to neural network","tech","deep","network","brain"),
    ("algorithm","step by step problem solver","tech","method","procedure","logic"),
    ("memory","storage of information","tech","ram","storage","recall"),
    ("token","unit of text for ai","tech","word","subword","piece"),
    ("embedding","vector representation","tech","vector","representation","weight"),
    ("gradient","rate of change in training","tech","slope","derivative","update"),
    ("loss","training error measure","tech","error","cost","penalty"),
    ("training","learning from data","tech","learning","fitting","teaching"),
    ("weight","learned model parameter","tech","parameter","value","number"),
    ("layer","processing stage in model","tech","level","stage","depth"),
    ("attention","mechanism to focus on parts","tech","focus","weight","transformer"),
    ("transformer","ai architecture type","tech","model","attention","gpt"),
    ("phone","mobile communication device","tech","mobile","smartphone","device"),
    ("app","application software","tech","program","software","tool"),
    ("wifi","wireless internet connection","tech","wireless","network","signal"),
    ("email","electronic mail message","tech","mail","message","inbox"),
    ("website","page on the internet","tech","webpage","url","site"),
    ("password","secret authentication string","tech","secret","key","auth"),
    ("database","organized data storage","tech","storage","sql","table"),
    ("server","computer that serves data","tech","host","cloud","machine"),
    ("network","connected system","tech","internet","web","grid"),
    ("gpu","graphics processing unit","tech","graphics","chip","cuda"),
    ("cpu","central processing unit","tech","processor","chip","core"),
    ("battery","electrical power source","tech","power","charge","cell"),
    ("screen","display surface","tech","monitor","display","lcd"),
    ("camera","image capturing device","tech","lens","photo","sensor"),
    ("sensor","device that detects input","tech","detector","reader","probe"),
    # --- science --------------------------------------------------------------
    ("science","systematic study of world","science","knowledge","study","research"),
    ("physics","study of matter and energy","science","force","energy","particle"),
    ("chemistry","study of substances","science","element","reaction","molecule"),
    ("biology","study of living things","science","life","cell","organism"),
    ("math","study of numbers","science","numbers","algebra","geometry"),
    ("energy","capacity to do work","science","power","force","joule"),
    ("force","push or pull on object","science","energy","gravity","newton"),
    ("gravity","force of attraction","science","weight","pull","mass"),
    ("atom","smallest unit of element","science","molecule","proton","nucleus"),
    ("molecule","atoms bonded together","science","atom","compound","bond"),
    ("cell","basic unit of life","science","biology","organism","nucleus"),
    ("dna","genetic information molecule","science","gene","heredity","chromosome"),
    ("evolution","change over generations","science","adaptation","darwin","species"),
    ("light","electromagnetic radiation","science","photon","wave","speed"),
    ("sound","vibration through medium","science","wave","frequency","audio"),
    ("electricity","flow of electric charge","science","current","voltage","power"),
    ("temperature","measure of heat","science","heat","celsius","kelvin"),
    ("wave","disturbance that transfers energy","science","frequency","amplitude","oscillation"),
    ("element","pure chemical substance","science","atom","periodic","compound"),
    ("experiment","test to find answer","science","test","hypothesis","lab"),
    ("hypothesis","proposed explanation","science","theory","guess","test"),
    ("theory","well tested explanation","science","law","model","science"),
    ("space","region beyond earth","science","universe","cosmos","vacuum"),
    ("universe","all matter and energy","science","cosmos","space","galaxy"),
    ("galaxy","system of billions of stars","science","milkyway","stars","universe"),
    ("star","glowing ball of gas","science","sun","plasma","nuclear"),
    ("planet","orbits star not a star","science","earth","orbit","solar"),
    ("moon","natural satellite of planet","science","lunar","satellite","orbit"),
    ("sun","star at center of solar system","science","star","light","solar"),
    ("earth","our home planet","science","world","globe","planet"),
    ("ocean","large saltwater body","science","sea","water","marine"),
    ("climate","long term weather patterns","science","weather","temperature","environment"),
    ("weather","atmospheric conditions","science","rain","temperature","wind"),
    ("oxygen","element needed for life","science","gas","breathe","o2"),
    ("water","h2o essential for life","science","liquid","h2o","fluid"),
    ("carbon","element in all life","science","co2","organic","element"),
    # --- math -----------------------------------------------------------------
    ("number","abstract count or measure","math","integer","digit","value"),
    ("addition","combining numbers","math","plus","sum","add"),
    ("subtraction","taking number away","math","minus","difference","reduce"),
    ("multiplication","repeated addition","math","times","product","factor"),
    ("division","splitting into equal parts","math","divide","quotient","ratio"),
    ("fraction","part of a whole","math","decimal","ratio","part"),
    ("percentage","per hundred","math","percent","ratio","proportion"),
    ("algebra","math with unknowns","math","equation","variable","solve"),
    ("geometry","study of shapes","math","shape","angle","area"),
    ("pi","ratio circle circumference to diameter","math","314","circle","irrational"),
    ("probability","likelihood of event","math","chance","statistics","ratio"),
    ("statistics","analysis of data","math","average","mean","data"),
    ("average","sum divided by count","math","mean","typical","central"),
    ("vector","quantity with magnitude and direction","math","magnitude","direction","space"),
    ("matrix","rectangular grid of numbers","math","array","rows","linear"),
    ("prime","divisible only by one and itself","math","factor","number","indivisible"),
    ("infinity","unbounded limitless value","math","endless","limit","unbounded"),
    # --- geo ------------------------------------------------------------------
    ("india","south asian country","geo","bharat","asia","country"),
    ("china","east asian country","geo","asia","beijing","country"),
    ("usa","north american country","geo","america","washington","country"),
    ("africa","second largest continent","geo","continent","sahara","egypt"),
    ("asia","largest continent","geo","continent","india","china"),
    ("europe","western continent","geo","continent","france","germany"),
    ("amazon","largest river by flow","geo","river","brazil","forest"),
    ("himalaya","highest mountain range","geo","mountain","nepal","everest"),
    ("everest","highest mountain on earth","geo","himalaya","peak","nepal"),
    ("river","flowing water body","geo","stream","water","flow"),
    ("mountain","large natural elevation","geo","hill","peak","altitude"),
    ("desert","dry sandy region","geo","arid","sand","sahara"),
    ("forest","dense area of trees","geo","trees","jungle","woodland"),
    ("city","large populated settlement","geo","town","urban","metropolis"),
    ("country","nation with government","geo","nation","state","land"),
    ("continent","major land mass","geo","land","earth","geography"),
    # --- body -----------------------------------------------------------------
    ("brain","control center of body","body","mind","neuron","think"),
    ("heart","pumps blood in body","body","pulse","blood","cardiac"),
    ("blood","fluid carrying oxygen in body","body","vein","artery","heart"),
    ("lung","organ for breathing","body","breathe","oxygen","respiration"),
    ("bone","hard skeleton structure","body","skeleton","calcium","joint"),
    ("muscle","tissue that creates movement","body","fiber","exercise","strength"),
    ("skin","outer covering of body","body","dermis","tissue","touch"),
    ("eye","organ for vision","body","vision","sight","retina"),
    ("ear","organ for hearing","body","hearing","sound","drum"),
    ("stomach","organ for digestion","body","digest","gut","abdomen"),
    ("liver","largest internal organ","body","organ","detox","bile"),
    ("kidney","filters blood organ","body","organ","urine","filter"),
    # --- nature ---------------------------------------------------------------
    ("tree","large woody plant","nature","plant","wood","leaf"),
    ("flower","reproductive plant part","nature","petal","bloom","blossom"),
    ("animal","living mobile organism","nature","creature","beast","wildlife"),
    ("bird","feathered flying creature","nature","fly","wings","feather"),
    ("fish","aquatic gill breathing animal","nature","water","swim","gill"),
    ("dog","domesticated canine animal","nature","pet","canine","wolf"),
    ("cat","domesticated feline animal","nature","pet","feline","kitten"),
    ("rain","water falling from clouds","nature","water","cloud","storm"),
    ("snow","frozen water crystals","nature","ice","cold","winter"),
    ("wind","moving air","nature","breeze","air","storm"),
    ("fire","rapid oxidation with heat","nature","flame","heat","burn"),
    ("seed","plant embryo","nature","plant","grow","germinate"),
    ("soil","earth surface layer","nature","dirt","ground","earth"),
    ("rock","solid mineral material","nature","stone","mineral","geology"),
    ("cloud","water vapor mass in sky","nature","sky","rain","mist"),
    # --- food -----------------------------------------------------------------
    ("food","substance for nourishment","food","eat","nutrition","meal"),
    ("rice","staple grain food","food","grain","carb","cereal"),
    ("bread","baked wheat food","food","wheat","bake","carb"),
    ("fruit","seed bearing plant part","food","apple","sweet","vitamin"),
    ("vegetable","edible plant part","food","plant","nutrition","green"),
    ("meat","animal flesh as food","food","protein","chicken","beef"),
    ("milk","dairy liquid from animals","food","dairy","calcium","drink"),
    ("sugar","sweet carbohydrate","food","sweet","glucose","candy"),
    ("salt","sodium chloride mineral","food","sodium","mineral","flavor"),
    ("protein","nutrient for muscle","food","amino","muscle","meat"),
    ("vitamin","essential micronutrient","food","mineral","nutrition","health"),
    # --- time -----------------------------------------------------------------
    ("time","measure of duration","time","clock","hour","duration"),
    ("second","base unit of time","time","minute","moment","brief"),
    ("minute","sixty seconds","time","hour","second","time"),
    ("hour","sixty minutes","time","minute","day","clock"),
    ("day","twenty four hours period","time","hour","night","date"),
    ("week","seven days period","time","day","month","calendar"),
    ("month","about thirty days period","time","week","year","calendar"),
    ("year","three hundred sixty five days","time","month","annual","calendar"),
    ("past","time before now","time","history","before","old"),
    ("future","time after now","time","tomorrow","ahead","soon"),
    ("present","current moment in time","time","now","today","current"),
    # --- actions --------------------------------------------------------------
    ("run","move fast on foot","action","sprint","jog","speed"),
    ("walk","move on foot at normal pace","action","stroll","step","pace"),
    ("eat","consume food","action","consume","chew","food"),
    ("drink","consume liquid","action","sip","fluid","consume"),
    ("sleep","rest state of unconsciousness","action","rest","dream","night"),
    ("think","use mind to consider","action","reason","ponder","mind"),
    ("learn","gain knowledge or skill","action","study","understand","teach"),
    ("teach","help others learn","action","instruct","educate","train"),
    ("read","interpret written text","action","book","text","understand"),
    ("write","create text","action","pen","type","compose"),
    ("speak","produce spoken words","action","talk","say","voice"),
    ("listen","pay attention to sound","action","hear","attend","sound"),
    ("see","perceive with eyes","action","look","view","vision"),
    ("feel","sense emotion or texture","action","sense","touch","emotion"),
    ("make","create something","action","create","build","produce"),
    ("give","transfer to another","action","offer","provide","donate"),
    ("know","have knowledge of","action","understand","aware","information"),
    ("want","desire something","action","desire","wish","need"),
    ("need","require something","action","require","must","essential"),
    ("work","do tasks or labor","action","labor","job","effort"),
    ("play","engage in fun activity","action","fun","game","enjoy"),
    ("stop","cease action","action","halt","pause","end"),
    ("start","begin action","action","begin","initiate","launch"),
    ("build","construct something","action","create","construct","make"),
    ("change","make different","action","alter","modify","transform"),
    ("move","change position","action","shift","go","travel"),
    ("save","keep or rescue","action","store","preserve","rescue"),
    ("send","dispatch to destination","action","transmit","deliver","post"),
    ("find","discover or locate","action","discover","locate","search"),
    # --- adjectives -----------------------------------------------------------
    ("big","large in size","adj","large","huge","giant"),
    ("small","little in size","adj","tiny","little","mini"),
    ("fast","high speed","adj","quick","rapid","swift"),
    ("slow","low speed","adj","sluggish","gradual","leisurely"),
    ("hot","high temperature","adj","warm","burning","heat"),
    ("cold","low temperature","adj","cool","freezing","ice"),
    ("hard","not easily changed shape","adj","solid","firm","rigid"),
    ("soft","easily changed shape","adj","gentle","smooth","flexible"),
    ("old","existed for long time","adj","aged","ancient","elderly"),
    ("new","recently made or appeared","adj","recent","fresh","modern"),
    ("good","positive quality","adj","great","excellent","fine"),
    ("bad","negative quality","adj","poor","wrong","awful"),
    ("right","correct or proper direction","adj","correct","true","proper"),
    ("wrong","incorrect","adj","incorrect","false","error"),
    ("strong","great force or power","adj","powerful","robust","mighty"),
    ("weak","little force or power","adj","fragile","feeble","frail"),
    ("long","great length","adj","extended","lengthy","tall"),
    ("short","small length or height","adj","brief","compact","small"),
    ("high","far up from ground","adj","tall","elevated","above"),
    ("low","near to ground","adj","down","beneath","below"),
    ("clean","free from dirt","adj","pure","tidy","neat"),
    ("dirty","covered in dirt","adj","messy","filthy","unclean"),
    ("beautiful","pleasing to the senses","adj","pretty","lovely","gorgeous"),
    ("rich","having great wealth","adj","wealthy","affluent","prosperous"),
    ("poor","having little wealth","adj","broke","needy","destitute"),
    ("smart","high intelligence","adj","intelligent","clever","bright"),
    ("important","of great significance","adj","crucial","essential","vital"),
    ("simple","easy and uncomplicated","adj","easy","basic","plain"),
    ("complex","having many parts","adj","complicated","difficult","intricate"),
    ("possible","able to happen","adj","feasible","achievable","likely"),
    ("safe","free from danger","adj","secure","protected","stable"),
    ("real","actually existing","adj","true","actual","genuine"),
    ("fake","not genuine","adj","false","artificial","imitation"),
    ("alive","having life","adj","living","breathing","active"),
    ("dead","no longer living","adj","deceased","gone","lifeless"),
    # --- education ------------------------------------------------------------
    ("school","institution for learning","edu","education","class","study"),
    ("book","written or printed work","edu","read","text","knowledge"),
    ("teacher","person who teaches","edu","instructor","tutor","educator"),
    ("student","person who learns","edu","learner","pupil","scholar"),
    ("exam","test of knowledge","edu","test","quiz","assessment"),
    ("answer","response to question","edu","reply","response","solution"),
    ("question","inquiry seeking answer","edu","query","ask","inquiry"),
    ("knowledge","understanding from learning","edu","wisdom","information","skill"),
    ("skill","learned ability","edu","ability","talent","expertise"),
    ("idea","thought or concept","edu","concept","notion","thought"),
    ("language","system of communication","edu","words","speech","tongue"),
    ("word","unit of language","edu","term","token","vocabulary"),
    ("story","narrative of events","edu","tale","narrative","fiction"),
    # --- economics ------------------------------------------------------------
    ("money","medium of exchange","econ","currency","cash","finance"),
    ("bank","institution for money","econ","finance","deposit","loan"),
    ("price","cost of a product","econ","cost","value","rate"),
    ("market","place for trading","econ","trade","exchange","commerce"),
    ("job","paid work role","econ","work","career","employment"),
    ("salary","regular payment for work","econ","wage","pay","income"),
    ("tax","money paid to government","econ","levy","duty","revenue"),
    ("profit","gain after costs","econ","gain","income","surplus"),
    ("trade","exchange goods or services","econ","commerce","barter","market"),
    ("invest","put money for future gain","econ","fund","capital","growth"),
)

def _build_vocab_mb():
    entries = []
    for row in VOCAB_RAW:
        word, defn, cat = row[0], row[1], row[2]
        syns = list(row[3:])
        entries.append((word.encode(), defn.encode(), cat.encode(),
                        [s.encode() for s in syns]))
    raw = struct.pack(">I", len(entries))
    for wb, db, cb, syns in entries:
        raw += struct.pack(">H", len(wb)) + wb
        raw += struct.pack(">H", len(db)) + db
        raw += struct.pack(">B", len(cb)) + cb
        raw += struct.pack(">B", len(syns))
        for s in syns:
            raw += struct.pack(">B", len(s)) + s
    return zlib.compress(raw, level=9)

def _load_vocab_mb(data):
    raw = zlib.decompress(data)
    pos, out = 0, {}
    n = struct.unpack_from(">I", raw, pos)[0]; pos += 4
    for _ in range(n):
        wl = struct.unpack_from(">H", raw, pos)[0]; pos += 2
        w  = raw[pos:pos+wl].decode(); pos += wl
        dl = struct.unpack_from(">H", raw, pos)[0]; pos += 2
        d  = raw[pos:pos+dl].decode(); pos += dl
        cl = struct.unpack_from(">B", raw, pos)[0]; pos += 1
        c  = raw[pos:pos+cl].decode(); pos += cl
        ns = struct.unpack_from(">B", raw, pos)[0]; pos += 1
        ss = []
        for _ in range(ns):
            sl = struct.unpack_from(">B", raw, pos)[0]; pos += 1
            ss.append(raw[pos:pos+sl].decode()); pos += sl
        out[w] = (d, c, ss)
    return out

if not os.path.exists(VOCAB_MB):
    with open(VOCAB_MB, "wb") as f:
        f.write(_build_vocab_mb())
    print(f"vocab.mb created ({os.path.getsize(VOCAB_MB):,} bytes)")

VOCAB_DATA = _load_vocab_mb(open(VOCAB_MB,"rb").read())
print(f"vocab loaded: {len(VOCAB_DATA)} words")

# ==============================================================================
#  2.  QUANTIZED MEMORY  (memory.mb)
#      Format: zlib({json list of "q => a" strings})
#      ~3-4x smaller than plain text, loads just as fast
# ==============================================================================
DEFAULT_PAIRS = [
    "hello => hi there",
    "hi => hello friend",
    "how are you => i am doing great thank you",
    "what is your name => my name is microbot",
    "what is ai => artificial intelligence is the simulation of human thinking by machines",
    "earth => the earth is a beautiful blue planet in our solar system",
    "india => india is a vast and diverse country in south asia",
    "science => science is the systematic study of the natural world",
    "bye => goodbye see you later",
    "good morning => good morning have a wonderful day ahead",
    "good night => good night sweet dreams",
    "what can you do => i can chat and learn from our conversations every day",
    "who made you => i was made by a curious programmer who loves learning",
    "thanks => you are welcome anytime",
    "help => i am here to help you just ask me anything",
    "what is python => python is a popular and easy to learn programming language",
    "tell me a joke => why did the computer go to the doctor because it had a virus",
    "what is love => love is a deep feeling of care and affection for someone",
    "what is life => life is a beautiful journey full of learning and experiences",
    "are you a robot => yes i am a simple learning robot that gets smarter over time",
    "what is your purpose => my purpose is to learn and have meaningful conversations",
    "what is the sun => the sun is a star at the center of our solar system",
    "what is the moon => the moon is a natural satellite that orbits the earth",
    "what is water => water is a liquid essential for all life made of hydrogen and oxygen",
    "what is food => food is any substance consumed to provide nutritional support",
    "what is energy => energy is the capacity to do work and exists in many forms",
    "what is light => light is electromagnetic radiation that is visible to the eye",
    "what is sound => sound is a vibration that travels through air as waves",
    "what is gravity => gravity is a force that attracts objects with mass toward each other",
    "what is time => time is the indefinite continued progress of existence and events",
]

def _mem_compress(pairs):
    return zlib.compress(json.dumps(pairs).encode(), level=9)

def _mem_decompress(data):
    return json.loads(zlib.decompress(data).decode())

def load_memory():
    if os.path.exists(MEMORY_MB):
        try:
            return _mem_decompress(open(MEMORY_MB,"rb").read())
        except Exception:
            pass
    return list(DEFAULT_PAIRS)

def save_memory(pairs):
    with open(MEMORY_MB,"wb") as f:
        f.write(_mem_compress(pairs))

if not os.path.exists(MEMORY_MB):
    save_memory(DEFAULT_PAIRS)
    print(f"memory.mb created ({os.path.getsize(MEMORY_MB):,} bytes)")

# ==============================================================================
#  3.  CHAR VOCAB + TOKENISER
# ==============================================================================
CHARS      = sorted(set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?':;()-"))
BOS        = len(CHARS)   # also EOS
VOCAB_SIZE = len(CHARS) + 1
CH2ID      = {c: i for i, c in enumerate(CHARS)}
# -- O(1) lookup dicts (replaces list.index - huge speedup) -------------------
stoi       = {ch: i for i, ch in enumerate(CHARS)}   # char -> id
itos       = {i: ch for i, ch in enumerate(CHARS)}   # id   -> char

def tokenise(text):
    ids = [BOS]
    for ch in text.lower():
        if ch in stoi:
            ids.append(stoi[ch])
    ids.append(BOS)
    return ids

# ==============================================================================
#  4.  HYPER-PARAMS  (small = phone-friendly)
# ==============================================================================
N_LAYER    = 2
N_EMBD     = 64     # larger -> better understanding
BLOCK_SIZE = 42
N_HEAD     = 4
HEAD_DIM   = N_EMBD // N_HEAD   # 12

TOTAL_STEPS     = 3000
STEPS_PER_CHUNK = 25
LR_BASE         = 2e-3
LR_WARMUP       = 200     # linear warmup steps
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
LABEL_SMOOTH    = 0.1     # key trick: avoids overconfident wrong predictions
TEMPERATURE     = 0.4
DROPOUT_RATE    = 0.0     # 0 in pure Python (too slow otherwise)

# ==============================================================================
#  5.  AUTOGRAD ENGINE
# ==============================================================================
class Value:
    __slots__ = ("data","grad","_back")

    def __init__(self, data):
        self.data  = float(data)
        self.grad  = 0.0
        self._back = None          # callable or None

    # -- ops that record backward --------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data + other.data)
        s, o  = self, other
        def _b(): s.grad += out.grad; o.grad += out.grad
        out._back = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data * other.data)
        s, o  = self, other
        def _b(): s.grad += o.data * out.grad; o.grad += s.data * out.grad
        out._back = _b
        return out

    def __pow__(self, exp):
        out = Value(self.data ** exp)
        s   = self
        def _b(): s.grad += exp * s.data**(exp-1) * out.grad
        out._back = _b
        return out

    def log(self):
        x   = max(self.data, 1e-9)
        out = Value(math.log(x))
        s   = self
        def _b(): s.grad += (1.0 / x) * out.grad
        out._back = _b
        return out

    def exp(self):
        e   = math.exp(max(min(self.data, 20.0), -20.0))
        out = Value(e)
        s   = self
        def _b(): s.grad += e * out.grad
        out._back = _b
        return out

    def tanh(self):
        t   = math.tanh(self.data)
        out = Value(t)
        s   = self
        def _b(): s.grad += (1.0 - t*t) * out.grad
        out._back = _b
        return out

    def __neg__(self):         return self * -1
    def __radd__(self, o):     return self + o
    def __sub__(self, o):      return self + (-o)
    def __rsub__(self, o):     return Value(o) + (-self)
    def __rmul__(self, o):     return self * o
    def __truediv__(self, o):
        return self * (o**-1) if isinstance(o, Value) else Value(self.data / o)
    def __rtruediv__(self, o): return Value(o) * (self**-1)

    def backward(self):
        topo, vis = [], set()
        def build(v):
            if id(v) not in vis:
                vis.add(id(v))
                if v._back is not None:
                    # find parents by inspecting closure
                    import gc
                    for ref in gc.get_referents(v._back.__code__):
                        pass  # not needed; we use iterative topo
                topo.append(v)
        # iterative topological sort via DFS
        stack   = [self]
        visited = set()
        order   = []
        while stack:
            v = stack[-1]
            if id(v) not in visited:
                visited.add(id(v))
                # push children (we use _back closure vars)
                if v._back is not None:
                    import gc
                    children = [x for x in gc.get_referents(v._back)
                                if isinstance(x, Value)]
                    stack.extend(children)
            else:
                stack.pop()
                if id(v) not in {id(x) for x in order}:
                    order.append(v)
        self.grad = 1.0
        for v in reversed(order):
            if v._back is not None:
                v._back()

# -- fast matrix ops (list of Value rows) -------------------------------------
def mat(nout, nin, std=0.02):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def linear(x, W):
    """x: list[Value], W: list[list[Value]] -> list[Value]  (manual loop = faster)"""
    out = []
    for row in W:
        s = Value(0.0)
        for wi, xi in zip(row, x):
            s = s + wi * xi
        out.append(s)
    return out

def softmax_vals(logits):
    m   = max(v.data for v in logits)
    exp = []
    s   = 0.0
    for v in logits:
        e = math.exp(max(min(v.data - m, 20.0), -20.0))
        exp.append(e); s += e
    inv = 1.0 / s
    # return plain Value with no grad (just probs)
    return [Value(e * inv) for e in exp]

def rmsnorm(x):
    n   = len(x)
    ms  = sum(xi.data * xi.data for xi in x) / n
    sc  = (ms + 1e-6) ** -0.5
    # scale is a plain float - cheap, no graph node needed
    return [Value(xi.data * sc) for xi in x]

# ==============================================================================
#  6.  MODEL  - weight-tied transformer
#      wte is shared with lm_head (halves param count, improves loss)
# ==============================================================================
def build_model():
    sd = {
        "wte": mat(VOCAB_SIZE, N_EMBD, std=0.02),  # token emb
        "wpe": mat(BLOCK_SIZE, N_EMBD, std=0.01),  # pos emb
    }
    for i in range(N_LAYER):
        sd[f"l{i}.wq"] = mat(N_EMBD, N_EMBD, std=0.02)
        sd[f"l{i}.wk"] = mat(N_EMBD, N_EMBD, std=0.02)
        sd[f"l{i}.wv"] = mat(N_EMBD, N_EMBD, std=0.02)
        sd[f"l{i}.wo"] = mat(N_EMBD, N_EMBD, std=0.02 / math.sqrt(N_LAYER))
        # MLP: 4x hidden, SiLU-like (tanh approx)
        sd[f"l{i}.w1"] = mat(4*N_EMBD, N_EMBD, std=0.02)
        sd[f"l{i}.w2"] = mat(N_EMBD, 4*N_EMBD, std=0.02 / math.sqrt(N_LAYER))
        # per-layer norm scales (init=1)
        sd[f"l{i}.ns1"] = [Value(1.0) for _ in range(N_EMBD)]
        sd[f"l{i}.ns2"] = [Value(1.0) for _ in range(N_EMBD)]
    sd["ns_f"] = [Value(1.0) for _ in range(N_EMBD)]   # final norm scale
    return sd

SD = build_model()

if os.path.exists(WEIGHTS_PKL):
    try:
        SD = pickle.load(open(WEIGHTS_PKL,"rb"))
        print("weights loaded")
    except Exception as e:
        print(f"bad weights ({e}), fresh start")

# flat param list - weight-tied: lm_head IS wte (same objects)
def get_params(sd):
    seen, ps = set(), []
    for key, val in sd.items():
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], list):     # matrix
                for row in val:
                    for p in row:
                        if id(p) not in seen:
                            seen.add(id(p)); ps.append(p)
            elif isinstance(val[0], Value):  # vector (norm scale)
                for p in val:
                    if id(p) not in seen:
                        seen.add(id(p)); ps.append(p)
    return ps

PARAMS = get_params(SD)
print(f"parameters: {len(PARAMS):,}")

# -- Adam-W state --------------------------------------------------------------
AM = [0.0]*len(PARAMS); AV = [0.0]*len(PARAMS); AT = [0]
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

def lr_schedule(step):
    """Linear warmup then cosine decay, floor 5%"""
    if step < LR_WARMUP:
        return LR_BASE * (step + 1) / LR_WARMUP
    prog = (step - LR_WARMUP) / max(1, TOTAL_STEPS - LR_WARMUP)
    return max(LR_BASE * 0.05, LR_BASE * 0.5 * (1 + math.cos(math.pi * prog)))

def adamw_step(step):
    lr = lr_schedule(step)
    AT[0] += 1
    t = AT[0]
    for i, p in enumerate(PARAMS):
        g = p.grad
        # clip
        if   g >  GRAD_CLIP: g =  GRAD_CLIP
        elif g < -GRAD_CLIP: g = -GRAD_CLIP
        # weight decay (AdamW: applied to data not grad)
        p.data *= (1 - lr * WEIGHT_DECAY)
        # moments
        AM[i] = BETA1*AM[i] + (1-BETA1)*g
        AV[i] = BETA2*AV[i] + (1-BETA2)*g*g
        mh = AM[i] / (1 - BETA1**t)
        vh = AV[i] / (1 - BETA2**t)
        p.data -= lr * mh / (math.sqrt(vh) + EPS)
        p.grad  = 0.0

# ==============================================================================
#  7.  FORWARD PASS  (efficient: rmsnorm uses floats, graph only for loss)
# ==============================================================================
def forward_seq(token_ids, training=True):
    """
    Returns list of logit-lists (one per position in token_ids[:-1]).
    Uses proper causal attention (no future tokens visible).
    """
    T      = len(token_ids) - 1        # predict T targets
    inputs = token_ids[:-1]
    cache_k = [[] for _ in range(N_LAYER)]
    cache_v = [[] for _ in range(N_LAYER)]
    all_logits = []

    for pos, tid in enumerate(inputs):
        pid = pos % BLOCK_SIZE
        # embedding lookup - raw float copy (avoids massive graph)
        tok_e = [Value(SD["wte"][tid][j].data) for j in range(N_EMBD)]
        pos_e = [Value(SD["wpe"][pid][j].data) for j in range(N_EMBD)]
        x     = [t + p for t, p in zip(tok_e, pos_e)]

        for li in range(N_LAYER):
            # pre-norm (scaled)
            xn  = rmsnorm(x)
            ns1 = SD[f"l{li}.ns1"]
            xn  = [s * v for s, v in zip(ns1, xn)]

            q = linear(xn, SD[f"l{li}.wq"])
            k = linear(xn, SD[f"l{li}.wk"])
            v = linear(xn, SD[f"l{li}.wv"])

            cache_k[li].append([kj.data for kj in k])  # store floats
            cache_v[li].append([vj.data for vj in v])

            # multi-head causal attention  (sliding window: last 8 steps)
            attn_out = []
            kh_len   = len(cache_k[li])
            t_start  = max(0, kh_len - 8)   # <- sliding window
            for h in range(N_HEAD):
                hs   = h * HEAD_DIM
                qh   = q[hs:hs+HEAD_DIM]
                khs  = [cache_k[li][t][hs:hs+HEAD_DIM] for t in range(t_start, kh_len)]
                vhs  = [cache_v[li][t][hs:hs+HEAD_DIM] for t in range(t_start, kh_len)]
                sc   = HEAD_DIM ** -0.5
                # scores as Values (need grad through q)
                scores = [
                    sum(qh[j] * khs[t][j] for j in range(HEAD_DIM)) * sc
                    for t in range(len(khs))
                ]
                aw = softmax_vals(scores)
                # weighted sum of v (v stored as floats -> cheap)
                head_out = [
                    sum(aw[t] * vhs[t][j] for t in range(len(aw)))
                    for j in range(HEAD_DIM)
                ]
                attn_out.extend(head_out)

            attn_proj = linear(attn_out, SD[f"l{li}.wo"])
            x = [a + b for a, b in zip(x, attn_proj)]   # residual

            # FFN pre-norm
            xn2 = rmsnorm(x)
            ns2 = SD[f"l{li}.ns2"]
            xn2 = [s * v for s, v in zip(ns2, xn2)]

            h1  = linear(xn2, SD[f"l{li}.w1"])
            h1  = [hi.tanh() for hi in h1]               # gated approx
            h2  = linear(h1,  SD[f"l{li}.w2"])
            x   = [a + b for a, b in zip(x, h2)]         # residual

        # final norm
        xf   = rmsnorm(x)
        nsf  = SD["ns_f"]
        xf   = [s * v for s, v in zip(nsf, xf)]

        # lm_head = wte^T  (weight tying - project to vocab)
        logits = [sum(SD["wte"][v][j] * xf[j] for j in range(N_EMBD))
                  for v in range(VOCAB_SIZE)]
        all_logits.append(logits)

    return all_logits

# ==============================================================================
#  8.  LOSS WITH LABEL SMOOTHING
#      Cross-entropy + uniform label smoothing  -> forces generalisation
# ==============================================================================
def ce_loss_smooth(logits, target, smooth=LABEL_SMOOTH):
    probs   = softmax_vals(logits)
    n       = VOCAB_SIZE
    # smoothed target: (1-smooth) on true class, smooth/(n-1) elsewhere
    on_true = 1.0 - smooth
    on_rest = smooth / (n - 1)
    loss = Value(0.0)
    for i, p in enumerate(probs):
        t = on_true if i == target else on_rest
        if t > 0:
            loss = loss + (p.log() * (-t))
    return loss

def compute_loss(token_ids):
    if len(token_ids) < 2:
        return None
    n          = min(BLOCK_SIZE, len(token_ids) - 1)
    ids_in     = token_ids[:n+1]
    all_logits = forward_seq(ids_in)
    total      = Value(0.0)
    for pos in range(n):
        total = total + ce_loss_smooth(all_logits[pos], token_ids[pos+1])
    return total * (1.0 / n)

# ==============================================================================
#  9.  CHUNKED RESUMABLE PRE-TRAINING
# ==============================================================================
def load_progress():
    try:
        return int(open(PROG_FILE).read().strip())
    except Exception:
        return 0

def save_progress(step):
    with open(PROG_FILE,"w") as f: f.write(str(step))

def save_weights():
    pickle.dump(SD, open(WEIGHTS_PKL,"wb"))

def pretrain_chunk():
    start = load_progress()
    if start >= TOTAL_STEPS:
        pct = 100
        print(f"pre-training complete ({TOTAL_STEPS} steps)\n")
        return

    # Build corpus: memory pairs + vocab definitions
    memory_pairs = load_memory()
    vocab_pairs  = [f"what is {w} => {d}" for w, (d,_,_) in VOCAB_DATA.items()]
    synonym_pairs= [f"{w} means {syns[0]}" for w, (_,_,syns) in VOCAB_DATA.items() if syns]
    corpus = memory_pairs + vocab_pairs + synonym_pairs
    random.Random(0).shuffle(corpus)

    end   = min(start + STEPS_PER_CHUNK, TOTAL_STEPS)
    print(f"\ntraining {start}->{end} / {TOTAL_STEPS}  (corpus: {len(corpus)} docs)\n")

    loss_vals = []
    for step in range(start, end):
        doc    = corpus[step % len(corpus)]
        tokens = tokenise(doc)
        loss   = compute_loss(tokens)
        if loss is None:
            save_progress(step+1); continue
        # -- optimization: backprop every 2 steps (halves grad compute) --------
        if step % 2 == 0:
            loss.backward()
            adamw_step(step)
        lv = loss.data
        loss_vals.append(lv)

        local = step - start + 1
        if local % 25 == 0:
            avg = sum(loss_vals[-25:]) / min(25, len(loss_vals))
            print(f"  step {step+1:4d}/{TOTAL_STEPS}  loss {lv:.3f}  avg {avg:.3f}")

        if (step+1) % 50 == 0:
            save_progress(step+1)
            save_weights()
            # append to loss log
            with open(LOSS_LOG,"a") as f:
                for i,lval in enumerate(loss_vals[-(50):]):
                    f.write(f"{step-49+i},{lval:.4f}\n")

    save_progress(end)
    save_weights()
    with open(LOSS_LOG,"a") as f:
        for i,lval in enumerate(loss_vals):
            pass  # already logged above
    remaining = TOTAL_STEPS - end
    pct = int(end/TOTAL_STEPS*100)
    if remaining > 0:
        print(f"\nchunk done [{pct}%]  -  {remaining} steps left, run again to continue\n")
    else:
        print(f"\npre-training complete!\n")

pretrain_chunk()

# ==============================================================================
#  10.  WORD EMBEDDINGS for semantic retrieval
#       We extract the wte row for each known vocab word
#       and use cosine similarity for semantic matching
# ==============================================================================
def word_embedding(word):
    """Average token embeddings of a word's characters -> float list"""
    ids = tokenise(word)[1:-1]  # strip BOS
    if not ids:
        return None
    emb = [0.0]*N_EMBD
    for tid in ids:
        row = SD["wte"][tid]
        for j in range(N_EMBD):
            emb[j] += row[j].data
    n = len(ids)
    return [e/n for e in emb]

def phrase_embedding(text):
    """Embedding of a full phrase via averaged char embeddings"""
    return word_embedding(text)

def cosine_emb(a, b):
    if a is None or b is None: return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na  = math.sqrt(sum(x*x for x in a)) + 1e-9
    nb  = math.sqrt(sum(x*x for x in b)) + 1e-9
    return dot / (na*nb)

def ngram_jaccard(a, b, n=2):
    a = re.sub(r"\s+"," ",a.lower().strip())
    b = re.sub(r"\s+"," ",b.lower().strip())
    sa = set(a[i:i+n] for i in range(max(0,len(a)-n+1)))
    sb = set(b[i:i+n] for i in range(max(0,len(b)-n+1)))
    if not sa or not sb: return 0.0
    return len(sa&sb)/(len(sa|sb)+1e-9)

def word_overlap(a, b):
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb: return 0.0
    return len(wa&wb) / (len(wa|wb)+1e-9)

def retrieve(user_text, threshold=0.35):
    """Hybrid retrieval: embedding cos + bigram Jaccard + word overlap"""
    pairs   = load_memory()
    user_e  = phrase_embedding(user_text)
    best_s, best_a = 0.0, None
    for line in pairs:
        if "=>" not in line: continue
        q, a = line.split("=>",1)
        q, a = q.strip(), a.strip()
        e_s  = cosine_emb(user_e, phrase_embedding(q))
        n_s  = ngram_jaccard(user_text, q, 2)
        w_s  = word_overlap(user_text, q)
        # vocab definition boost: if user word is a known vocab word
        for kw in user_text.lower().split():
            if kw in VOCAB_DATA and kw in q:
                e_s = min(1.0, e_s + 0.15)
        score = 0.4*e_s + 0.35*n_s + 0.25*w_s
        if score > best_s:
            best_s = score; best_a = a
    if best_s >= threshold:
        return best_a, best_s
    return None, best_s

# ==============================================================================
#  11.  GENERATION  (nucleus / top-p sampling)
# ==============================================================================
def top_p_sample(probs_list, p=0.9):
    """Nucleus sampling: sample from smallest set covering p of probability"""
    indexed = sorted(enumerate(probs_list), key=lambda x:-x[1])
    cumsum  = 0.0
    nucleus = []
    for idx, prob in indexed:
        nucleus.append((idx, prob)); cumsum += prob
        if cumsum >= p: break
    total = sum(pr for _,pr in nucleus)
    ids   = [i for i,_ in nucleus]
    wts   = [pr/total for _,pr in nucleus]
    return random.choices(ids, weights=wts)[0]

def generate(prompt_text, max_new=56):
    tokens   = tokenise(prompt_text)
    cache_k  = [[] for _ in range(N_LAYER)]
    cache_v  = [[] for _ in range(N_LAYER)]
    response = []

    # -- pre-cache weight floats once per generate call (avoids .data in hot loops) -
    wte_f  = [[v.data for v in row] for row in SD["wte"]]
    wpe_f  = [[v.data for v in row] for row in SD["wpe"]]
    nsf_f  = [v.data for v in SD["ns_f"]]
    lc = []
    for li in range(N_LAYER):
        lc.append({
            "wq": [[v.data for v in row] for row in SD[f"l{li}.wq"]],
            "wk": [[v.data for v in row] for row in SD[f"l{li}.wk"]],
            "wv": [[v.data for v in row] for row in SD[f"l{li}.wv"]],
            "wo": [[v.data for v in row] for row in SD[f"l{li}.wo"]],
            "w1": [[v.data for v in row] for row in SD[f"l{li}.w1"]],
            "w2": [[v.data for v in row] for row in SD[f"l{li}.w2"]],
            "ns1":[v.data for v in SD[f"l{li}.ns1"]],
            "ns2":[v.data for v in SD[f"l{li}.ns2"]],
        })

    def _step(tid, pos):
        pid   = pos % BLOCK_SIZE
        tok_e = wte_f[tid]
        pos_e = wpe_f[pid]
        x     = [tok_e[j]+pos_e[j] for j in range(N_EMBD)]

        for li in range(N_LAYER):
            c    = lc[li]
            xn   = rmsnorm([Value(v) for v in x])
            ns1  = c["ns1"]
            xn   = [ns1[j] * xn[j].data for j in range(N_EMBD)]
            wq,wk,wv = c["wq"],c["wk"],c["wv"]
            q = [sum(wq[r][j]*xn[j] for j in range(N_EMBD)) for r in range(N_EMBD)]
            k = [sum(wk[r][j]*xn[j] for j in range(N_EMBD)) for r in range(N_EMBD)]
            v = [sum(wv[r][j]*xn[j] for j in range(N_EMBD)) for r in range(N_EMBD)]
            cache_k[li].append(k); cache_v[li].append(v)

            attn_out = []
            kh_li    = cache_k[li]
            kh_len2  = len(kh_li)
            t_start2 = max(0, kh_len2 - 8)
            for h in range(N_HEAD):
                hs   = h*HEAD_DIM
                qh   = q[hs:hs+HEAD_DIM]
                sc   = HEAD_DIM**-0.5
                win_k = kh_li[t_start2:]
                win_v = cache_v[li][t_start2:]
                m_sc = max(sum(qh[j]*win_k[t][hs+j] for j in range(HEAD_DIM))*sc
                           for t in range(len(win_k)))
                raw  = [math.exp(sum(qh[j]*win_k[t][hs+j]
                                     for j in range(HEAD_DIM))*sc - m_sc)
                        for t in range(len(win_k))]
                s    = sum(raw)+1e-9
                aw   = [r/s for r in raw]
                head_out = [sum(aw[t]*win_v[t][hs+j]
                                for t in range(len(aw)))
                            for j in range(HEAD_DIM)]
                attn_out.extend(head_out)

            wo = c["wo"]
            ap = [sum(wo[r][j]*attn_out[j] for j in range(N_EMBD)) for r in range(N_EMBD)]
            x  = [x[j]+ap[j] for j in range(N_EMBD)]

            xn2  = rmsnorm([Value(v) for v in x])
            ns2  = c["ns2"]
            xn2  = [ns2[j]*xn2[j].data for j in range(N_EMBD)]
            w1,w2= c["w1"],c["w2"]
            h1   = [math.tanh(sum(w1[r][j]*xn2[j] for j in range(N_EMBD)))
                    for r in range(4*N_EMBD)]
            h2   = [sum(w2[r][j]*h1[j] for j in range(4*N_EMBD))
                    for r in range(N_EMBD)]
            x    = [x[j]+h2[j] for j in range(N_EMBD)]

        xf  = rmsnorm([Value(v) for v in x])
        nsf = nsf_f
        xf  = [nsf[j]*xf[j].data for j in range(N_EMBD)]
        logits = [sum(wte_f[v][j]*xf[j] for j in range(N_EMBD))
                  for v in range(VOCAB_SIZE)]
        # temperature
        logits = [l/TEMPERATURE for l in logits]
        m_l = max(logits)
        exp_l = [math.exp(l-m_l) for l in logits]
        s_l  = sum(exp_l)+1e-9
        probs = [e/s_l for e in exp_l]
        return probs

    # encode prompt
    for pos, tid in enumerate(tokens[1:-1]):  # skip BOS/EOS
        _step(tid, pos)

    token_id = BOS
    offset   = len(tokens)-1
    for step in range(max_new):
        probs    = _step(token_id, offset+step)
        token_id = top_p_sample(probs, p=0.92)
        if token_id == BOS:
            break
        response.append(itos[token_id])

    return "".join(response).strip()

# ==============================================================================
#  12.  ONLINE LEARNING  (replay buffer + immediate fine-tune)
# ==============================================================================
REPLAY = deque(maxlen=64)
ONLINE_STEP = [load_progress()]   # continues from pretrain step

def learn_pair(q, a, steps=15):
    """Fine-tune on a new pair + replay, then save."""
    pair  = f"{q.strip().lower()} => {a.strip().lower()}"
    pairs = load_memory()
    if pair not in pairs:
        pairs.append(pair)
        save_memory(pairs)
    REPLAY.append(pair)

    cur_step = ONLINE_STEP[0]
    for s in range(steps):
        tokens = tokenise(pair)
        loss   = compute_loss(tokens)
        if loss:
            loss.backward()
            adamw_step(cur_step + s)
        ONLINE_STEP[0] += 1

    # replay 2 random recent pairs
    if len(REPLAY) >= 4:
        for rp in random.sample(list(REPLAY), min(2, len(REPLAY))):
            tokens = tokenise(rp)
            loss   = compute_loss(tokens)
            if loss:
                loss.backward()
                adamw_step(ONLINE_STEP[0])
                ONLINE_STEP[0] += 1

    save_weights()

# ==============================================================================
#  13.  VOCAB-AWARE RESPONSE BUILDER
#       If user mentions a vocab word -> include its definition in reply
# ==============================================================================
def vocab_response(user_text):
    words   = re.findall(r"[a-z]+", user_text.lower())
    matches = [(w, VOCAB_DATA[w]) for w in words if w in VOCAB_DATA]
    if not matches:
        return None
    w, (defn, cat, syns) = matches[0]
    syn_str = " and ".join(syns[:2]) if syns else ""
    reply = f"{w} means {defn}"
    if syn_str:
        reply += f", also related to {syn_str}"
    return reply

# ==============================================================================
#  14.  PLOTTING  (loss curve + embedding vectors)
# ==============================================================================
def plot_loss():
    try:
        import matplotlib.pyplot as plt
        if not os.path.exists(LOSS_LOG):
            print("no loss data yet"); return
        steps, losses = [], []
        for line in open(LOSS_LOG):
            parts = line.strip().split(",")
            if len(parts)==2:
                steps.append(int(parts[0])); losses.append(float(parts[1]))
        if not steps:
            print("no loss data"); return
        # smoothed
        w = 10
        smoothed = [sum(losses[max(0,i-w):i+1])/len(losses[max(0,i-w):i+1])
                    for i in range(len(losses))]
        plt.figure(figsize=(8,4))
        plt.plot(steps, losses, alpha=0.3, color="steelblue", label="raw loss")
        plt.plot(steps, smoothed, color="steelblue", lw=2, label="smoothed")
        plt.xlabel("Training Step"); plt.ylabel("Loss")
        plt.title("MicroBot Training Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig("loss_plot.png", dpi=120)
        # plt.show() - disabled for Pydroid (Agg backend); open loss_plot.png manually
        print("loss_plot.png saved - open it in your file manager")
    except Exception as e:
        print(f"plot failed: {e}")

def plot_vectors(words=None):
    """2D PCA projection of word embeddings"""
    try:
        import matplotlib.pyplot as plt
        if words is None:
            words = ["hello","love","ai","science","earth","food",
                     "robot","happy","sad","learn","water","time",
                     "python","brain","music","code","star","fast"]
        embs = []
        valid_words = []
        for w in words:
            e = word_embedding(w)
            if e:
                embs.append(e); valid_words.append(w)
        if len(embs) < 3:
            print("not enough words for plot"); return

        # manual 2-component PCA (no numpy needed)
        n, d  = len(embs), len(embs[0])
        # center
        mean  = [sum(embs[i][j] for i in range(n))/n for j in range(d)]
        X     = [[embs[i][j]-mean[j] for j in range(d)] for i in range(n)]
        # covariance (dxd) - use power iteration for top 2 eigenvectors
        def mat_vec(M, v):
            return [sum(M[i][j]*v[j] for j in range(d)) for i in range(d)]
        def cov_vec(X, v):
            # cov*v = X^T(Xv)/n
            Xv = [sum(X[i][j]*v[j] for j in range(d)) for i in range(n)]
            return [sum(X[i][j]*Xv[i] for i in range(n))/n for j in range(d)]
        def normalize(v):
            norm = math.sqrt(sum(x*x for x in v))+1e-9
            return [x/norm for x in v]
        def power_iter(X, iters=30):
            v = [random.gauss(0,1) for _ in range(d)]
            v = normalize(v)
            for _ in range(iters):
                v = cov_vec(X, v)
                v = normalize(v)
            return v
        pc1 = power_iter(X)
        # deflate
        X2  = [[X[i][j] - sum(X[i][k]*pc1[k] for k in range(d))*pc1[j]
                for j in range(d)] for i in range(n)]
        pc2 = power_iter(X2)
        # project
        p1  = [sum(X[i][j]*pc1[j] for j in range(d)) for i in range(n)]
        p2  = [sum(X[i][j]*pc2[j] for j in range(d)) for i in range(n)]

        plt.figure(figsize=(9,6))
        plt.scatter(p1, p2, s=60, alpha=0.7, color="steelblue")
        for i, w in enumerate(valid_words):
            plt.annotate(w, (p1[i], p2[i]), fontsize=9,
                         xytext=(4,4), textcoords="offset points")
        plt.title("Word Embedding Vectors (PCA 2D)")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig("vectors_plot.png", dpi=120)
        # plt.show() - disabled for Pydroid (Agg backend); open vectors_plot.png manually
        print("vectors_plot.png saved - open it in your file manager")
    except Exception as e:
        print(f"vector plot failed: {e}")

# ==============================================================================
#  15.  CHAT LOOP
# ==============================================================================
done = load_progress()
pct  = min(100, int(done/TOTAL_STEPS*100))
status = "fully trained" if done>=TOTAL_STEPS else f"{pct}% trained - run again to continue"
print(f"\nMicroBot ready [{status}]")
print("commands:")
print("  quit              - exit")
print("  teach: q => a     - manually teach a pair")
print("  plot loss         - show training loss curve")
print("  plot vectors      - show word embedding space")
print("  plot vectors w1 w2 w3 ...  - plot specific words\n")

mem_cache = load_memory()

while True:
    try:
        user = input("you: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nbye!"); break

    if not user: continue

    ul = user.lower()

    if ul in ("quit","exit","q"):
        print("bot: goodbye! see you next time."); break

    if ul.startswith("plot loss"):
        plot_loss(); continue

    if ul.startswith("plot vectors"):
        parts = ul.replace("plot vectors","").strip().split()
        if parts:
            plot_vectors(parts)
        else:
            plot_vectors()
        continue

    if ul.startswith("teach:"):
        body = user[6:].strip()
        if "=>" in body:
            q, a = body.split("=>",1)
            q, a = q.strip(), a.strip()
            learn_pair(q, a, steps=30)
            print(f"bot: learned! '{q}' -> '{a}'")
        else:
            print("bot: format is   teach: question => answer")
        continue

    # -- retrieval (fast path) ---------------------------------------------
    answer, score = retrieve(user)
    if answer:
        print(f"bot: {answer}")
        learn_pair(user, answer, steps=5)
        mem_cache = load_memory()
        continue

    # -- vocab definition check --------------------------------------------
    vr = vocab_response(user)
    if vr:
        print(f"bot: {vr}")
        learn_pair(user, vr, steps=8)
        mem_cache = load_memory()
        continue

    # -- generation (learning path) ----------------------------------------
    reply = generate(user)
    if len(reply) < 3:
        # fallback: check if any vocab word in user input
        words = re.findall(r"[a-z]+", ul)
        for w in words:
            if w in VOCAB_DATA:
                d,c,s = VOCAB_DATA[w]
                reply = f"i think {w} means {d}"
                break
        else:
            reply = "i am still learning, please teach me with  teach: your question => the answer"

    print(f"bot: {reply}")
    print("     [type correction or press Enter to skip]")
    try:
        correction = input("     > ").strip()
    except (EOFError, KeyboardInterrupt):
        correction = ""

    if correction:
        learn_pair(user, correction, steps=30)
        print(f"bot: got it, learned that now!")
    else:
        learn_pair(user, reply, steps=8)
    mem_cache = load_memory()
