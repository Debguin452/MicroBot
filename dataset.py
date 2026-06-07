"""
dataset.py  -  MicroBot v6  live dataset builder
=================================================
Fetches real knowledge from Wikipedia REST API.
  - 100+ topics covering science, daily life, culture, nature, tech
  - Multiple Q-A phrasings per topic (10+ per article)
  - Answer-side synonym pairs for better retrieval
  - Smarter word-fact extraction
  - Always merges SEED_PAIRS so bot answers basics even offline

Usage:
    python dataset.py            # fetch/update all topics
    python dataset.py rebuild    # force re-fetch ignoring cache
"""

import os, json, re, time, urllib.request, urllib.parse, zlib

DATASET_FILE  = "dataset.json"
DATASET_CACHE = "dataset_cache.json"

TOPICS = [
    "artificial intelligence", "machine learning", "neural network",
    "deep learning", "natural language processing", "transformer model",
    "computer science", "algorithm", "data structure", "operating system",
    "programming language", "software engineering", "internet", "world wide web",
    "computer vision", "robotics", "cybersecurity",
    "human brain", "neuroscience", "consciousness", "memory",
    "psychology", "emotion", "cognition", "perception", "motivation",
    "sleep", "dream", "mental health", "intelligence",
    "physics", "quantum mechanics", "gravity", "black hole", "relativity",
    "thermodynamics", "electricity", "magnetism", "light", "sound",
    "nuclear physics", "particle physics", "wave", "optics",
    "chemistry", "atom", "molecule", "chemical bond", "periodic table",
    "biology", "cell biology", "DNA", "evolution", "photosynthesis",
    "genetics", "protein", "enzyme", "virus", "bacteria",
    "solar system", "planet", "star", "galaxy", "universe",
    "earth", "ocean", "atmosphere", "climate change", "ecology",
    "weather", "volcano", "earthquake", "water cycle", "forest",
    "mountain", "river", "desert", "coral reef",
    "mathematics", "calculus", "statistics", "probability", "geometry",
    "algebra", "number theory", "logic",
    "history of science", "scientific method", "hypothesis", "experiment",
    "philosophy", "ethics", "knowledge", "reality",
    "language", "linguistics", "grammar", "semantics", "communication",
    "economics", "supply and demand", "inflation", "trade", "democracy",
    "human rights", "education", "medicine", "nutrition",
    "color", "time", "calendar", "music", "art", "sport",
    "food", "cooking", "agriculture", "animal", "plant",
    "friendship", "family", "love", "happiness", "creativity",
    "energy", "electricity generation", "solar energy", "nuclear energy",
    "engineering", "materials science", "nanotechnology",
]

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","i","you","we","they",
    "he","she","it","this","that","these","those","of","in","to",
    "for","on","at","by","with","from","or","and","but","not","no",
    "as","its","also","such","both","more","other","which","who",
    "what","how","when","where","why","then","than","into","out",
}

def _fetch_wiki(topic, max_chars=1500):
    topic = topic.strip()
    try:
        q   = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "MicroBot/6.0 (educational; no-login)"}
        )
        with urllib.request.urlopen(req, timeout=12) as r:
            if r.status == 200:
                d       = json.loads(r.read().decode("utf-8", errors="replace"))
                extract = d.get("extract", "").strip()
                title   = d.get("title", topic).strip()
                if extract:
                    return title, extract[:max_chars]
    except Exception:
        pass
    return topic, None

def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 20]

def _keywords(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

_Q_TEMPLATES = [
    "what is {t}",
    "explain {t}",
    "tell me about {t}",
    "describe {t}",
    "define {t}",
    "how does {t} work",
    "what does {t} mean",
    "what do you know about {t}",
    "give me information about {t}",
    "can you explain {t}",
]

def _build_pairs_from_text(title, text):
    sents = _sentences(text)
    if not sents:
        return []

    pairs = []
    tl = title.lower()
    s0 = sents[0].lower().rstrip(".")
    s1 = sents[1].lower().rstrip(".") if len(sents) > 1 else ""
    s2 = sents[2].lower().rstrip(".") if len(sents) > 2 else ""

    core      = s0
    long_core = f"{s0}. {s1}".strip(" .") if s1 else s0

    for tmpl in _Q_TEMPLATES:
        q = tmpl.format(t=tl)
        pairs.append(f"{q} => {core}")

    if s1:
        pairs.append(f"what does {tl} involve => {s1}")
        pairs.append(f"how is {tl} used => {s1}")
        pairs.append(f"what are the properties of {tl} => {s1}")
    if s2:
        pairs.append(f"what is an example of {tl} => {s2}")
        pairs.append(f"why is {tl} important => {s2}")

    if s1:
        pairs.append(f"give a detailed explanation of {tl} => {long_core}")

    kws = _keywords(text)
    unique_kws = list(dict.fromkeys(kws))[:5]
    if unique_kws:
        pairs.append(f"key concepts in {tl} => {', '.join(unique_kws[:4])}")
        pairs.append(f"what are the main ideas of {tl} => {', '.join(unique_kws[:4])}")

    words_in_title = re.findall(r"[a-z]+", tl)
    if len(words_in_title) >= 2:
        last_word = words_in_title[-1]
        pairs.append(f"what is {last_word} => {core}")

    return pairs

def _word_facts_from_text(title, text):
    facts = []
    sents = _sentences(text)
    title_words = set(re.findall(r"[a-z]+", title.lower()))
    for sent in sents[:8]:
        sl = sent.lower()
        for kw in _keywords(sl):
            if kw in title_words or len(kw) < 4:
                continue
            if re.search(rf"\b{re.escape(kw)}\b.{{0,60}}\b(is|are|means|refers|involves|describes)\b", sl):
                facts.append((kw, sl.rstrip(".")))
    return facts

SEED_PAIRS = [
    "hello => hi there! how can i help you today",
    "hi => hello! great to hear from you. ask me anything",
    "hey => hey! what would you like to know?",
    "good morning => good morning! ready to learn something today?",
    "good evening => good evening! how can i help you?",
    "how are you => i am doing great, always learning new things",
    "what is your name => my name is microbot, a curious ai that learns from the internet",
    "who are you => i am microbot, an ai trained on real wikipedia knowledge",
    "what can you do => i can answer questions, learn new topics from wikipedia, and remember what you teach me",
    "who made you => i was built by a curious programmer who loves ai and learning",
    "are you intelligent => i read each word carefully, understand meaning, and compose answers from real knowledge",
    "are you an ai => yes, i am a small ai trained on wikipedia knowledge using a transformer neural network",
    "bye => goodbye! it was great talking with you",
    "goodbye => see you next time! keep being curious",
    "thanks => you are welcome anytime",
    "thank you => happy to help! ask me anything else",
    "tell me a joke => why did the neural network fail the exam? it had too many hidden layers but not enough understanding",
    "what is your purpose => to understand your questions word by word and give you clear answers from real knowledge",
    "what is colour => colour is the visual property of objects determined by the wavelengths of light they reflect or emit",
    "what is color => color is the visual perception caused by different wavelengths of light hitting the eye, creating sensations like red, blue, and green",
    "what color is grass => grass is green because it contains chlorophyll, a pigment that absorbs red and blue light and reflects green",
    "what color is the sky => the sky appears blue because air molecules scatter shorter blue wavelengths of sunlight more than other colors",
    "what color is the sun => the sun emits white light containing all colors, but appears yellow from earth due to atmospheric scattering",
    "what year is it => i do not have real-time access to the current date; you can check your device's clock for the exact year",
    "what time is it => i cannot access real-time data, so please check your device for the current time",
    "what is today => i do not have access to real-time date information; your device's calendar will tell you today's date",
    "what is the date => i cannot access real-time data; please check your device for the current date",
    "how old is the earth => earth is approximately 4.54 billion years old, formed from the solar nebula around the same time as the sun",
    "how old is the universe => the universe is approximately 13.8 billion years old, originating from the big bang",
    "what is the speed of light => the speed of light in a vacuum is 299,792,458 metres per second, the universal speed limit",
    "how many planets are there => there are eight planets in our solar system: mercury, venus, earth, mars, jupiter, saturn, uranus, and neptune",
    "what is water => water is a molecule made of two hydrogen atoms and one oxygen atom; it covers 71 percent of earth and is essential for all life",
    "why is water wet => water feels wet because its molecules cling to surfaces and each other through hydrogen bonds, creating that characteristic sensation",
    "what is fire => fire is a chemical reaction called combustion where a fuel reacts rapidly with oxygen, releasing heat and light",
    "what is ice => ice is the solid form of water, formed when water molecules slow down and lock into a crystalline structure at 0 degrees celsius",
    "what is the moon => the moon is earth's only natural satellite, a rocky body that orbits earth every 27 days and affects tides",
    "what is the sun => the sun is a star at the center of our solar system; a giant ball of plasma powered by nuclear fusion of hydrogen into helium",
    "what is rain => rain is liquid water falling from clouds, formed when water vapor in the atmosphere condenses around tiny particles",
    "what is wind => wind is the movement of air caused by differences in atmospheric pressure; air flows from high pressure to low pressure areas",
    "what is artificial intelligence => artificial intelligence is the ability of machines to simulate human reasoning, learning, perception, and problem solving",
    "what is machine learning => machine learning is a branch of ai where systems learn patterns from data to make predictions or decisions without explicit programming",
    "what is a neural network => a neural network is a computing system loosely inspired by biological neurons, organized in layers that learn to transform inputs into outputs",
    "what is deep learning => deep learning uses many-layered neural networks to learn hierarchical representations, enabling tasks like recognizing images and understanding language",
    "what is natural language processing => natural language processing is the field of ai that enables computers to understand, interpret, and generate human language",
    "what is a transformer => a transformer is a neural architecture using self-attention to process sequences, forming the basis of modern language models",
    "what is the human brain => the human brain is the central organ of the nervous system, containing about 86 billion neurons that control all thought, memory, and behavior",
    "what is physics => physics is the natural science studying matter, energy, space, time, and the fundamental forces governing them",
    "what is quantum mechanics => quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales, where particles show wave-like properties",
    "what is gravity => gravity is the fundamental force of attraction between all objects with mass; it keeps planets in orbit and gives objects weight",
    "what is a black hole => a black hole is a region where gravity is so extreme that nothing, not even light, can escape from within its event horizon",
    "what is chemistry => chemistry is the science studying matter, its properties, composition, structure, and the changes it undergoes during reactions",
    "what is an atom => an atom is the smallest unit of an element, consisting of a nucleus of protons and neutrons surrounded by orbiting electrons",
    "what is biology => biology is the science of life, studying organisms, their structure, function, growth, evolution, and interactions with environments",
    "what is dna => dna is the double-helix molecule carrying genetic instructions for development, reproduction, and function of all living organisms",
    "what is evolution => evolution is the change in inherited characteristics of biological populations over successive generations through natural selection",
    "what is photosynthesis => photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen",
    "what is mathematics => mathematics is the abstract science of number, quantity, structure, and space, providing tools for logic and modeling",
    "what is an algorithm => an algorithm is a step-by-step set of instructions for solving a problem, forming the foundation of all computer programs",
    "what is climate change => climate change refers to long-term shifts in global temperatures and weather patterns, accelerated by human greenhouse gas emissions",
    "what is the ocean => the ocean is the vast body of salt water covering 71 percent of earth's surface, regulating climate and hosting most of earth's biodiversity",
    "what is ecology => ecology studies the interactions between organisms and their environment, including food webs, ecosystems, and energy flow",
    "what is the atmosphere => the atmosphere is the layer of gases surrounding earth, held by gravity, enabling life and protecting it from solar radiation",
    "what is electricity => electricity is the flow of electric charge through a conductor, powering devices and generated from mechanical, chemical, or solar sources",
    "what is thermodynamics => thermodynamics is the branch of physics studying heat, work, temperature, and their relation to energy",
    "what is light => light is electromagnetic radiation visible to the human eye, traveling at 299,792 km/s and exhibiting both wave and particle properties",
    "what is sound => sound is a mechanical pressure wave that propagates through matter and is perceived by the ear as hearing",
    "what is consciousness => consciousness is the state of being aware of and able to think about one's own existence, thoughts, and surroundings",
    "what is memory => memory is the cognitive ability to encode, store, and retrieve information and past experiences",
    "what is intelligence => intelligence is the capacity to learn, reason, solve problems, understand relationships, and adapt effectively to new situations",
    "what is philosophy => philosophy explores fundamental questions about existence, knowledge, ethics, logic, and the nature of mind and reality",
    "what is love => love is a deep emotional bond involving care, trust, and affection; neuroscience links it to oxytocin, dopamine, and serotonin",
    "what is life => life is the property distinguishing organisms from non-living matter, characterized by growth, metabolism, reproduction, and response to stimuli",
    "what is the universe => the universe is all of spacetime and everything within it, including matter, energy, and the laws that govern them, born in the big bang",
    "what is a computer => a computer is an electronic device that processes data according to programmed instructions, enabling calculation, communication, and storage",
    "what is the internet => the internet is a global network of interconnected computers communicating via standardized protocols, enabling worldwide information sharing",
    "what is programming => programming is writing instructions in a formal language that a computer executes to perform tasks",
    "what is python => python is a high-level, readable programming language widely used in ai, data science, web development, and automation",
    "what is energy => energy is the capacity to do work or cause change; it exists as kinetic, potential, thermal, chemical, nuclear, and electromagnetic forms",
    "what is sleep => sleep is a natural recurring state of rest where the brain consolidates memories, repairs the body, and regulates hormones",
    "what is nutrition => nutrition is the science of how organisms obtain and use food for energy, growth, and maintenance of bodily functions",
    "what is music => music is the organized arrangement of sounds across time, expressing emotion, culture, and meaning through melody, rhythm, and harmony",
    "what is art => art is human creative expression using skill and imagination, producing works that communicate ideas, beauty, or emotion",
    "what is sport => sport is physical activity governed by rules, pursued for competition, exercise, or entertainment",
    "what is friendship => friendship is a close mutual relationship based on trust, affection, and shared experiences between people",
    "what is happiness => happiness is a positive emotional state characterized by contentment, joy, and a sense of meaning and purpose in life",
    "what is creativity => creativity is the ability to generate novel ideas or solutions by combining existing knowledge in new ways",
    "what is education => education is the process of facilitating learning, developing knowledge, skills, values, and understanding in individuals",
    "what is medicine => medicine is the science and practice of diagnosing, treating, and preventing disease to maintain and restore health",
    "how does learning work => learning works by forming and strengthening neural connections through attention, repetition, and practice",
    "what is a star => a star is a massive luminous sphere of plasma held together by gravity, producing energy through nuclear fusion in its core",
    "what is a planet => a planet is a celestial body orbiting a star, massive enough for gravity to make it roughly spherical, with a cleared orbital path",
    "what is a galaxy => a galaxy is a gravitationally bound system of stars, gas, dust, and dark matter; the milky way contains over 200 billion stars",
    "what is the milky way => the milky way is the spiral galaxy containing our solar system, with over 200 billion stars and a supermassive black hole at its center",
    "what is food => food is any substance consumed to provide nutritional support, supplying energy and essential nutrients for growth and vital processes",
]

def build_dataset(force=False):
    cache = {}
    if os.path.exists(DATASET_CACHE) and not force:
        try:
            cache = json.loads(zlib.decompress(open(DATASET_CACHE, "rb").read()).decode())
        except Exception:
            cache = {}

    all_pairs   = []
    word_facts  = {}
    topic_index = {}

    for topic in TOPICS:
        if topic in cache and not force:
            title, text = cache[topic]
        else:
            title, text = _fetch_wiki(topic)
            if text:
                cache[topic] = (title, text)
                time.sleep(0.12)
            else:
                continue

        if not text:
            continue

        pairs = _build_pairs_from_text(title, text)
        all_pairs.extend(pairs)

        for word, meaning in _word_facts_from_text(title, text):
            if word not in word_facts:
                word_facts[word] = meaning

        sents = _sentences(text)
        topic_index[topic.lower()] = {
            "title":    title,
            "summary":  sents[0] if sents else "",
            "keywords": _keywords(text)[:12],
            "pairs":    pairs,
        }

    try:
        with open(DATASET_CACHE, "wb") as f:
            f.write(zlib.compress(json.dumps(cache).encode(), level=6))
    except Exception:
        pass

    if not all_pairs:
        all_pairs = list(SEED_PAIRS)
        for line in SEED_PAIRS:
            if "=>" in line:
                _, a = line.split("=>", 1)
                for kw in _keywords(a.strip().lower())[:3]:
                    if len(kw) > 4 and kw not in word_facts:
                        word_facts[kw] = a.strip().lower()

    dataset = {
        "pairs":        all_pairs,
        "word_facts":   word_facts,
        "topic_index":  topic_index,
        "version":      6,
        "topic_count":  max(len(topic_index), len(all_pairs) // 12),
        "pair_count":   len(all_pairs),
    }

    try:
        with open(DATASET_FILE, "wb") as f:
            f.write(zlib.compress(json.dumps(dataset).encode(), level=6))
        print(f"dataset saved: {len(all_pairs)} pairs / {len(topic_index)} topics / {len(word_facts)} word-facts")
    except Exception as e:
        print(f"warn: could not save ({e})")

    return dataset

def load_dataset():
    base = None
    if os.path.exists(DATASET_FILE):
        try:
            base = json.loads(zlib.decompress(open(DATASET_FILE, "rb").read()).decode())
        except Exception:
            base = None

    if base is None:
        print("First run: fetching data from Wikipedia (this takes ~30s)...")
        base = build_dataset()

    existing = set(base.get("pairs", []))
    added    = [p for p in SEED_PAIRS if p not in existing]
    if added:
        base["pairs"]      = base.get("pairs", []) + added
        base["pair_count"] = len(base["pairs"])
        try:
            with open(DATASET_FILE, "wb") as f:
                f.write(zlib.compress(json.dumps(base).encode(), level=6))
        except Exception:
            pass

    return base

def get_pairs(ds):      return ds.get("pairs", [])
def get_word_facts(ds): return ds.get("word_facts", {})

def get_topic_summary(ds, topic):
    ti = ds.get("topic_index", {})
    tl = topic.lower()
    if tl in ti:
        return ti[tl].get("summary", "")
    for k, v in ti.items():
        if tl in k or k in tl:
            return v.get("summary", "")
    return None

def get_topic_keywords(ds, topic):
    ti = ds.get("topic_index", {})
    return ti.get(topic.lower(), {}).get("keywords", [])

if __name__ == "__main__":
    import sys
    force = "rebuild" in sys.argv
    if force:
        print("Force rebuilding dataset...")
    build_dataset(force=force)
