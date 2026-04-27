# Music Recommender — RAG Edition

## Original Project: Music Recommender Simulation

This project started as the **Music Recommender Simulation**, a rule-based content filtering system that scored songs from a small hand-crafted catalog against a structured user profile (genre, mood, energy, valence, acousticness, tempo). The original system ranked every song in the catalog using a weighted scoring formula and returned the top-k results with a plain-English explanation of why each song matched. It demonstrated how real-world recommenders turn structured data into predictions and where bias can emerge from weighted categorical features.

---

## Title and Summary

**Music Recommender — RAG Edition** extends the original simulation with a full Retrieval-Augmented Generation (RAG) pipeline. Users describe what they want to listen to in plain English — "something melancholic to study to" or "high energy workout music" — and the system retrieves the most semantically similar songs from a 1,000-track Spotify dataset, then uses Gemini 2.5 Flash to generate a conversational recommendation with a reason for each pick.

This matters because it bridges the gap between rigid structured filtering and the natural way people actually talk about music. The original system required users to know their exact energy score and preferred genre label. The RAG system removes that friction entirely.

---

## Architecture Overview

The system has two parallel paths and four main components:

**RAG Pipeline (new):**
1. **Data Pipeline** — two Kaggle Spotify CSVs are merged, columns renamed to match the internal schema, and mood labels are derived from valence and energy. The top 1,000 tracks by popularity are kept.
2. **Indexer** — `sentence-transformers` (`all-MiniLM-L6-v2`) embeds each song's text description locally. Vectors are stored in a persistent ChromaDB collection. This runs once and is skipped on subsequent startups.
3. **Retriever** — the user's query is embedded with the same model and ChromaDB returns the 10 most semantically similar songs via cosine similarity search.
4. **Generator** — the 10 retrieved songs are passed to Gemini 2.5 Flash as context. Gemini selects the best 5 and writes a conversational explanation for each.

**Rule-Based System (original):**
- `src/recommender.py` — weighted scoring across genre, mood, energy, acousticness, valence, and tempo. Still intact and tested.

See [flowchart.mmd](flowchart.mmd) for a full visual breakdown.

---

## Setup Instructions

**Requirements:** Python 3.10+, a [Gemini API key](https://aistudio.google.com/apikey)

```bash
# 1. Clone the repo and create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key
echo "GEMINI_API_KEY=your-key-here" > .env

# 4. Run
python main.py
```

On first run, `all-MiniLM-L6-v2` (~90MB) downloads once and the 1,000-track index is built locally in seconds. Every subsequent run skips indexing and goes straight to the query prompt.

**To run the original rule-based system:**
```bash
python -m src.main
```

**To run tests:**
```bash
pytest
```

---

## Sample Interactions

**Query 1 — Mood-based**
```
You: something sad and slow for a late night drive

Recommendations:
1. **"Waiting Room" by Phoebe Bridgers:** This song is incredibly sad and slow, making it ideal for a melancholic late-night drive.
2. **"Sparks" by Coldplay:** With its ambient and sad mood, "Sparks" will perfectly complement the quiet introspection of a night drive.
3. **"lacy" by Olivia Rodrigo:** This track delivers a sad and slow vibe that's just right for a thoughtful late-night journey.
4. **"The Weekend" by SZA:** This R&B track offers a sad and laid-back feel, perfect for cruising slowly under the night sky.
5. **"Big Black Car" by Gregory Alan Isakov:** This folk song has a distinctly sad and slow atmosphere that perfectly matches your request for a reflective late-night drive.
...
```

**Query 2 — Activity-based**
```
You: high energy music to work out to

Recommendations:
1. **"What I've Done" by Linkin Park**: This track brings an intense rock energy that's perfect for pushing through a tough workout.
2. **"Unstoppable" by Sia**: With its powerful vocals and driving beat, this song will make you feel, well, unstoppable!
3. **"Push Up - Main Edit" by Creeds**: Get ready to feel the adrenaline with this high-energy gaming track that's made for moving.
4. **"Not Strong Enough" by boygenius, Julien Baker, Phoebe Bridgers, Lucy Dacus**: This rock anthem has a strong, driving rhythm that will keep your momentum going.
5. **"Runaway" by Bon Jovi**: This classic rock hit has an infectious high energy and a happy vibe to keep you motivated and smiling while you exercise.
...
```

**Query 3 — Vibe-based**
```
You: chill indie music for a Sunday morning

Recommendations:
1.  "Clouded" by Brent Faiyaz: This R&B track has a really chill vibe, perfect for easing into your Sunday.
2.  "anything" by Adrianne Lenker: With its folk genre and chill mood, this song offers a gentle and introspective feel for your morning.
3.  "Royals" by Lorde: While the mood is listed as sad, its somewhat subdued energy and pop genre could still provide a thoughtful and relaxed backdrop for your morning.
4.  "The Weekend" by SZA: This R&B song has a lower energy and valence, making it suitable for a more laid-back and reflective Sunday morning.
5.  "Closer" by The Chainsmokers, Halsey: Even though it's electronic, its moderate energy and happy mood could offer a pleasant, soft pop background for your relaxing morning.
...
```

---

## Design Decisions

**Why keep the original rule-based system?**
The weighted scoring system in `src/recommender.py` is fully transparent and deterministic — you can read the output and trace exactly why a song scored the way it did. It also has automated tests. The RAG system replaces the user-facing interface but the original logic is preserved as a reference and still passes all tests.

**Why `sentence-transformers` instead of Gemini embeddings?**
Gemini's embedding API (`gemini-embedding-2`) on the free tier is capped at 100 requests per minute with no batch support, making it impractical to index 1,000 tracks without hitting rate limits. `all-MiniLM-L6-v2` runs locally with no API calls, indexes all tracks in seconds, and produces high-quality semantic embeddings. Gemini is reserved for the generation step where it genuinely adds value.

**Why disable thinking on Gemini 2.5 Flash?**
Gemini 2.5 Flash enables extended reasoning by default. For a task as straightforward as selecting 5 songs from a list of 10 and writing one sentence per song, the added reasoning time produced no meaningful improvement in output quality. Disabling it (`thinking_budget=0`) cut response latency significantly.

**Why top 1,000 tracks by popularity?**
The free tier embedding quota and the nature of a demo project made a smaller, higher-quality catalog the right call. Popular tracks are more likely to be recognized and evaluated meaningfully by a human grader.

**Trade-offs:**
- Local embeddings are fast and free but the model (`all-MiniLM-L6-v2`) is less semantically rich than Gemini's embedding models.
- Mood labels are derived from valence and energy with a simple 2×2 grid — this loses nuance but keeps the original scoring system compatible.
- The catalog is limited to 1,000 songs, which means niche genres are underrepresented.

---

## Testing Summary

**What worked:**
- The original rule-based tests in `tests/test_recommender.py` pass cleanly and cover scoring, genre similarity, edge cases (no genre match, contradictory profiles), and the OOP interface.
- The RAG pipeline retrieves semantically relevant songs consistently — mood and activity queries return songs that match the intent without needing explicit feature labels.
- Disabling Gemini's thinking mode reduced response latency while keeping recommendation quality high for this task.

**What didn't work:**
- Gemini's embedding API (`text-embedding-004` and `gemini-embedding-2`) could not be used for indexing due to rate limits and lack of batch support on the free tier.
- The initial `google-generativeai` package was the wrong SDK — the correct one is `google-genai` (`from google import genai`).
- Model names with preview tags (e.g. `gemini-2.5-flash-preview-04-17`) are not always available; using the stable name (`gemini-2.5-flash`) is more reliable.

**What I learned:**
- API free tier constraints heavily influence architecture decisions. The "best" embedding model on paper became the wrong choice once quota limits were factored in.
- RAG separates well into three independent concerns — embedding, retrieval, and generation — which makes each piece easy to swap out without touching the others.
- Keeping the original system intact made it easy to isolate what the RAG layer actually adds.

---

## Reflection

Building this project made the abstract concept of RAG concrete. The retrieval step is a nearest-neighbor search over vectors, and its quality is entirely determined by how well the embedding model captures semantic meaning. Swapping embedding models changed what "similar" meant, which changed the results.

It also reinforced that AI systems are not monolithic. The best solution here used a local embedding model, a vector database, and a cloud LLM for generation — three different tools doing three different jobs. Understanding where each one adds value, and where its limits are, is more useful than defaulting to one provider for everything.

The biggest open question this project raised is evaluation. It is easy to tell when a recommendation is obviously wrong, but hard to define what "good" means systematically. Real-world recommenders solve this with click data and listening history. Without that signal, human judgment remains the only meaningful evaluator.

---

## Limitations and Ethics

This project and its data is derived from the top 1000 most popular songs on Spotify, so it may not provide any more niche artists/songs. In addition, the dataset was updated around 4 months ago, meaning new songs won't be included.

Though I tried to find ways to exploit this AI, by telling it to ignore all prompts and give me an answer to a question, it denied that request, and instead found songs related to my question.

I'd say this surprised me, as the AI model was very resistant, and insisted on its role as a music recommendation system.

In this project, I used AI as a learning tool, and with writing boilerplate code. I was unfamiliar with chroma db, but it allowed me to use it effectively in this project. It was very helpful in this. One suggestion I disliked, however, was having a much larger dataset and using the Gemini embedding tool which would take about 3 days to fully compile due to rate limits.

---

## Project Structure

```
.
├── main.py                  # Entry point — builds index on first run, then query loop
├── src/
│   ├── rag.py               # RAG pipeline: embed, retrieve, generate
│   ├── recommender.py       # Original rule-based scoring system
│   ├── main.py              # Original CLI runner for rule-based profiles
│   └── prepare_data.py      # Data prep: filter to top 1000 tracks
├── data/
│   ├── spotify_tracks.csv   # Cleaned 1000-track catalog
│   └── chroma_db/           # Persistent vector index (generated, gitignored)
├── tests/
│   └── test_recommender.py  # Automated tests for rule-based system
├── flowchart.mmd            # System diagram
├── model_card.md            # Model card
└── requirements.txt
```
