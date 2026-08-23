# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions about a 
document by combining semantic search with an LLM, exposed both as a CLI and 
a REST API.

## How It Works

1. **Chunking** — The source document is split into overlapping text chunks 
   using LangChain's `RecursiveCharacterTextSplitter`, preserving context 
   across chunk boundaries.
2. **Embedding** — Each chunk is converted into a vector representation using 
   the `all-MiniLM-L6-v2` Sentence Transformer model.
3. **Storage & Retrieval** — Embeddings are stored in a ChromaDB collection. 
   When a question is asked, it's embedded the same way and matched against 
   the stored chunks using semantic similarity search.
4. **Generation** — The most relevant chunks are passed as context to an LLM 
   (GPT-4o-mini), which answers strictly based on the provided context — and 
   explicitly says so if the answer isn't present, avoiding hallucination.

## Example

**Question:** What original question did Turing replace because it was too ambiguous?

**Answer:** Turing replaced the original question "Can machines think?" because 
it was difficult to define.

**Question:** What's the capital of Turkey?

**Answer:** I cannot find the answer in the provided document.

This confirms the system grounds its answers in the source document rather 
than falling back on the model's general knowledge.

## Tech Stack

- **LangChain** — text splitting
- **Sentence Transformers** — embedding generation
- **ChromaDB** — vector storage and similarity search
- **OpenAI API** (GPT-4o-mini) — answer generation
- **FastAPI** — REST API layer

## Usage

### Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Create a `.env` file with your OpenAI API key:

\`\`\`
OPENAI_API_KEY=your_key_here
\`\`\`

### Run as CLI

\`\`\`bash
python main.py
\`\`\`

### Run as API

\`\`\`bash
uvicorn main:app --reload
\`\`\`

Then visit `http://localhost:8000/docs` for the interactive API documentation, 
or send a POST request:

\`\`\`bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the imitation game?"}'
\`\`\`

## Key Learnings

- Chunk overlap prevents important information from being split across 
  chunk boundaries and lost during retrieval.
- A low temperature (0.2) keeps answers grounded and consistent rather than 
  creative — important for factual Q&A.
- Explicitly instructing the model to admit when it can't find an answer in 
  the context is critical for preventing hallucination in RAG systems.

## Author

Mehmet Eren Bekar
