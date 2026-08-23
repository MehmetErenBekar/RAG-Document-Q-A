import os
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()


with open("file.txt", "r", encoding="utf-8") as file:
    text = file.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = text_splitter.split_text(text)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)

collection = chroma_client.create_collection(name="turing_paper")
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(
    documents=chunks, embeddings=embeddings.tolist(), ids=ids
)


def rag_query(question: str):
    q_embedding = embed_model.encode([question])
    results = collection.query(
        query_embeddings=q_embedding.tolist(), n_results=2
    )

    context_chunks = results["documents"][0]
    context_text = "\n\n".join(context_chunks)

    system_prompt = (
        "You are an assistant answering questions strictly based on the provided context. "
        "If the answer is not present in the context, say 'I cannot find the answer in the provided document.'"
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content, context_chunks



app = FastAPI(
    title="RAG Document Q&A API",
    description="Retrieval-Augmented Generation pipeline using ChromaDB and OpenAI",
)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(req: QuestionRequest):
    answer, context = rag_query(req.question)
    return {"question": req.question, "answer": answer, "context": context}



if __name__ == "__main__":
    print("\n--- Document Q&A CLI (type 'exit' to leave) ---\n")
    while True:
        user_input = input("Question: ").strip()
        if user_input.lower() in ["exit", "q", "quit"]:
            break
        if not user_input:
            continue
        ans, src = rag_query(user_input)
        print(f"\nAnswer: {ans}\n")
