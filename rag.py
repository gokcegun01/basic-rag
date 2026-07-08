import os, json, math
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("API_KEY"))
get_emb = lambda t: client.models.embed_content(model="gemini-embedding-2", contents=t).embeddings[0].values

def cosine_similarity(v1, v2):
    return sum(a*b for a,b in zip(v1, v2)) / (math.sqrt(sum(a*a for a in v1)) * math.sqrt(sum(b*b for b in v2)))

def build_vector_db():
    json_db = []
    
    if not os.path.exists("data"):
        print("Error: 'data' folder not found!")
        return
        
    for file_name in os.listdir("data"):
        if file_name.endswith(".txt"):
            with open(f"data/{file_name}", "r", encoding="utf-8") as f:
                chunks = [p.strip() for p in f.read().split("\n\n") if p.strip()]
                for chunk in chunks:
                    json_db.append({
                        "file_name": file_name,
                        "chunk_text": chunk,
                        "embedding": get_emb(chunk)
                    })
                    
    with open("vector_db.json", "w", encoding="utf-8") as f:
        json.dump(json_db, f, ensure_ascii=False, indent=4)

def is_db_outdated():
    if not os.path.exists("vector_db.json"):
        return True
        
    db_mtime = os.path.getmtime("vector_db.json")
    for file_name in os.listdir("data"):
        if file_name.endswith(".txt"):
            file_path = os.path.join("data", file_name)
            if os.path.getmtime(file_path) > db_mtime:
                return True
    return False

def retrieve_relevant_chunks(query, top_k=3):
    with open("vector_db.json", "r", encoding="utf-8") as f:
        json_db = json.load(f)
    
    query_emb = get_emb(query)
    scored_chunks = []
    for item in json_db:
        score = cosine_similarity(query_emb, item["embedding"])
        scored_chunks.append((score, item))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]

if __name__ == "__main__":
    if is_db_outdated():
        build_vector_db()

    user_query = input("Ask a question about the documents: ")
    best_matches = retrieve_relevant_chunks(user_query, top_k=3)
    
    context = ""
    sources = set()
    for score, item in best_matches:
        context += f"--- Document Source ({item['file_name']}) ---\n{item['chunk_text']}\n\n"
        sources.add(item['file_name'])
        
    prompt = f"""Answer the following question based ONLY on the provided context. ```
If the answer is not in the context, say "I don't know based on the documents". Do not make up information.

CONTEXT:
{context}

QUESTION:
{user_query}
"""

    print("\nGemini is thinking...")
    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    
    print("\n" + "="*50)
    print(f"ANSWER:\n{response.text}")
    print("="*50)
    print(f"SOURCES USED: {', '.join(sources)}")
    print("="*50)