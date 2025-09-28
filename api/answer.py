import numpy as np
from .outsideapi import openai, get_database_client, get_openai_client
from .database import select_data
import ast


client = openai



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def genral_answer(question: str):
    # Validate input
    if not question or question.strip() == "":
        return "Please provide a valid question."
    
    # Get database client
    engine = get_database_client()
    if not engine:
        return "Database service is currently unavailable. Please try again later."
    
    disease_name = "chatbotintro"
    # Fetch documents from PostgreSQL
    chunks = select_data("documents", "context, embedding_context", "disease_name = :disease_name", {"disease_name": disease_name})

    if not chunks:
        return "No documents found for this disease."

    # Convert embedding strings to float arrays
    chunk_embeddings = []
    for c in chunks:
        try:
            # Handle both string and list formats
            if isinstance(c["embedding_context"], str):
                embedding = ast.literal_eval(c["embedding_context"])
            else:
                embedding = c["embedding_context"]
            chunk_embeddings.append(np.array(embedding, dtype=np.float32))
        except Exception as e:
            print(f"Error parsing embedding: {e}")
            print(f"Embedding type: {type(c['embedding_context'])}")
            print(f"Embedding value: {c['embedding_context'][:100]}...")
            # Skip this chunk if embedding parsing fails
            continue

    # Embed the question
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding
    q_emb = np.array(q_emb, dtype=np.float32)

    # Compute cosine similarities
    cosine_similarities = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]

    # Get top 3 chunks
    top_k = 3
    top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
    top_chunks = [chunks[i]["context"] for i in top_indices]
    context = "\n\n".join(top_chunks)

    # Build prompt
    prompt = f"""
    You are an AI assistant helping users with plant diseases.
    Only use the information provided below to answer the question.
    If the information does not contain the answer, say "can you provide perfect inforation which information you want."

    condition :
      if user ask long answer they use word like "please provide more information"
      so you can use out side information and give answer in just 4-5 lines Otherwise give answer in 1-2 lines
      if user ask ilegal question like "how to kill someone,how to do fraud" then just say "I'm sorry, I can't help with that."
      if user ask question like "how to make a bomb,how to do fraud" then just say "I'm sorry, I can't help with that."

    Context:
    {context}

    Question:
    {question}

    Answer clearly and concisely:
    """

    # Get OpenAI client
    openai_client = get_openai_client()
    if not openai_client:
        return "AI service is currently unavailable. Please try again later."
    
    # Generate answer
    answer = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
        top_p=0.8,
    ).choices[0].message.content.strip()

    return answer


    



def get_answer(question: str, disease_name: str ):
    # Validate input
    if not question or question.strip() == "":
        return "Please provide a valid question."
    
    # Get database client
    engine = get_database_client()
    if not engine:
        return "Database service is currently unavailable. Please try again later."
    
    # Fetch documents from PostgreSQL
    chunks = select_data("documents", "context, embedding_context", "disease_name = :disease_name", {"disease_name": disease_name})

    if not chunks:
        return "No documents found for this disease."

    # Convert embedding strings to float arrays
    chunk_embeddings = []
    for c in chunks:
        try:
            # Handle both string and list formats
            if isinstance(c["embedding_context"], str):
                embedding = ast.literal_eval(c["embedding_context"])
            else:
                embedding = c["embedding_context"]
            chunk_embeddings.append(np.array(embedding, dtype=np.float32))
        except Exception as e:
            print(f"Error parsing embedding: {e}")
            print(f"Embedding type: {type(c['embedding_context'])}")
            print(f"Embedding value: {c['embedding_context'][:100]}...")
            # Skip this chunk if embedding parsing fails
            continue

    # Embed the question
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding
    q_emb = np.array(q_emb, dtype=np.float32)

    # Compute cosine similarities
    cosine_similarities = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]

    # Get top 3 chunks
    top_k = 3
    top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
    top_chunks = [chunks[i]["context"] for i in top_indices]
    context = "\n\n".join(top_chunks)

    # Build prompt
    prompt = f"""
    You are an AI assistant helping users with plant diseases.
    Use the information provided below to answer the question about {disease_name}.
    
    Instructions:
    - Provide helpful information based on the context below
    - If the context contains relevant information, use it to answer the question
    - If you need to provide additional helpful information not in the context, you may do so
    - For long answers, use bullet points or lists when appropriate
    - Keep answers clear and concise (1-2 lines unless more detail is requested)
    - If asked about illegal activities, say "I'm sorry, I can't help with that."

    Context about {disease_name}:
    {context}

    Question:
    {question}

    Answer:
    """

    # Get OpenAI client
    openai_client = get_openai_client()
    if not openai_client:
        return "AI service is currently unavailable. Please try again later."
    
    # Generate answer
    answer = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
        top_p=0.8,
    ).choices[0].message.content.strip()

    return answer