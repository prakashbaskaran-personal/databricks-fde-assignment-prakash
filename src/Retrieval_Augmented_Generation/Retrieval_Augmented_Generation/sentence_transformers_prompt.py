# Databricks notebook source
# MAGIC %md
# MAGIC # Indexing ; Retrieval ; ![Augmented](![path](path)) Generation (RAG)

# COMMAND ----------

# MAGIC %pip install sentence-transformers faiss-cpu pandas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
import mlflow.deployments

# 1. Fetch data from Spark table and convert to Pandas
df = spark.sql("SELECT * FROM bronze.document_embeddings")
df.createOrReplaceTempView("document_embeddings")
pdf = df.select("chunk_id", "document", "section", "text", "embedding").toPandas()

# 2. Parse embeddings into a 2D NumPy array of type float32
if isinstance(pdf['embedding'].iloc[0], str):
    embeddings_list = pdf['embedding'].apply(json.loads).tolist()
else:
    embeddings_list = pdf['embedding'].tolist()

embeddings_array = np.array(embeddings_list, dtype=np.float32)

# 3. Create and populate the FAISS Index
dimension = embeddings_array.shape[1]  # 1024 dimensions
print(f"Index Dimension: {dimension}")

index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# 4. Load a 1024-dimensional model
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

client = mlflow.deployments.get_deploy_client("databricks")


# ---------------------------------------------------------
# 5. Reusable Search Function
# ---------------------------------------------------------
def search_similar_chunks(query_text: str, top_k: int = 3):
    """Encodes text query, queries FAISS index, and returns matching rows as a DataFrame."""
    query_vector = embedding_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    
    assert query_vector.shape[1] == index.d, (
        f"Dimension Mismatch Error! Query dimension ({query_vector.shape[1]}) "
        f"does not match FAISS index dimension ({index.d})."
    )
    
    distances, indices = index.search(query_vector, top_k)
    results_df = pdf.iloc[indices[0]].copy()
    results_df['similarity_distance'] = distances[0]
    
    return results_df[['similarity_distance', 'document', 'section', 'text', 'chunk_id']]


# ---------------------------------------------------------
# 6. Prompt Builder Function (RAG Context Construction)
# ---------------------------------------------------------
def build_rag_prompt(user_query: str, top_k: int = 3) -> str:
    """
    Retrieves top matching context chunks and formats them 
    into an LLM prompt.
    """
    # Step A: Retrieve relevant records
    context_df = search_similar_chunks(user_query, top_k=top_k)
    
    # Step B: Format context snippets with metadata
    context_str = ""
    for idx, row in context_df.reset_index(drop=True).iterrows():
        context_str += (
            f"[Source {idx+1}]\n"
            f"Document: {row['document']}\n"
            f"Section: {row['section']}\n"
            f"Content: {row['text']}\n\n"
        )
    
    # Step C: Construct structured prompt
    prompt = f"""You are a helpful assistant answering questions based on the provided context documents.

Instructions:
1. Answer the question using ONLY the facts provided in the Context below.
2. If the answer cannot be found in the context, state "I cannot find the answer in the provided documents."
3. Cite the document name and section where applicable.

---
Context:
{context_str.strip()}
---

User Question: {user_query}

Answer:"""
    
    return prompt

def build_rag_prompt1(user_query: str, top_k: int = 3) -> str:
    context_df = search_similar_chunks(user_query, top_k=top_k)
    
    context_str = ""
    for idx, row in context_df.reset_index(drop=True).iterrows():
        context_str += (
            f"[Source {idx+1}]\n"
            f"Document: {row['document']}\n"
            f"Section: {row['section']}\n"
            f"Content: {row['text']}\n\n"
        )
    
    prompt = f"""You are a helpful assistant answering questions based on the provided context.

Instructions:
1. Summarize any relevant details found in the Context to answer the User Question.
2. If the context only gives partial information (e.g., just a name or subject), state what is known from the context.
3. If no relevant facts exist at all, state "I cannot find the answer in the provided documents."

---
Context:
{context_str.strip()}
---

User Question: {user_query}

Answer:"""
    
    return prompt

query = "What is TPM Promotion Analytics?"
rag_prompt = build_rag_prompt(query, top_k=2)

response = client.predict(
    endpoint="databricks-meta-llama-3-3-70b-instruct", # Or your active serving endpoint databricks-meta-llama-3-1-70b-instruct
    inputs={
        "messages": [
            {"role": "user", "content": rag_prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.1
    }
)

# Print the LLM's answer
print(response['choices'][0]['message']['content'])




# COMMAND ----------

query = "What business rules exist for promotions?"
rag_prompt = build_rag_prompt(query, top_k=2)

# Print the constructed prompt ready for an LLM
print(rag_prompt)

# COMMAND ----------

query = "What business rules exist for promotions?"
rag_prompt = build_rag_prompt1(query, top_k=2)

# Print the constructed prompt ready for an LLM
print(rag_prompt)

# COMMAND ----------



# COMMAND ----------

import numpy as np
import faiss
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1. Fetch data from Spark table and convert to Pandas
df = spark.sql("SELECT * FROM bronze.document_embeddings")
df.createOrReplaceTempView("document_embeddings")
pdf = df.select("chunk_id", "document", "section", "text", "embedding").toPandas()

# 2. Parse embeddings into a 2D NumPy array of type float32
if isinstance(pdf['embedding'].iloc[0], str):
    embeddings_list = pdf['embedding'].apply(json.loads).tolist()
else:
    embeddings_list = pdf['embedding'].tolist()

embeddings_array = np.array(embeddings_list, dtype=np.float32)

# 3. Create and populate the FAISS Index
dimension = embeddings_array.shape[1]  # 1024 dimensions
print(f"Index Dimension: {dimension}")

index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# 4. Load a 1024-dimensional embedding model
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 5. Load LLM Generator using modern `text-generation` pipeline
# 'Qwen/Qwen2.5-0.5B-Instruct' or 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' run well on CPU driver nodes
generator = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-0.5B-Instruct", 
    torch_dtype=torch.float32
)


# ---------------------------------------------------------
# 6. Reusable Search Function
# ---------------------------------------------------------
def search_similar_chunks(query_text: str, top_k: int = 3):
    """Encodes text query, queries FAISS index, and returns matching rows as a DataFrame."""
    query_vector = embedding_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    
    assert query_vector.shape[1] == index.d, (
        f"Dimension Mismatch Error! Query dimension ({query_vector.shape[1]}) "
        f"does not match FAISS index dimension ({index.d})."
    )
    
    distances, indices = index.search(query_vector, top_k)
    results_df = pdf.iloc[indices[0]].copy()
    results_df['similarity_distance'] = distances[0]
    
    return results_df[['similarity_distance', 'document', 'section', 'text', 'chunk_id']]


# ---------------------------------------------------------
# 7. Prompt Builder Function
# ---------------------------------------------------------
def build_rag_prompt(user_query: str, top_k: int = 3) -> str:
    context_df = search_similar_chunks(user_query, top_k=top_k)
    
    context_str = ""
    for idx, row in context_df.reset_index(drop=True).iterrows():
        context_str += (
            f"[Source {idx+1}]\n"
            f"Document: {row['document']}\n"
            f"Section: {row['section']}\n"
            f"Content: {row['text']}\n\n"
        )
    
    prompt = f"""You are a helpful assistant answering questions based on the provided context.

Instructions:
1. Summarize any relevant details found in the Context to answer the User Question.
2. If the context only gives partial information (e.g., just a name or subject), state what is known from the context.
3. If no relevant facts exist at all, state "I cannot find the answer in the provided documents."

---
Context:
{context_str.strip()}
---

User Question: {user_query}

Answer:"""
    
    return prompt


# ---------------------------------------------------------
# 8. Execution & LLM Generation
# ---------------------------------------------------------
query = "What is TPM Promotion Analytics?"
rag_prompt = build_rag_prompt(query, top_k=2)

# Pass prompt formatted as messages for chat models
messages = [{"role": "user", "content": rag_prompt}]
llm_output = generator(messages, max_new_tokens=256, do_sample=False)

# Extract and print generated response
print(llm_output[0]["generated_text"][-1]["content"])

# COMMAND ----------

# ---------------------------------------------------------
# 8. Execution & LLM Generation
# ---------------------------------------------------------
query = "What business rules exist for promotions?"
rag_prompt = build_rag_prompt(query, top_k=2)

# Pass prompt formatted as messages for chat models
messages = [{"role": "user", "content": rag_prompt}]
llm_output = generator(messages, max_new_tokens=256, do_sample=False)

# Extract and print generated response
print(llm_output[0]["generated_text"][-1]["content"])

# COMMAND ----------

test_questions = [
    "What is TPM Promotion Analytics?",
    "How is CustomerID mapped?",
    "What business rules exist for promotions?",
    "What source system provides promotion data?",
    "What are the validation rules?"
]

# COMMAND ----------

# ============================================================
# STEP 2 - RAG Function
# ============================================================

def answer_question(question, top_k=3):

    # Retrieve relevant chunks
    retrieved_chunks = search_similar_chunks(
        question,
        top_k=top_k
    )

    # Build Context
    context = ""

    for idx, row in retrieved_chunks.iterrows():

        context += f"""
[Source {idx+1}]
Document: {row['document']}
Section: {row['section']}
Chunk ID: {row['chunk_id']}

Content:
{row['text']}

"""

    # Build Prompt
    prompt = f"""
You are an assistant answering questions about STTM documents.

Use ONLY the information supplied in the context.

If the answer cannot be found in the context,
say 'Information not found in provided documents'.

Provide citations for every answer.

CONTEXT
========
{context}

QUESTION
========
{question}

ANSWER FORMAT
=============
Answer:

Sources:
- Document | Section | Chunk ID
"""

    # Call Databricks FM API
    response = client.predict(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        inputs={
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500
        }
    )

    answer = response["choices"][0]["message"]["content"]

    return answer, retrieved_chunks

# COMMAND ----------

# ============================================================
# STEP 3 - Execute Questions
# ============================================================

results = []

for question in test_questions:

    print("\n")
    print("=" * 100)
    print("QUESTION:")
    print(question)

    answer, chunks = answer_question(question)

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED SOURCES:")

    for _, row in chunks.iterrows():

        print(
            f"""
            Document : {row['document']}
            Section  : {row['section']}
            Chunk ID : {row['chunk_id']}
            """
        )

    results.append({
    "question": question,
    "answer": answer,
    "sources": ", ".join(
        [
            f"{r['document']}:{r['section']}"
            for _, r in chunks.iterrows()
        ]
    )
    })

# COMMAND ----------

# ============================================================
# STEP 4 - Save Results
# ============================================================

results_df = spark.createDataFrame(results)

display(results_df)