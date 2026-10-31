# Databricks notebook source
# MAGIC %md
# MAGIC # Indexing ; Retrieval

# COMMAND ----------

# MAGIC %pip install sentence-transformers faiss-cpu pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# 1. Install required open-source libraries in your notebook


import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 2. Sample Data
documents = [
    {"id": 1, "text": "Databricks Community Edition provides free Spark clusters."},
    {"id": 2, "text": "Vector search allows semantic querying over text embeddings."},
    {"id": 3, "text": "Python and PySpark can be used to run machine learning models."}
]

# 3. Generate Embeddings using Hugging Face models
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts, convert_to_numpy=True)

# 4. Build a FAISS Vector Index locally in memory
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. Perform Vector Similarity Search
query = "How do I query data semantically?"
query_vector = model.encode([query])

k = 2  # Number of nearest neighbors
distances, indices = index.search(query_vector, k)

# Display Top Matches
for idx in indices[0]:
    print(f"ID: {documents[idx]['id']} | Text: {documents[idx]['text']}")

# COMMAND ----------

import numpy as np
import faiss
import json

# 1. Assuming `df` is your PySpark DataFrame with the sample schema:
# (chunk_id, document, embedding, section, text)

# Convert the relevant columns to Pandas for local querying
# (Note: For very large datasets, sample or filter first before collecting to driver memory)
df = spark.sql("SELECT * FROM bronze.document_embeddings")
df.createOrReplaceTempView("document_embeddings")
pdf = df.select("chunk_id", "document", "section", "text", "embedding").toPandas()

# 2. Parse embeddings into a 2D NumPy array of type float32
# If the 'embedding' column is stored as a JSON string, parse it first:
if isinstance(pdf['embedding'].iloc[0], str):
    embeddings_list = pdf['embedding'].apply(json.loads).tolist()
else:
    embeddings_list = pdf['embedding'].tolist()

embeddings_array = np.array(embeddings_list, dtype=np.float32)

# 3. Create and populate the FAISS Index
dimension = embeddings_array.shape[1]  # Vector dimension (e.g., 384 or 768)
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# 4. Perform Vector Search using a sample query vector
# (Ensure query_vector dimension matches the indexed dimension)
query_vector = embeddings_array[0:1]  # Using the first row vector as a test query
k = 2  # Top-k nearest neighbors

distances, indices = index.search(query_vector, k)

# 5. Retrieve matching metadata records
for idx, dist in zip(indices[0], distances[0]):
    row = pdf.iloc[idx]
    print(f"Distance: {dist:.4f}")
    print(f"Chunk ID: {row['chunk_id']}")
    print(f"Document: {row['document']}")
    print(f"Section : {row['section']}")
    print(f"Text    : {row['text']}\n" + "-"*40)

# COMMAND ----------

# DBTITLE 1,SetenceTransofmer_index_1024_dimesnions
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

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
dimension = embeddings_array.shape[1]  # Automatically sets dimension to 1024
print(f"Index Dimension: {dimension}")

index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# 4. Load a 1024-dimensional SentenceTransformer model
# BAAI/bge-large-en-v1.5 outputs 1024 dimensions to match your index
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")


# ---------------------------------------------------------
# 5. Reusable Search Function
# ---------------------------------------------------------
def search_similar_chunks(query_text: str, top_k: int = 3):
    """
    Encodes a text query into an embedding vector, validates dimension match,
    queries the FAISS index, and returns top matching documents.
    """
    # Generate vector embedding for the search query
    query_vector = embedding_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    
    # Safety Check: ensure query vector dimension matches FAISS index dimension (1024)
    assert query_vector.shape[1] == index.d, (
        f"Dimension Mismatch Error! Query dimension ({query_vector.shape[1]}) "
        f"does not match FAISS index dimension ({index.d})."
    )
    
    # Execute vector search on FAISS index
    distances, indices = index.search(query_vector, top_k)
    
    # Extract results and attach distance scores
    results_df = pdf.iloc[indices[0]].copy()
    results_df['similarity_distance'] = distances[0]
    
    # Reorder columns for clean display
    return results_df[['similarity_distance', 'document', 'section', 'text', 'chunk_id']]


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------
results = search_similar_chunks("What is TPM Promotion Analytics?", top_k=2)
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer STTM

# COMMAND ----------

results = search_similar_chunks("How is CustomerID mapped?", top_k=2)
display(results)

# COMMAND ----------

results = search_similar_chunks("How is SalesQty transformed?", top_k=2)
display(results)

# COMMAND ----------

results = search_similar_chunks("What data quality checks are implemented", top_k=2)
display(results)

# COMMAND ----------

results = search_similar_chunks("What is the purpose of the Customer Sales solution", top_k=2)
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promotion

# COMMAND ----------

results = search_similar_chunks("What is the business key for Promotion", top_k=2)
display(results)

# COMMAND ----------

# What is the source table for Customer Sales?

# What is the target table for Customer Sales?

# What columns are mapped from the source to target?

# How is CustomerID mapped?

# How is SalesQty transformed?

# What data type is used for NetSales?

# What business rules are defined in Customer STTM?

# What data quality checks are implemented?

# --------Promotion------------------------
# What is the source system for Promotion data?

# What is the target table for Promotion data?

# How is PromotionID mapped?

# What is the business key for Promotion?

# What joins are used in Promotion STTM?

# What validations are applied to ExpectedRevenue?

# What load type is defined for Promotion?

# What is the watermark column?

# ------------2. DRD Related Queries---------------
# What is the purpose of the Customer Sales solution?

# What are the key fields in Customer Sales?

# What business requirements are defined?

# What reporting capabilities are supported?

# What table stores customer sales information?
# ------------------------------------------------

# Promotion DRD------------------
# What is the purpose of Promotion Analytics?

# What key fields exist in Promotion data?

# What promotion metrics are captured?

# What business rules are documented?
# ---------------------------Release Notes
# What features were delivered in Customer Sales STTM release 1.0?

# What enhancements were introduced?

# What known limitations exist?

# What future enhancements are planned?

# What deployment impact is expected?
# --------------------------
# What features were delivered in Promotion STTM?

# What business rules were introduced?

# What limitations exist in the flowback calculation?

# What future enhancements are planned for promotions?

# What reporting capabilities are enabled?

# ----------6. Promotion Analytics Queries
# How is Flowback % calculated?

# What are the known limitations of Flowback analysis?

# How does Promotion join to Customer data?

# How does Promotion join to Product data?

# What future improvements are planned for promotion attribution?

# How should sales be matched to promotions?

# ----------------------7. Comparison Queries
# Compare Customer Sales and Promotion STTM.

# Which document contains Flowback logic?

# Which solution uses Customer Master lookup?

# What business rules exist only in Promotion STTM?

# What reporting capabilities are common between both solutions?

# ----------------9. Natural Language Questions (Typical Demo Questions)
# Explain the Customer Sales solution.

# Explain the Promotion solution.

# What happens if CustomerID is null?

# How is a promotion validated before loading?

# What are the known limitations of the current implementation?

# What future enhancements are planned?

# Summarize the Customer STTM.

# Summarize the Promotion STTM.