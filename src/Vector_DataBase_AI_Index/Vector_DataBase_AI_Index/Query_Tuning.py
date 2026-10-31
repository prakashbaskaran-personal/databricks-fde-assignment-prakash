# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

index = client.get_index(
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_ai_index_v2"
)

# COMMAND ----------

query = "customer revenue"

# COMMAND ----------

results_top3 = index.similarity_search(
    query_text=query,
    columns=["id", "document_name", "content"],
    num_results=3
)

print(results_top3)

# COMMAND ----------

results_top15 = index.similarity_search(
    query_text=query,
    columns=["id", "document_name", "content"],
    num_results=15
)

print(results_top15)

# COMMAND ----------

import pandas as pd

df3 = pd.DataFrame(
    results_top3["result"]["data_array"],
    columns=["id","document_name","content","score"]
)

df15 = pd.DataFrame(
    results_top15["result"]["data_array"],
    columns=["id","document_name","content","score"]
)

display(df3)
display(df15)

# COMMAND ----------

context3 = " ".join(
    [row[2] for row in results_top3["result"]["data_array"]]
)

context15 = " ".join(
    [row[2] for row in results_top15["result"]["data_array"]]
)

print("Top 3 characters:", len(context3))
print("Top 15 characters:", len(context15))