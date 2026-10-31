# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT id, document_name, category, content
# MAGIC FROM tiger_assignment.rag.documents;

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

index = client.get_index(
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_ai_index_v2"
)

# COMMAND ----------

results = index.similarity_search(
    query_text="report information",
    columns=["id", "document_name", "category", "content"],
    filters={"category": "Sales"},
    num_results=5
)

print(results)