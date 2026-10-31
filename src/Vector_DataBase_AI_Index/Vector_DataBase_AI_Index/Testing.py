# Databricks notebook source
# MAGIC %pip install databricks-ai-search
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import inspect
print(inspect.signature(vsc.create_direct_access_index))

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print(dir(w.vector_search_indexes))

# COMMAND ----------

help(w.vector_search_indexes.create_index)

# COMMAND ----------

spec = DeltaSyncVectorIndexSpecRequest()
print(spec)

# COMMAND ----------

help(EmbeddingSourceColumn)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM tiger_assignment.rag.documents;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE tiger_assignment.rag.documents;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM tiger_assignment.rag.documents;

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

w.vector_search_indexes.delete_index(
    index_name="tiger_assignment.rag.documents_vs_index"
)

print("Index 'tiger_assignment.rag.documents_vs_index' has been deleted.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

w.vector_search_indexes.delete_index(
    index_name="tiger_assignment.rag.documents_ai_index"
)

print("Index 'tiger_assignment.rag.documents_ai_index' has been deleted.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM tiger_assignment.rag.documents;

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# List all indexes registered to your endpoint
endpoint_name = "prakash-ai-search-index-v2"  # e.g., "ai_search_endpoint"
endpoint_info = vsc.get_endpoint(name=endpoint_name)

print("Endpoint Status:", endpoint_info.get("endpoint_status", {}).get("state"))
print("Indexes on this endpoint:")
for idx in endpoint_info.get("indexes", []):
    print(" -", idx.get("name"))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

indexes = client.list_indexes(endpoint_name="prakash-ai-search-index-v2")

for idx in indexes.get("vector_indexes", []):
    print(idx.get("name"))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

[m for m in dir(client) if not m.startswith("_")]

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

print(client.list_indexes('prakash-ai-search-index-v2'))

# COMMAND ----------

index = client.get_index(
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_ai_index_v2"
)

print(index.describe())

# COMMAND ----------

# Check if Change Data Feed is enabled on the table
detail_df = spark.sql("DESCRIBE DETAIL tiger_assignment.rag.documents")
properties = detail_df.select("properties").collect()[0]["properties"]

cdf_enabled = properties.get("delta.enableChangeDataFeed", "false")
print(f"delta.enableChangeDataFeed = {cdf_enabled}")

if cdf_enabled.lower() == "true":
    print("✅ Change Data Feed is ENABLED on tiger_assignment.rag.document")
else:
    print("❌ Change Data Feed is NOT enabled on tiger_assignment.rag.document")
    print("To enable it, run:")
    print("  ALTER TABLE tiger_assignment.rag.document SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM tiger_assignment.rag.document