# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
client = WorkspaceClient()

# COMMAND ----------

docs = [
    {
        "id": "1",
        "document_name": "Customer_Sales_DRD.docx",
        "content": "Customer sales report contains revenue information."
    },
    {
        "id": "2",
        "document_name": "Promotion_DRD.docx",
        "content": "Promotion records contain discount information."
    }
]

# COMMAND ----------

import inspect
print(inspect.signature(vsc.create_direct_access_index))


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

vsc.create_direct_access_index(
    embedding_vector_column='embedding',
    schema={'id': 'string', 'embedding': 'vector'},
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_direct_index",
    primary_key="id",
    embedding_dimension=1024
)

# COMMAND ----------

vsc.create_direct_access_index(
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_direct_index",
    primary_key="id",
    embedding_dimension=1024,  # adjust if needed
    embedding_vector_column="embedding",
    schema={
        "id": "string",
        "document_name": "string",
        "content": "string"
    }
)

# COMMAND ----------

vsc.create_direct_access_index(
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_direct_index",
    primary_key="id",
    embedding_dimension=1024,
    embedding_vector_column="embedding",
    schema={
        "id": "string",
        "document_name": "string",
        "content": "string",
        "embedding": "array<float>"
    }
)

# COMMAND ----------

help(vsc.create_direct_access_index)

# COMMAND ----------

direct_index = vsc.get_index(
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_direct_index"
)

print(direct_index.describe())

# COMMAND ----------

records = [
    {
        "id": "1",
        "document_name": "Customer_Sales_DRD.docx",
        "content": "Customer sales report contains revenue information."
    },
    {
        "id": "2",
        "document_name": "Promotion_DRD.docx",
        "content": "Promotion records contain discount information."
    }
]

# COMMAND ----------

print([m for m in dir(direct_index) if "upsert" in m.lower()])

# COMMAND ----------

help(direct_index.upsert)