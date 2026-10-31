# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog tiger_assignment

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA rag

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE tiger_assignment.rag.documents (
# MAGIC     id STRING,
# MAGIC     document_name STRING,
# MAGIC     category STRING,
# MAGIC     content STRING
# MAGIC )
# MAGIC TBLPROPERTIES (
# MAGIC     delta.enableChangeDataFeed = true
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TBLPROPERTIES tiger_assignment.rag.documents;

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO tiger_assignment.rag.documents VALUES
# MAGIC (
# MAGIC '1',
# MAGIC 'Customer_Sales_DRD.docx',
# MAGIC 'Sales',
# MAGIC 'Customer sales report contains revenue, customer id, fiscal year and sales metrics.'
# MAGIC ),
# MAGIC (
# MAGIC '2',
# MAGIC 'Promotion_DRD.docx',
# MAGIC 'Promotion',
# MAGIC 'Promotion records contain promotion id, start date, end date and discount amount.'
# MAGIC ),
# MAGIC (
# MAGIC '3',
# MAGIC 'Inventory_DRD.docx',
# MAGIC 'Inventory',
# MAGIC 'Inventory records track product stock quantity and warehouse location.'
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Step 3: Create Vector Search Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType

w = WorkspaceClient()

endpoint_name = "prakash-vector-endpoint"

try:
    w.vector_search_endpoints.create_endpoint(
        name=endpoint_name,
        endpoint_type=EndpointType.STANDARD
    )
except Exception as e:
    print(e)

# COMMAND ----------

w.vector_search_endpoints.get_endpoint(endpoint_name)

# COMMAND ----------

for ep in w.serving_endpoints.list():
    print(ep.name)

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,End-to-end index build
from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

index = client.get_index(
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_ai_index_v2"
)

results = index.similarity_search(
    query_text="customer revenue",
    columns=["id", "document_name", "content"],
    num_results=3
)

print(results)