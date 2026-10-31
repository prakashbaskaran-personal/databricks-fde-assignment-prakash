# Databricks notebook source
from datetime import datetime
start_time = datetime.now()
print("Insert Start Time:", start_time)

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO tiger_assignment.rag.documents
# MAGIC VALUES (
# MAGIC   '4',
# MAGIC   'Freshness_Test_DRD.docx',
# MAGIC   'Testing',
# MAGIC   'FreshnessTest123 unique AI Search record for sync validation'
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM tiger_assignment.rag.documents
# MAGIC WHERE id='4';

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

index = client.get_index(
    endpoint_name="prakash-ai-search-index-v2",
    index_name="tiger_assignment.rag.documents_ai_index_v2"
)

results = index.similarity_search(
    query_text="FreshnessTest123",
    columns=["id","document_name","content"],
    num_results=5
)

print(results)

# COMMAND ----------

from datetime import datetime

end_time = datetime.now()

print("Retrievable Time:", end_time)

# COMMAND ----------

sync_time = end_time - start_time

print("Time to become searchable:", sync_time)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM tiger_assignment.rag.documents
# MAGIC WHERE id='4';