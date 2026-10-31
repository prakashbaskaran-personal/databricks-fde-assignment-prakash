# Databricks notebook source
# MAGIC %md
# MAGIC ## Embedding

# COMMAND ----------

# MAGIC %pip install openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

for endpoint in w.serving_endpoints.list():
    print(endpoint.name)

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

response = client.predict(
    endpoint="databricks-bge-large-en",
    inputs={
        "input": [
            "CustomerID must not be null"
        ]
    }
)

print(response)

# COMMAND ----------

import mlflow.deployments
from pyspark.sql import Row

client = mlflow.deployments.get_deploy_client("databricks")

chunks_df = spark.table("bronze.document_chunks")
rows = chunks_df.collect()

embedding_records = []

for row in rows:

    response = client.predict(
        endpoint="databricks-bge-large-en",
        inputs={
            "input": [row["text"]]
        }
    )

    embedding_vector = response["data"][0]["embedding"]

    embedding_records.append({
        "chunk_id": row["chunk_id"],
        "document": row["document"],
        "section": row["section"],
        "text": row["text"],
        "embedding": embedding_vector
    })

# COMMAND ----------

embedding_df = spark.createDataFrame(
    embedding_records
)

display(embedding_df)

# COMMAND ----------

(
    embedding_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("bronze.document_embeddings")
)


# COMMAND ----------

display(
    spark.table("bronze.document_embeddings")
)

# COMMAND ----------

from pyspark.sql.functions import size

display(
    spark.table("bronze.document_embeddings")
         .withColumn(
             "embedding_dim",
             size("embedding")
         )
)

# COMMAND ----------

df = spark.sql("SELECT * FROM bronze.document_embeddings")
df.createOrReplaceTempView("document_embeddings")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM document_embeddings LIMIT 5