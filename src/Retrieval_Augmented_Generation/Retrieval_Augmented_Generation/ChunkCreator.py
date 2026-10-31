# Databricks notebook source
# MAGIC %md
# MAGIC ## ChunkCreator

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import SparkSession

# Read customer JSON
customer_df = spark.read.option("multiline", "true").json(
    "/Volumes/workspace/default/customer_promotion/Promotion.json"
)

# Read promotion JSON
promotion_df = spark.read.option("multiline", "true").json(
    "/Volumes/workspace/default/customer_promotion/Customer_json.json"
)

# Combine both
documents_df = customer_df.unionByName(promotion_df)

# Create chunk_id
chunks_df = (
    documents_df
    .withColumn("chunk_id", F.expr("uuid()"))
)

display(chunks_df)


# COMMAND ----------

(chunks_df.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("bronze.document_chunks"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze.document_chunks