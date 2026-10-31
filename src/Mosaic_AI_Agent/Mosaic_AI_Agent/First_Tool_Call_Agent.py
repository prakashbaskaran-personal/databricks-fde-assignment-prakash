# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog tiger_assignment

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema functions

# COMMAND ----------

# DBTITLE 1,Create UC Function
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION tiger_assignment.functions.calculate_discount(
# MAGIC     amount DOUBLE
# MAGIC )
# MAGIC RETURNS DOUBLE
# MAGIC RETURN amount * 0.10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT tiger_assignment.functions.calculate_discount(1000);

# COMMAND ----------

# MAGIC %pip install  databricks-sql-connector
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

tools = [
    {
        "name": "calculate_discount",
        "description": "Calculate 10 percent discount"
    }
]

# COMMAND ----------

# DBTITLE 1,Create Tool Wrapper
from databricks import sql
import os

def calculate_discount_tool(amount: float):

    host = dbutils.secrets.get(
        scope="workflow-url-scope",
        key="workflow-url",
    ).removeprefix("https://").rstrip("/")
    http_path = dbutils.secrets.get(
        scope="databricks-sql-http-path-scope",
        key="databricks-sql-http-path",
    )
    token = dbutils.secrets.get(scope="ai-search-scope", key="databricks-pat")
    with sql.connect(
        server_hostname=host,


        http_path=http_path,
        access_token=token
    ) as connection:

        cursor = connection.cursor()

        cursor.execute(f"""
        SELECT tiger_assignment.functions.calculate_discount({amount})
        """)

        return cursor.fetchone()[0]

# COMMAND ----------

calculate_discount_tool(5000)

# COMMAND ----------

# DBTITLE 1,Create Agent Logic
import re

def my_agent(question):

    q = question.lower()

    if "what is databricks" in q:

        return (
            "Databricks is a unified analytics and AI platform."
        )

    if "discount" in q:

        match = re.search(r"(\d+)", question)

        if match:

            amount = float(match.group(1))

            discount = calculate_discount_tool(amount)

            return (
                f"10% discount for {amount:,.0f} "
                f"is {discount:,.0f}"
            )

    return "I don't know."

# COMMAND ----------

# DBTITLE 1,Demonstrate Tool Calling
my_agent("What is Databricks?")

# COMMAND ----------

my_agent("Calculate discount for 2500")

# COMMAND ----------

# DBTITLE 1,Add MLflow Tracing
import mlflow

mlflow.set_experiment("/Shared/agent-demo")

with mlflow.start_run():

    result = my_agent(
        "Calculate discount for 5000"
    )

    mlflow.log_param(
        "question",
        "Calculate discount for 5000"
    )

    mlflow.log_text(
        result,
        "answer.txt"
    )

    print(result)