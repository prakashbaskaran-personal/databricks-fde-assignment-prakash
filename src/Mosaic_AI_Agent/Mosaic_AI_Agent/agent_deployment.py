# Databricks notebook source
import mlflow
import pandas as pd

class MultiToolAgent(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input):

        question = model_input["question"][0]

        answer = multi_tool_agent(question)

        return [answer]

# COMMAND ----------

with mlflow.start_run():

    mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=MultiToolAgent()
    )

    run_id = mlflow.active_run().info.run_id

print(run_id)

# COMMAND ----------

model_uri = f"runs:/{run_id}/agent"

mlflow.register_model(
    model_uri=model_uri,
    name="tiger_assignment.tiger_bronze.sales_transactions"
)