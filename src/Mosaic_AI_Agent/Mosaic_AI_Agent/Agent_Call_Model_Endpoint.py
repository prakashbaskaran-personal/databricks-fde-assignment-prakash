# Databricks notebook source
# DBTITLE 1,Test 1: Discount Tool
import requests

token = dbutils.secrets.get(scope="ai-search-scope", key="databricks-pat")
workspace_url = dbutils.secrets.get(
    scope="workflow-url-scope",
    key="workflow-url",
).rstrip("/")
endpoint_url = (
    f"{workspace_url}/serving-endpoints/"
    "sales-multi-tool-agent-endpoint/invocations"
)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

payload = {
    "dataframe_records": [
        {
            "question": "Calculate discount for 1000"
        }
    ]
}

response = requests.post(
    endpoint_url,
    headers=headers,
    json=payload
)

print(response.json())

# COMMAND ----------

# DBTITLE 1,Test 2: Genie Tool
payload = {
    "dataframe_records": [
        {
            "question": "Top 3 revenue customers"
        }
    ]
}

response = requests.post(
    endpoint_url,
    headers=headers,
    json=payload
)

print(response.json())

# COMMAND ----------

# DBTITLE 1,Test 3: Multi-Tool Query
payload = {
    "dataframe_records": [
        {
            "question":
            "Who is the top revenue customer and what is a 10% discount on their revenue?"
        }
    ]
}

response = requests.post(
    endpoint_url,
    headers=headers,
    json=payload
)

print(response.json())

# COMMAND ----------

# DBTITLE 1,Option 2: Logged Interactions
mlflow.log_param("question", question)
mlflow.log_text(answer, "final_answer.txt")
mlflow.log_metric("latency_seconds", latency)