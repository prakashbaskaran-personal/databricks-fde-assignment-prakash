# Databricks notebook source
import mlflow
mlflow.set_experiment("/Shared/multi-tool-agent")

# COMMAND ----------

# MAGIC %pip install  databricks-sql-connector
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_experiment("/Shared/multi-tool-agent")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests

w = WorkspaceClient()

tools = [
    {
        "name": "calculate_discount",
        "description": "Calculate 10 percent discount"
    }
]

# COMMAND ----------


workspace_url = dbutils.secrets.get(
    scope="workflow-url-scope",
    key="workflow-url",
).rstrip("/")
host = workspace_url.removeprefix("https://")
http_path = dbutils.secrets.get(
    scope="databricks-sql-http-path-scope",
    key="databricks-sql-http-path",
)
token = dbutils.secrets.get(scope="ai-search-scope", key="databricks-pat")
space_id = "01f1d2600871168385853bc45472e607"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# COMMAND ----------

# DBTITLE 1,Create Tool Wrapper
from databricks import sql
import os

def calculate_discount_tool(amount):
    # Use a tag (mutable) so it does not conflict with the Genie tool's
    # "tool_used" param when both tools run in the same MLflow run
    mlflow.set_tag(
        "tool_used",
        "discount_function"
    )
    mlflow.log_metric(
        "discount_input",
        amount
    )
    cursor.execute(f"""
        SELECT tiger_assignment.functions.calculate_discount({amount})
    """)
    result = cursor.fetchone()[0]
    mlflow.log_metric(
        "discount_output",
        result
    )
    return result

# COMMAND ----------

# DBTITLE 1,Test discount tool
connection = sql.connect(
    server_hostname=host.replace("https://", ""),
    http_path=http_path,
    access_token=token
)
cursor = connection.cursor()

test_result = calculate_discount_tool(5000)

# The tool logs to MLflow outside a run, which auto-starts one; end it
# so later mlflow.start_run() calls do not fail
mlflow.end_run()

test_result

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

def ask_genie(question):

    mlflow.log_param("tool_used", "genie")
    mlflow.log_param("genie_question", question)

    payload = {
        "content": question
    }

    response = requests.post(
        f"{workspace_url}/api/2.0/genie/spaces/{space_id}/start-conversation",
        headers=headers,
        json=payload
    )

    conversation_id = response.json()["conversation_id"]
    message_id = response.json()["message_id"]

    while True:

        response = requests.get(
            f"{workspace_url}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}",
            headers=headers
        )

        result = response.json()

        if result["status"] == "COMPLETED":

            for attachment in result.get("attachments", []):

                if "text" in attachment:

                    answer = attachment["text"]["content"]

                    mlflow.log_text(
                        answer,
                        "genie_response.txt"
                    )

                    return answer

            return "No answer returned"


# COMMAND ----------

import re

def multi_tool_agent(question):

    q = question.lower()

    # BOTH TOOLS
    if (
        "top revenue customer" in q
        and "discount" in q
    ):

        genie_answer = ask_genie(
            "Who is the top revenue customer?"
        )

        # Extract revenue value from Genie response
        match = re.search(
            r'(\d+(?:,\d{3})*(?:\.\d+)?)',
            genie_answer
        )

        if match:

            revenue = float(
                match.group(1).replace(",", "")
            )

            discount = calculate_discount_tool(
                revenue
            )

            return (
                f"Top customer revenue: "
                f"{revenue:,.2f}\n"
                f"10% discount: "
                f"{discount:,.2f}"
            )

        return genie_answer

    # TOOL 1
    if "discount" in q and re.search(r"\d+", q):

        amount = float(
            re.search(r"\d+", q).group()
        )

        discount = calculate_discount_tool(amount)

        return (
            f"10% discount on {amount:,.0f} "
            f"is {discount:,.0f}"
        )

    # TOOL 2
    if any(
        word in q
        for word in [
            "revenue",
            "customer",
            "sales",
            "country",
            "product"
        ]
    ):
        return ask_genie(question)

    return "I can answer sales questions or discount calculations."

# COMMAND ----------

# DBTITLE 1,Traced agent
def traced_agent(question):

    # End any run left open (e.g. auto-started by a tool call outside a run)
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():

        mlflow.log_param(
            "question",
            question
        )

        answer = multi_tool_agent(
            question
        )

        mlflow.log_text(
            answer,
            "answer.txt"
        )

        return answer

# COMMAND ----------

traced_agent(
    "Top 3 revenue customers"
)

# COMMAND ----------

traced_agent(
    "Calculate discount for 5000"
)

# COMMAND ----------

traced_agent(
    "Who is the top revenue customer and what is a 10% discount on their revenue?"
)

# COMMAND ----------

def ask_genie(question):

    mlflow.log_param(
        "tool_used",
        "genie"
    )

    return "No customers found"

# COMMAND ----------

traced_agent(
    "Top 3 revenue customers"
)