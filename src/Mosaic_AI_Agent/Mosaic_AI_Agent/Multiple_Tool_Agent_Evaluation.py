# Databricks notebook source
# %sql
# use catalog tiger_assignment

# COMMAND ----------

# %sql
# create schema functions

# COMMAND ----------

# DBTITLE 1,Create UC Function
# %sql
# CREATE OR REPLACE FUNCTION tiger_assignment.functions.calculate_discount(
#     amount DOUBLE
# )
# RETURNS DOUBLE
# RETURN amount * 0.10;

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

def calculate_discount_tool(amount: float):
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

def ask_genie(question):
    
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
                    return attachment["text"]["content"]

            return "No answer returned"

# COMMAND ----------

# DBTITLE 1,Demonstrate Tool Calling
my_agent("What is Databricks?")

# COMMAND ----------

my_agent("Calculate discount for 2500")

# COMMAND ----------

ask_genie("top 3 revenue customers")

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

genie_answer = ask_genie(
    "Who is the top revenue customer?"
)

print(genie_answer)
import re

match = re.search(r"(\d+(?:,\d{3})*(?:\.\d+)?)", genie_answer)

if match:
    revenue = float(match.group(1).replace(",", ""))

# COMMAND ----------

discount = calculate_discount_tool(revenue)
print(discount)

# COMMAND ----------

multi_tool_agent(
    "Who is the top revenue customer and what is a 10% discount on their revenue?"
)

# COMMAND ----------

eval_set = [
    {
        "question": "Calculate discount for 1000",
        "expected": "100"
    },
    {
        "question": "Calculate discount for 5000",
        "expected": "500"
    },
    {
        "question": "Calculate discount for 0",
        "expected": "0"
    },
    {
        "question": "Top 3 revenue customers",
        "expected": "Walmart"
    },
    {
        "question": "Which customer has highest revenue?",
        "expected": "Walmart"
    },
    {
        "question": "Total sales by country",
        "expected": "USA"
    },
    {
        "question": "What is Databricks?",
        "expected": "analytics platform"
    },
    {
        "question": "Who is the top revenue customer and what is a 10% discount on their revenue?",
        "expected": "270"
    },
    {
        "question": "Calculate discount for -100",
        "expected": "error"
    },
    {
        "question": "Calculate discount for 999999",
        "expected": "99999.9"
    }
]

# COMMAND ----------

results = []

for test in eval_set:

    answer = multi_tool_agent(
        test["question"]
    )

    results.append({
        "question": test["question"],
        "expected": test["expected"],
        "actual": answer
    })

results

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(results)

display(df)

# COMMAND ----------

def evaluate(row):

    expected = row["expected"].lower()
    actual = str(row["actual"]).lower()

    if expected in actual:
        return "PASS"

    return "FAIL"

# COMMAND ----------

df["result"] = df.apply(
    evaluate,
    axis=1
)

display(df)

# COMMAND ----------

print("Evaluation Results")
print("==================")
print(f"Total Tests: {len(df)}")
print(f"Passes: {(df['result']=='PASS').sum()}")
print(f"Failures: {(df['result']=='FAIL').sum()}")