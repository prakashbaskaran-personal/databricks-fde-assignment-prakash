# Databricks notebook source
# calculator.py

def calculate_discount(amount, percentage):
    return amount * percentage / 100

# COMMAND ----------

calculate_discount(100, 10)

# COMMAND ----------

assert calculate_discount(100,10) == 10
assert calculate_discount(200,20) == 40
assert calculate_discount(50,5) == 2.5