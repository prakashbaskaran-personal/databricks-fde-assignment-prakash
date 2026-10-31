# Databricks notebook source
table_name = "tiger_assignment.tiger_bronze.sales_transactions"

result = spark.sql(f"select * from {table_name}")