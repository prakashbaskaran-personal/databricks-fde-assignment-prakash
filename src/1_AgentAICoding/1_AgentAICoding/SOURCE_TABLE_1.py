# Databricks notebook source
SOURCE_TABLE = "tiger_assignment.tiger_bronze.sales_transactions"

df = spark.table(SOURCE_TABLE)