# Databricks notebook source
catalog = "austin_choi_demo_catalog"
dbName = "rag_demo"
volumeName = "ac_nov_rag_volume"
folderName = "sample_pdf_folder"
vectorSearchIndexName = "pdf_content_embeddings_index"
chunk_size = 500
chunk_overlap = 50
embeddings_endpoint = "databricks-gte-large-en"
VECTOR_SEARCH_ENDPOINT_NAME = "one-env-shared-endpoint-4"

# COMMAND ----------

chatBotModel = "databricks-meta-llama-3-3-70b-instruct"
max_tokens = 2000
finalchatBotModelName = "ac_nov_rag_bot"
yourEmailAddress = "austin.choi@databricks.com"

# COMMAND ----------

DATABRICKS_SITEMAP_URL = "https://docs.databricks.com/en/doc-sitemap.xml"