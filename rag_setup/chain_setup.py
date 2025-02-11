# Databricks notebook source
# MAGIC %run ./00-helper

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

spark.sql(f"USE {catalog}.{dbName}")

# COMMAND ----------

# For this first basic demo, we'll keep the configuration as a minimum. In real app, you can make all your RAG as a param (such as your prompt template to easily test different prompts!)
vector_search_index_name = f"{catalog}.{dbName}.{vectorSearchIndexName}"
chain_config = {
    "llm_model_serving_endpoint_name": "databricks-meta-llama-3-3-70b-instruct",  # the foundation model we want to use
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,  # the endoint we want to use for vector search
    "vector_search_index": f"{catalog}.{dbName}.{vectorSearchIndexName}",
    "embeddings_endpoint": embeddings_endpoint,
    "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""", # LLM Prompt template
}

# COMMAND ----------

# We'll register the chain as an MLflow model and inspect the MLflow Trace to understand what is happening inside the chain 

from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import DatabricksEmbeddings
from databricks_langchain import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import mlflow

## Enable MLflow Tracing
# Traces will be logged to the active MLflow Experiment when calling invocation APIs on chains
mlflow.langchain.autolog()

## Load the chain's configuration
model_config = mlflow.models.ModelConfig(development_config=chain_config)

## Turn the Vector Search index into a LangChain retriever
vs_client = VectorSearchClient(disable_notice=True)

vector_search_as_retriever = DatabricksVectorSearch(
    vector_search_index_name,
    columns=["id", "url", "content"],
).as_retriever(search_kwargs={"k": 3})

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)


# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain import ChatDatabricks
from operator import itemgetter

prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", model_config.get("llm_prompt_template")), # Contains the instructions from the configuration
        ("user", "{question}") #user's questions
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

# COMMAND ----------

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    } #retrieval chain
    | prompt
    | model
    | StrOutputParser()
)

input_example = {"messages": [ {"role": "user", "content": "What is Retrieval-augmented Generation?"}]}

# COMMAND ----------

