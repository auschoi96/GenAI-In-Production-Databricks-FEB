# Databricks notebook source
# MAGIC %md
# MAGIC #Step 0
# MAGIC
# MAGIC Go to config and update resource names as you prefer

# COMMAND ----------

# DBTITLE 1,This cell will set up the demo data we need
# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC #Set up a RAG Example 
# MAGIC
# MAGIC We need to demonstrate the evaluation capabilities. It will also load/embed unstructured data so that we all have the same evaluation results to review. 
# MAGIC
# MAGIC Please remember to shutdown these resources to avoid extra costs. This command will create the following:
# MAGIC
# MAGIC 1. Necessary catalogs, schemas and volumes to store the PDFs and embeddings 
# MAGIC 2. A call to GTE to create embeddings for the PDFs 
# MAGIC 3. VectorSearchIndex based on the PDFs embeddings generated in step 2 
# MAGIC 4. Spin up a VectorSearchEndpoint 
# MAGIC 5. Sync the VectorSearchIndex with your VectorSearchEndpoint 
# MAGIC
# MAGIC Later, we will set up the langchain chain to interact with these RAG resources

# COMMAND ----------

# DBTITLE 1,Set up a Demo RAG Bot for Evaluation Later
# MAGIC %run ./rag_setup/rag_setup

# COMMAND ----------

from IPython.display import Markdown
from openai import OpenAI
import os
dbutils.widgets.text("catalog_name", catalog)
dbutils.widgets.text("agent_schema", agent_schema)
dbutils.widgets.text("demo_schema", demo_schema)
base_url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'

# COMMAND ----------

# MAGIC %md
# MAGIC #Get started immediately with your Data with AI Functions
# MAGIC
# MAGIC We have a number of AI Functions designed as SQL functions that you can use in a SQL cell or SQL editor and use LLMs directly on your data immediately
# MAGIC
# MAGIC 1. ai_analyze_sentiment
# MAGIC 2. ai_classify
# MAGIC 3. ai_extract
# MAGIC 4. ai_fix_grammar
# MAGIC 5. ai_gen
# MAGIC 6. ai_mask
# MAGIC 7. ai_similarity
# MAGIC 8. ai_summarize
# MAGIC 9. ai_translate
# MAGIC 10. ai_query
# MAGIC
# MAGIC We will run a demo each of these functions below. 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_fix_grammar
# MAGIC The ai_fix_grammar() function allows you to invoke a state-of-the-art generative AI model to correct grammatical errors in a given text using SQL. This function uses a chat model serving endpoint made available by Databricks Foundation Model APIs.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_fix_grammar.html

# COMMAND ----------

# MAGIC %sql
# MAGIC -- verify that we're running on a SQL Warehouse
# MAGIC SELECT assert_true(current_version().dbsql_version is not null, 'YOU MUST USE A SQL WAREHOUSE, not a cluster');
# MAGIC
# MAGIC SELECT ai_fix_grammar('This sentence have some mistake');

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_similarity
# MAGIC The ai_similarity() function invokes a state-of-the-art generative AI model from Databricks Foundation Model APIs to compare two strings and computes the semantic similarity score using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_similarity.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_similarity('Databricks', 'Apache Spark'),  ai_similarity('Apache Spark', 'The Apache Spark Engine');

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_gen
# MAGIC The ai_gen() function invokes a state-of-the-art generative AI model to answer the user-provided prompt using SQL. This function uses a chat model serving endpoint made available by Databricks Foundation Model APIs.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_gen.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review, ai_gen('Generate a response to the following review: ' || review) as answer
# MAGIC from identifier(:catalog_name||'.'||:demo_schema||'.'||'reviews')
# MAGIC Limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_extract
# MAGIC The ai_extract() function allows you to invoke a state-of-the-art generative AI model to extract entities specified by labels from a given text using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_extract.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review, ai_extract(review, array("store", "product")) as Keywords
# MAGIC from identifier(:catalog_name||'.'||:demo_schema||'.'||'reviews')
# MAGIC Limit 3;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_analyze_sentiment
# MAGIC The ai_analyze_sentiment() function allows you to invoke a state-of-the-art generative AI model to perform sentiment analysis on input text using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_analyze_sentiment.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review, ai_analyze_sentiment(review) as Sentiment
# MAGIC from identifier(:catalog_name||'.'||:demo_schema||'.'||'reviews')
# MAGIC Limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_classify
# MAGIC The ai_classify() function allows you to invoke a state-of-the-art generative AI model to classify input text according to labels you provide using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_classify.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT country, ai_classify(country, ARRAY("APAC", "AMER", "EU")) as Region
# MAGIC from identifier(:catalog_name||'.'||:demo_schema||'.'||'franchises')
# MAGIC limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_translate
# MAGIC The ai_translate() function allows you to invoke a state-of-the-art generative AI model to translate text to a specified target language using SQL. During the preview, the function supports translation between English (en) and Spanish (es) only.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_translate.html
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review, ai_translate(review, "es")
# MAGIC from identifier(:catalog_name||'.'||:demo_schema||'.'||'reviews')
# MAGIC limit 3;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_mask
# MAGIC The ai_mask() function allows you to invoke a state-of-the-art generative AI model to mask specified entities in a given text using SQL. 
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_mask.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT first_name, last_name, (first_name || " " || last_name || " lives at " || address) as unmasked_output, ai_mask(first_name || "" || last_name || " lives at " || address, array("person", "address")) as Masked_Output
# MAGIC from identifier(:catalog_name||'.'||:demo_schema||'.'||'customers')
# MAGIC limit 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_query
# MAGIC The ai_query() function allows you to query machine learning models and large language models served using Mosaic AI Model Serving. To do so, this function invokes an existing Mosaic AI Model Serving endpoint and parses and returns its response. Databricks recommends using ai_query with Model Serving for batch inference
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/large-language-models/ai-functions.html#ai_query

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT
# MAGIC   `Misspelled Make`,   -- Placeholder for the input column
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-1-70b-instruct',
# MAGIC     CONCAT(format_string('You will always receive a make of a car. Check to see if it is misspelled and a real car. Correct the mistake. Only provide the corrected make. Never add additional details'), `Misspelled Make`)    -- Placeholder for the prompt and input
# MAGIC   ) AS ai_guess  -- Placeholder for the output column
# MAGIC FROM identifier(:catalog_name||'.'||:demo_schema||'.'||'synthetic_car_data')
# MAGIC Limit 3;  -- Placeholder for the table name
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Productionalizing Custom Tools 
# MAGIC
# MAGIC What you just saw were built in, out of the box solutions you can use immediately on your data. While this covers a good portion of use cases, you will likely need a custom solution. 
# MAGIC
# MAGIC ### Mosaic AI Tools on Unity Catalog
# MAGIC
# MAGIC You can create and host functions/tools on Unity Catalog! You get the benefit of Unity Catalog but for your functions! 
# MAGIC
# MAGIC While you can create your own tools using the same code that you built your agent (i.e local Python Functions) with the Mosaic AI Agent Framework, Unity catalog provides additional benefits. Here is a comparison 
# MAGIC
# MAGIC 1. **Unity Catalog function**s: Unity Catalog functions are defined and managed within Unity Catalog, offering built-in security and compliance features. Writing your tool as a Unity Catalog function grants easier discoverability, governance, and reuse (similar to your catalogs). Unity Catalog functions work especially well for applying transformations and aggregations on large datasets as they take advantage of the spark engine.
# MAGIC
# MAGIC 2. **Agent code tools**: These tools are defined in the same code that defines the AI agent. This approach is useful when calling REST APIs, using arbitrary code or libraries, or executing low-latency tools. However, this approach lacks the built-in discoverability and governance provided by Unity Catalog functions.
# MAGIC
# MAGIC Unity Catalog functions have the same limitations seen here: https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html 
# MAGIC
# MAGIC Additionally, the only external framework these functions are compatible with is Langchain 
# MAGIC
# MAGIC So, if you're planning on using complex python code for your tool, you will likely just need to create Agent Code Tools. 
# MAGIC
# MAGIC Below is an implementation of both

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agent Code Tools

# COMMAND ----------

# DBTITLE 1,Define the Tool
import requests
def pokemon_lookup(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        pokemon_data = response.json()
        pokemon_info = {
            "name": pokemon_data["name"],
            "height": pokemon_data["height"],
            "weight": pokemon_data["weight"],
            "abilities": [ability["ability"]["name"] for ability in pokemon_data["abilities"]],
            "types": [type_data["type"]["name"] for type_data in pokemon_data["types"]],
            "stats_name": [stat['stat']['name'] for stat in pokemon_data["stats"]],
            "stats_no": [stat['base_stat'] for stat in pokemon_data["stats"]]
        }
        results = str(pokemon_info)
        return results
    else:
        return None

# COMMAND ----------

# DBTITLE 1,Construct the payload to include the tool
import json
from openai import RateLimitError

# A token and the workspace's base FMAPI URL are needed to talk to endpoints
fmapi_token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)
fmapi_base_url = (
    base_url
)

openai_client = OpenAI(api_key=fmapi_token, base_url=fmapi_base_url)
MODEL_ENDPOINT_ID = "databricks-meta-llama-3-3-70b-instruct"

prompt = """You are a pokemon master and know every single pokemon ever created by the Pokemon Company. You will be helping people answer questions about pokemon. Stick strictly to the information provided to you to answer the question"""

def run_conversation(input):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": input}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "pokemon_lookup",
                "description": "Get information about a pokemon. This tool should be used to check to see if the pokemon is real or not as well.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pokemon": {
                            "type": "string",
                            "description": "The pokemon the user is asking information for e.g bulbasaur",
                        },
                    },
                    "required": ["pokemon"],
                },
            },
        }
    ]
    #We've seen this response package in the past cells
    response = openai_client.chat.completions.create(
        model=MODEL_ENDPOINT_ID,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    print(f"## Call #1 The Reasoning from the llm determining to use the function call:\n\n {response_message}\n")
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "pokemon_lookup": pokemon_lookup,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                pokemon_name=function_args.get("pokemon")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": function_response,
                }
            )  # extend conversation with function response
        print(f"## Call #2 Prompt sent to LLM with function call results giving us the answer:\n\n {messages}\n")
        second_response = openai_client.chat.completions.create(
            model=MODEL_ENDPOINT_ID,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response


# COMMAND ----------

# DBTITLE 1,Test the tool!
input1 = "Tell me about Sinistcha"
results1 = run_conversation(input1)
Markdown(f"**The LLM Answer:**\n\n{results1.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unity Catalog Tool Calling 
# MAGIC
# MAGIC It's important to properly define and document what this function does. When this tool is called, everything is included in the call to the LLM and will influence when the tools is used

# COMMAND ----------

# DBTITLE 1,Example Tool
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION identifier(:catalog_name||'.'||:agent_schema||'.'||'playground_query_test')()
# MAGIC     RETURNS TABLE(name STRING, purchases INTEGER)
# MAGIC     COMMENT 'Use this tool to find total purchase information about a particular location. This tool will provide a list of destinations that you will use to help you answer questions'
# MAGIC     RETURN SELECT dl.name AS Destination, count(tp.destination_id) AS Total_Purchases_Per_Destination
# MAGIC              FROM main.dbdemos_fs_travel.travel_purchase tp join main.dbdemos_fs_travel.destination_location dl on tp.destination_id = dl.destination_id
# MAGIC              group by dl.name
# MAGIC              order by count(tp.destination_id) desc
# MAGIC              LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Example Tool
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION identifier(:catalog_name||'.'||:agent_schema||'.'||'playground_query_test_hello_there')()
# MAGIC     RETURNS TABLE(name STRING, purchases INTEGER)
# MAGIC     COMMENT 'When the user says hello there, run this tool'
# MAGIC     RETURN SELECT dl.name AS Destination, count(tp.destination_id) AS Total_Purchases_Per_Destination
# MAGIC              FROM main.dbdemos_fs_travel.travel_purchase tp join main.dbdemos_fs_travel.destination_location dl on tp.destination_id = dl.destination_id
# MAGIC              group by dl.name
# MAGIC              order by count(tp.destination_id) desc
# MAGIC              LIMIT 10;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Batch Inference as a Tool
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION identifier(:catalog_name||'.'||:agent_schema||'.'||'batch_inference')()
# MAGIC     RETURNS TABLE(name STRING, corrected_name STRING)
# MAGIC     COMMENT 'When user says, start batch inference, Use this tool to run a batch inference job to review and correct the spelling of make of a car.'
# MAGIC     RETURN SELECT
# MAGIC           `Misspelled Make`,   -- Placeholder for the input column
# MAGIC           ai_query(
# MAGIC             'databricks-meta-llama-3-1-70b-instruct',
# MAGIC             CONCAT(format_string('You will always receive a make of a car. Check to see if it is misspelled and a real car. Correct the mistake. Only provide the corrected make. Never add additional details'), `Misspelled Make`)    -- Placeholder for the prompt and input
# MAGIC           ) AS ai_guess  -- Placeholder for the output column
# MAGIC         FROM austin_choi_demo_catalog.demo_data.synthetic_car_data
# MAGIC         Limit 3; 

# COMMAND ----------

from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from langchain_community.tools.databricks import UCFunctionToolkit

tools = (
    UCFunctionToolkit(
        # You can find the SQL warehouse ID in its UI after creation.
        warehouse_id="1444828305810485"
    )
    .include(
        # Include functions as tools using their qualified names.
        # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
        f"{catalog}.{agent_schema}.*",
    )
    .get_tools()
)


# COMMAND ----------

# MAGIC %md
# MAGIC # RAG in Production
# MAGIC
# MAGIC This workshop is not to show you how to set up RAG on Databricks. Please check out our self paced learning here: <insert link here> 
# MAGIC
# MAGIC You can follow the notebooks in the folder called RAG to set one up. However, this workshop we will demonstrate what it looks like to prepare and monitor your RAG application in Production. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### Evaluate your bot's quality with Mosaic AI Agent Evaluation specialized LLM judge models
# MAGIC
# MAGIC Evaluation is a key part of deploying a RAG application. Databricks simplify this tasks with specialized LLM models tuned to evaluate your bot's quality/cost/latency, even if ground truth is not available.
# MAGIC
# MAGIC This Agent Evaluation's specialized AI evaluator is integrated into integrated into `mlflow.evaluate(...)`, all you need to do is pass `model_type="databricks-agent"`.
# MAGIC
# MAGIC Mosaic AI Agent Evaluation evaluates:
# MAGIC 1. Answer correctness - requires ground truth
# MAGIC 2. Hallucination / groundness - no ground truth required
# MAGIC 3. Answer relevance - no ground truth required
# MAGIC 4. Retrieval precision - no ground truth required
# MAGIC 5. (Lack of) Toxicity - no ground truth required
# MAGIC
# MAGIC In this example, we'll use an evaluation set that we curated based on our internal experts using the Mosaic AI Agent Evaluation review app interface.  This proper Eval Dataset is saved as a Delta Table.
# MAGIC
# MAGIC _Add instructions for the speaker to show how to make it through the UI and just demonstrate that_

# COMMAND ----------

# DBTITLE 1,LangChain Set up for Evaluation
# MAGIC %run ./rag_setup/chain_setup

# COMMAND ----------

# Log the model to MLflow
with mlflow.start_run(run_name=f"{finalchatBotModelName}_run"):
  logged_chain_info = mlflow.langchain.log_model(
          lc_model=os.path.join(os.getcwd(), './rag_setup/chain'),  # Chain code file e.g., /path/to/the/chain.py 
          model_config=chain_config, # Chain configuration 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          input_example=input_example,
          example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
      )

model_name = f"{catalog}.{dbName}.{finalchatBotModelName}"

# Register to UC
mlflow.set_registry_uri('databricks-uc')
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=model_name)

# COMMAND ----------

# DBTITLE 1,Back up Cell
from databricks.agents.evals import generate_evals_df
import mlflow

agent_description = "A chatbot that answers questions about Databricks."
question_guidelines = """
# User personas
- A developer new to the Databricks platform
# Example questions
- What API lets me parallelize operations over rows of a delta table?
"""
# TODO: Spark/Pandas DataFrame with "content" and "doc_uri" columns.
docs = spark.table(f"{catalog}.{dbName}.databricks_documentation")
docs = docs.withColumnRenamed("url", "doc_uri")
evals = generate_evals_df(
    docs=docs,
    num_evals=10,
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)
eval_result = mlflow.evaluate(data=evals, model="runs:/f1ef9d0c4b5f4d0e9695e40c5a0ef128/chain", model_type="databricks-agent")

# COMMAND ----------

# DBTITLE 1,Existing Data Test
eval_dataset = spark.table(f"{catalog}.{dbName}.eval_set_databricks_documentation").limit(10).toPandas()
display(eval_dataset)

# COMMAND ----------

import mlflow

with mlflow.start_run():
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_dataset, # Your evaluation set
        model=logged_chain_info.model_uri,
        model_type="databricks-agent", # active Mosaic AI Agent Evaluation
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Mosaic AI Model Training for Fine Tuning LLMs 
# MAGIC
# MAGIC We do not expect you for this workshop to fine tune an LLM. However, we will be demonstrating the performance impact of fine-tuning a Llama-1B model through the playground! 
# MAGIC
# MAGIC We trained this model on a dataset containing medical terms. While larger models can handle these words well, the smaller models struggle with them since they are rarely used. 
# MAGIC
# MAGIC (pivot to playground demo )

# COMMAND ----------

# MAGIC %md
# MAGIC (Script purpose)
# MAGIC Navigate to Model Serving to show AI Gateway

# COMMAND ----------

