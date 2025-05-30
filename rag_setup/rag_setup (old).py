# Databricks notebook source
# MAGIC %run ./00-init-advanced $reset_all_data=false

# COMMAND ----------

spark.sql(f"USE {catalog}.{demo_schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{demo_schema}.{volumeName}")

# COMMAND ----------

volume_folder =  f"/Volumes/{catalog}/{demo_schema}/{volumeName}"

# COMMAND ----------

dbutils.fs.mkdirs(f"{volume_folder}/{folderName}")

# COMMAND ----------

# folderVolumePath = f"{volume_folder}/{folderName}"
# print(folderVolumePath)

# COMMAND ----------

# Run this cell if there's no sample PDFs for testing. This will upload a number of Databricks Docs PDF
# upload_pdfs_to_volume(folderVolumePath)

# COMMAND ----------

if not spark.catalog.tableExists("raw_documentation") or spark.table("raw_documentation").isEmpty():
    # Download Databricks documentation to a DataFrame (see _resources/00-init for more details)
    doc_articles = download_databricks_documentation_articles()
    #Save them as a raw_documentation table
    doc_articles.write.mode('overwrite').saveAsTable("raw_documentation")

display(spark.table("raw_documentation").limit(2))

# COMMAND ----------

# DBTITLE 1,Ingesting PDF files as binary format using Auto Loader
# df = (spark.readStream
#         .format('cloudFiles')
#         .option('cloudFiles.format', 'BINARYFILE')
#         .option("pathGlobFilter", "*.pdf")
#         .load('dbfs:'+folderVolumePath))

# # Write the data as a Delta table
# (df.writeStream
#   .trigger(availableNow=True)
#   .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/raw_docs')
#   .table('pdf_raw').awaitTermination())

# COMMAND ----------

# %sql SELECT * FROM pdf_raw

# COMMAND ----------

# DBTITLE 1,To extract our PDF,  we'll need to setup libraries in our nodes
# import warnings
# from pypdf import PdfReader

# def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
#     try:
#         pdf = io.BytesIO(raw_doc_contents_bytes)
#         reader = PdfReader(pdf)
#         parsed_content = [page_content.extract_text() for page_content in reader.pages]
#         return "\n".join(parsed_content)
#     except Exception as e:
#         warnings.warn(f"Exception {e} has been thrown during parsing")
#         return None

# COMMAND ----------

# DBTITLE 1,Trying our text extraction function with a single pdf file
# import io
# import re
# with requests.get('https://github.com/databricks-demos/dbdemos-dataset/blob/main/llm/databricks-pdf-documentation/Databricks-Customer-360-ebook-Final.pdf?raw=true') as pdf:
#   doc = parse_bytes_pypdf(pdf.content)  
#   print(doc)

# COMMAND ----------

import re
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 500

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "header2")])

# Split on H2, but merge small h2 chunks together to avoid having too small chunks. 
def split_html_on_h2(html, min_chunk_size=20, max_chunk_size=500):
    if not html:
        return []
    #removes b64 images captured in the md    
    html = re.sub(r'data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=\n]+', '', html, flags=re.MULTILINE)
    chunks = []
    previous_chunk = ""
    for c in md_splitter.split_text(html):
        content = c.metadata.get('header2', "") + "\n" + c.page_content
        if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size / 2:
            previous_chunk += content + "\n"
        else:
            chunks.extend(text_splitter.split_text(previous_chunk.strip()))
            previous_chunk = content + "\n"
    if previous_chunk:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
    return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]

# Let's try our chunking function
html = spark.table("raw_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import Document, set_global_tokenizer
# from transformers import AutoTokenizer
# from typing import Iterator

# # Reduce the arrow batch size as our PDF can be big in memory
# # spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# # spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", 10)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer to match our model size (will stay below gte 1024 limit)
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=10)
    def extract_and_split(b):
      txt = parse_bytes_pypdf(b)
      if txt is None:
        return []
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# MAGIC  %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS databricks_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
    
(spark.table("raw_documentation")
      .filter('text is not null')
      .repartition(30)
      .withColumn('content', F.explode(parse_and_split('text')))
      .drop("text")
      .write.mode('overwrite').saveAsTable("databricks_documentation"))

display(spark.table("databricks_documentation"))

# COMMAND ----------

# DBTITLE 1,Using Databricks Foundation model GTE as embedding endpoint
from mlflow.deployments import get_deploy_client

# gte-large-en Foundation models are available using the /serving-endpoints/databricks-gtegte-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint=embeddings_endpoint, inputs={"input": ["What is Apache Spark?"]})
pprint(embeddings)

# COMMAND ----------

# DBTITLE 1,Create the final databricks_pdf_documentation table containing chunks and embeddings
# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS pdf_content_embeddings (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS databricks_documentation_embeddings (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint=embeddings_endpoint, inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

# (spark.readStream.table('pdf_raw')
#       .withColumn("content", F.explode(read_as_chunk("content")))
#       .withColumn("embedding", get_embedding("content"))
#       .selectExpr('path as url', 'content', 'embedding')
#   .writeStream
#     .trigger(availableNow=True)
#     .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/pdf_chunk')
#     .table('pdf_content_embeddings').awaitTermination())

# #Let's also add our documentation web page content
# if table_exists(f'{catalog}.{demo_schema}.databricks_documentation'):
#   (spark.readStream.table('databricks_documentation')
#       .withColumn("content", F.explode(read_as_chunk("content"))) 
#       .withColumn('embedding', get_embedding("content"))
#       .select('url', 'content', 'embedding')
#   .writeStream
#     .trigger(availableNow=True)
#     .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/docs_chunks')
#     .table('pdf_content_embeddings').awaitTermination())

# COMMAND ----------

# DBTITLE 1,Creating the Vector Search endpoint
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# DBTITLE 1,Create the Self-managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{demo_schema}.pdf_content_embeddings"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{demo_schema}.{vectorSearchIndexName}"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

# COMMAND ----------


