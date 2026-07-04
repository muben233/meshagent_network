# README
## Prerequisites
To run the code, install the following packages. Please note that the pip install azure-search-documents==11.4.0a20230509004 is currently using the Dev Feed. For instructions on how to connect to the dev feed, please visit [Azure-Python-SDK Azure Search Documents Dev Feed](https://dev.azure.com/azure-sdk/public/_artifacts/feed/azure-sdk-for-python/connect/pip).

To set up Google VertexAI API, follow the below steps:
1. [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install)
2. [Create your credential file](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev)
3. [Create a project and enable VertexAI](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction/create-notebook) (follow error messages for detailed setup)

```
! pip install azure-search-documents==11.4.0a20230509004
! pip install openai
! pip install python-dotenv
! pip install jsonlines==3.1.0
! pip3 install prototxt-parser
! pip install google-generativeai
! pip install langchain google-cloud-aiplatform
! pip install langchain==0.0.350
! pip install langchain-experimental
```

## Setup .env
Fill in the required API or address in `.env`.
```
# For OpenAI LLM model
OPENAI_API_TYPE=''
OPENAI_API_VERSION=''
OPENAI_API_KEY=""
OPENAI_API_BASE=""

# For Azure
AZURE_SEARCH_ADMIN_KEY=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_INDEX_NAME=""
AZURE_OPENAI_API_VERSION=''

# For Google LLM model
GOOGLE_API_KEY=""

# For RAG index
RAG_MALT_CONSTRAINT=""
RAG_MALT_TOOL=""
```


## Build RAG
Add constraints RAG under `data/rag_constraints.json`, add tools RAG under `data/rag_tools.json`.

Go to `create_RAG_index`, run the two jupyter notebook to create index for rag_constraints and rag_tools.

## Build error_checker

## Experiment instruction
It adds on the module one by one.

1. Baseline: All constraints as static prompt.
```
python baseline_static_prompt.py
```
2. Query-specific constraints.
```
python query_specific_constraint_prompt.py
```
3. Query-specific constraints + CoT.
```
python cot_with_query_specific.py
```
4. Query-specific constraints + CoT + error reduce.
```
python cot_with_error_check.py
```
5. Query-specific constraints + CoT + error reduce + tools.
```
python full_cot_with_tools.py
```