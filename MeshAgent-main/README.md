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
! pip install Faker
```

## File update tracing
`ai_models_constraints.py`: Use together with `code_gen_rag.py`

`ai_models_constraints_selfdebug.py`: Use together with `self_debug_rag.py`

`code_gen_rag.py`: Code-gen with RAG searched constraints.

`self_debug_rag.py`: Use self-debug for simple 'code cannot run' errors.

`code_gen_baseline_all_constraints.py`: Code-gen with all available constraints.

`verifier.py`: invariants verification for MALT.
