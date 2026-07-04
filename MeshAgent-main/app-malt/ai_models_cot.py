import os
import openai
from dotenv import load_dotenv
import json
import pandas as pd
import inspect
import re
from langchain.llms import VertexAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
from langchain.callbacks import get_openai_callback
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools
# For GPT3.5 or GPT4
from langchain.chat_models import AzureChatOpenAI
# For other models: text-davinci-003
from langchain.llms import AzureOpenAI

# Load environ variables from .env, will not override existing environ variables
load_dotenv()

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# For GPT-4 in Azure
llm = AzureChatOpenAI(
    openai_api_type='azure',
    openai_api_base=OPENAI_API_BASE,
    openai_api_version="2023-05-15",
    deployment_name='gpt-4-32k',
    model_name='gpt-4-32k',
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0,
    max_tokens=4000,
    )

# # For GPT-3.5 in Azure
# llm = AzureChatOpenAI(
#     openai_api_type='azure',
#     openai_api_base=OPENAI_API_BASE,
#     openai_api_version="2023-05-15",
#     deployment_name='gpt-35-turbo-16k',
#     model_name='gpt-35-turbo-16k',
#     openai_api_key=OPENAI_API_KEY,
#     temperature=0.3,
#     max_tokens=4000,
#     )

# genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
# llm = VertexAI(model_name="gemini-pro",
#                max_output_tokens=1000,
#                temperature=0.5)

# For baseline and query-specific constraint only:
constraint_prefix = """
Generate the Python code needed to process the network graph to answer the user query. 
The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph) and returns a single object return_object. 
The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query. 
If the output type is 'text' then the 'data' key should be convert to a string. 
If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table. 
If the output type is 'graph' then the 'data' key should be a networkx graph.

All of your output should only contain the defined function, and display in a Python code block.
"""

constraint_suffix = """Begin! Strictly generate Python code with the following format:

Answer:
```python
${{Code that will answer the user question or request}}
```
Question: {input}
Constraints: {constraints}
"""

constraint_prompt = PromptTemplate(
    input_variables=["input", "constraints"],
    template=constraint_prefix + constraint_suffix
)

constraint_only_chain = LLMChain(llm=llm, prompt=constraint_prompt)


# For summary of steps
summary_prefix = """
You should behave with chain of thoughts, the first answer is three summarized steps you need to take to answer the user query.

Each node has a ‘type’ attribute and other attributes depending on its type. The ‘type’ attribute is a list, and each element is in the format of ‘EK_{{TYPE}}’. For example, EK_PACKET_SWITCH indicates this node is a packet switch node. Because it is a list, each node can have multiple types include EK_SUPERBLOCK, EK_CHASSIS, EK_RACK, EK_AGG_BLOCK, EK_JUPITER, EK_PORT, EK_SPINEBLOCK, EK_PACKET_SWITCH, EK_CONTROL_POINT, EK_CONTROL_DOMAIN.
Each directed edge also has a ‘type’ attribute, where the value RK_CONTAINS indicates the source node contains the destination node, and the value RK_CONTROLS indicates the source node controls the destination node. 
"""

summary_suffix = """Begin! Strictly generate steps with the following string format:

'
Step 1: the first step in your chain of thoughts.
Step 2: the second step in your chain of thoughts.
Step 3: the third step in your chain of thoughts.
'

Question: {input}
"""

summary_gen_prompt = PromptTemplate(
    input_variables=["input"],
    template=summary_prefix + summary_suffix
)

summary_gen_chain = summary_gen_prompt | llm

# CoT only
cot_prefix = """
For the given breakdown step, generate the Python code needed to process the network graph to answer the user question or request. 
If there is code available from the last step, you should expand the new code based on it. If there is no code available, just generate from scratch.

The network graph data is stored as a networkx graph object, the Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data and returns a single object return_object. The input argument graph_data will be a networkx graph object with nodes and edges.

The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table' or 'graph'.
The 'data' key should contain the data needed to render the output. If the output type is 'text' then the 'data' key should contain a string. If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.If the output type is 'graph' then the 'data' key should contain a networkx graph.
"""

cot_suffix = """Begin! Do NOT include any text after the code block. Strictly generate Python code with the following format:

Answer:
```python
${{Code that will answer the user question or request}}
```
Question: {input}
Constraints: {constraints}
Step: {step}
Code_from_last_step: {code}
"""

cot_prompt = PromptTemplate(
    input_variables=["input", "constraints", "step", "code"],
    template=cot_prefix+cot_suffix
)

cot_only_chain = cot_prompt | llm


# CoT plus tools
cot_tool_prefix = """
For the given breakdown step, generate the Python code needed to process the network graph to answer the user question or request. 
If there is code available from the last step, you should expand the new code based on it. If there is no code available, just generate from scratch. 
If a new step is not needed, just use the same code from last step.
Before generating, check if the extracted tool is useful for the current query, if it is, then you should try to leverage it.

Strictly follow the data input and out format:
The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph object) and returns a single object return_object. 

The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table' or 'graph'.
The 'data' key should contain the data needed to render the output. If the output type is 'text' then the 'data' key should contain a string. If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.If the output type is 'graph' then the 'data' key should contain a networkx graph.
"""

cot_tool_suffix = """Begin! Your code should only contain the process_graph(). Strictly generate Python code with the following format, without comments:

Answer:
```python
${{Code that will answer the user question or request}}
```

Question: {input}
Constraints: {constraints}
Step: {step}
Code_from_last_step: {code}
Extracted tool: {tool}
"""

cot_plus_tool_prompt = PromptTemplate(
    input_variables=["input", "constraints", "step", "code", "tool"],
    template=cot_tool_prefix+cot_tool_suffix
)

cot_plus_tool_chain = cot_plus_tool_prompt | llm


# For self-debug
debug_prefix = """
Generate the Python code needed to process the network graph to answer the user query. 
The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph) and returns a single object return_object. 
The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query. 
If the output type is 'text' then the 'data' key should be convert to a string. 
If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table. 
If the output type is 'graph' then the 'data' key should be a networkx graph.

All of your output should only contain the defined function, and display in a Python code block.
"""

debug_suffix = """Please debug the following code you generated before:
Question: {input}
Constraints: {constraints}
Code: {code}
Error: {error}
"""

self_debug_prompt = PromptTemplate(
    input_variables=["input", "constraints", "code", "error"],
    template=debug_prefix + debug_suffix
)

pySelfDebugger = LLMChain(llm=llm, prompt=self_debug_prompt)
