import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")


class _Chain:
    """Replacement for LangChain LLMChain — same .run() interface."""
    def __init__(self, system_prompt, user_template):
        self.system = system_prompt
        self.template = user_template

    def run(self, kwargs):
        user = self.template.format(**kwargs)
        resp = client.chat.completions.create(
            model=MODEL, temperature=0, max_tokens=4000,
            messages=[{"role": "system", "content": self.system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content

    def invoke(self, kwargs):
        return self.run(kwargs)


# ---- baseline / query-specific constraint ----
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

constraint_only_chain = _Chain(constraint_prefix, constraint_suffix)


# ---- summary / CoT step decomposition ----
summary_prefix = """
You should behave with chain of thoughts, the first answer is three summarized steps you need to take to answer the user query.

Each node has a 'type' attribute and other attributes depending on its type. The 'type' attribute is a list, and each element is in the format of 'EK_{{TYPE}}'. For example, EK_PACKET_SWITCH indicates this node is a packet switch node. Because it is a list, each node can have multiple types include EK_SUPERBLOCK, EK_CHASSIS, EK_RACK, EK_AGG_BLOCK, EK_JUPITER, EK_PORT, EK_SPINEBLOCK, EK_PACKET_SWITCH, EK_CONTROL_POINT, EK_CONTROL_DOMAIN.
Each directed edge also has a 'type' attribute, where the value RK_CONTAINS indicates the source node contains the destination node, and the value RK_CONTROLS indicates the source node controls the destination node.
"""

summary_suffix = """Begin! Strictly generate steps with the following string format:

'
Step 1: the first step in your chain of thoughts.
Step 2: the second step in your chain of thoughts.
Step 3: the third step in your chain of thoughts.
'

Question: {input}
"""

summary_gen_chain = _Chain(summary_prefix, summary_suffix)


# ---- CoT step code generation ----
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

cot_only_chain = _Chain(cot_prefix, cot_suffix)


# ---- CoT + tools ----
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

cot_plus_tool_chain = _Chain(cot_tool_prefix, cot_tool_suffix)


# ---- self-debug ----
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

pySelfDebugger = _Chain(debug_prefix, debug_suffix)
