import json
import traceback
from dotenv import load_dotenv
import openai
import pandas as pd
from prototxt_parser.prototxt import parse
from collections import Counter
import os
from ai_models_cot import constraint_only_chain
import networkx as nx
import jsonlines
import random
from networkx.readwrite import json_graph
from langchain.callbacks import get_openai_callback
import json
import re
import time
import sys
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
# from azure.search.documents.models import Vector

# Load environ variables from .env, will not override existing environ variables
load_dotenv()
GRAPH_PATH = "data/test_graph.json"

# For traffic analysis graph
def getGraphData():
    # Read the grpah json file and return the contents as json
    with open(GRAPH_PATH, "r") as f:
        rawData = json.load(f)

    G = json_graph.node_link_graph(rawData)

    return G

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return cb.total_tokens

def node_attributes_are_equal(node1_attrs, node2_attrs):
    # Check if both nodes have the exact same set of attributes
    if set(node1_attrs.keys()) != set(node2_attrs.keys()):
        return False

    # Check if all attribute values are equal
    for attr_name, attr_value in node1_attrs.items():
        if attr_value != node2_attrs[attr_name]:
            return False

    return True


def extract_constraints(results):
    '''
    Iterates over results iterator and checks if each item contains 'constraint'.
    If it does, it appends the constraint to the constraints_list.
    Finally, it joins all constraints into a single string.
    '''
    constraints_list = []
    for result in results:
        if 'constraint' in result:
            constraints_list.append(result['constraint'])

            # join all constraints into a single string
    constraints_string = ' '.join(constraints_list)
    return constraints_string

def clean_up_llm_output_func(answer):
    '''
    Extract only the def process_graph() funtion from the output of LLM
    :param answer: output of LLM
    :return: cleaned function
    '''
    start = answer.find("def process_graph")
    end = -1
    index = 0
    for _ in range(2):  # change the number 2 to any 'n' to find the nth occurrence
        end = answer.find("```", index)
        index = end + 1
    clean_code = answer[start:end].strip()
    return clean_code

def check_list_equal(lst1, lst2):
    if lst1 and isinstance(lst1[0], list):
        return Counter(json.dumps(i) for i in lst1) == Counter(json.dumps(i) for i in lst2)
    else:
        return Counter(lst1) == Counter(lst2)


def clean_up_output_graph_data(ret):
    if isinstance(ret['data'], nx.Graph):
        # Create a nx.graph copy, so I can compare two nx.graph later directly
        ret_graph_copy = ret['data']
        jsonGraph = nx.node_link_data(ret['data'])
        ret['data'] = jsonGraph

    else:  # Convert the jsonGraph back to nx.graph, to check if they are identical later
        ret_graph_copy = json_graph.node_link_graph(ret['data'])

    return ret_graph_copy


def extract_tools(results):
    '''
    Iterates over results iterator and checks if each item contains 'tool'.
    If it does, it appends the tool to the tool_list.
    Finally, it joins all constraints into a single string.
    '''
    tool_list = []
    for result in results:
        if result['@search.score'] < 0.80:
            return "no tools available"
        else:
            if 'tool' in result:
                tool_list.append(result['tool'])

    # join all constraints into a single string
    tool_string = ' '.join(tool_list)
    return tool_string