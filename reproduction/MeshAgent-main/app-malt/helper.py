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
import json
import re
import time
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def getGraphData():
    input_string = open("data/malt-example-final.textproto.txt").read()
    parsed_dict = parse(input_string)

    # Load MALT data
    G = nx.DiGraph()

    # Insert all the entities as nodes
    for entity in parsed_dict['entity']:
        # Check if the node exists
        if entity['id']['name'] not in G.nodes:
            G.add_node(entity['id']['name'], type=[entity['id']['kind']], name=entity['id']['name'])
        else:
            G.nodes[entity['id']['name']]['type'].append(entity['id']['kind'])
        # Add all the attributes
        for key, value in entity.items():
            if key == 'id':
                continue
            for k, v in value.items():
                G.nodes[entity['id']['name']][k] = v

    # Insert all the relations as edges
    for relation in parsed_dict['relationship']:
        G.add_edge(relation['a']['name'], relation['z']['name'], type=relation['kind'])

    rawData = json_graph.node_link_data(G)

    return rawData, G

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
    Extract the full code (including imports) from LLM output.
    Finds the code block between ```python and ``` markers.
    Falls back to extracting from def process_graph if no markers.
    '''
    # Find code block start (after ```python or ```)
    start = answer.find("```python")
    if start == -1:
        start = answer.find("```")
    if start != -1:
        start = answer.find("\n", start) + 1  # skip the marker line
        end = answer.find("```", start)
        if end != -1:
            return answer[start:end].strip()

    # Fallback: extract from def process_graph
    start = answer.find("def process_graph")
    if start == -1:
        return answer.strip()
    return answer[start:].strip()

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
        if result['@search.score'] < 0.85:
            return "no tools available"
        else:
            if 'tool' in result:
                tool_list.append(result['tool'])

    # join all constraints into a single string
    tool_string = ' '.join(tool_list)
    return tool_string