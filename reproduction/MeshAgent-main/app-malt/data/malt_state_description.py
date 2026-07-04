import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph
from prototxt_parser.prototxt import parse

def getGraphData():
    input_string = open("malt-example-final.textproto.txt").read()
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

# ouput a json file about all unique object based on type in the graph
def getUniqueObject():
    rawData, G = getGraphData()
    uniqueObject = {}
    for node in rawData['nodes']:
        for type in node['type']:
            if type not in uniqueObject:
                uniqueObject[type] = []
            if node['name'] not in uniqueObject[type]:
                uniqueObject[type].append(node['name'])
    # for each type, "value" is a list of unique object, "attribute" is a list of unique attribute of the object
    for type in uniqueObject:
        uniqueObject[type] = {"value": uniqueObject[type], "attribute": list(G.nodes[uniqueObject[type][0]].keys())}

    with open('uniqueObject.json', 'w') as f:
        json.dump(uniqueObject, f)

getUniqueObject()