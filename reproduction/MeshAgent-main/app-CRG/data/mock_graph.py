import random
import string
import json
import networkx as nx
from faker import Faker

# Instantiate Faker
fake = Faker()

"""
The "Resources" graph have multiple attr such as: "name, type, properties".

'type' is a list, the items can vary, include 'virtualmachines', 'Networkinterfaces', 'virtualnetworks', 'networksecuritygroups'.

'name' is a string depending on the 'type',
when 'type'=='virtualmachines' it should be a randomly generated string with length of 12, for example "GCRAZGDL2246".
When 'type'=='Networkinterfaces' it should be one of "ipconfig1", "ipconfig2", until "ipconfig6".
When 'type'=='virtualnetworks' it should be one of ["Subnet-1", "Subnet-2", "jumptainer", "AzureBastionSubnet"].
When 'type'=='networksecuritygroups' it should be one of ["AllowVnetInBound", "AllowVnetOutBound", "DenyAllInBound", "DenyAllOutBound",  "AllowInternetOutBound",  "AllowInternetInBound"]

'properties' can include more attributes depending on the 'type', 
when 'type'=='virtualmachines' an example as below
"properties": {
                 "osType": "Linux" or "Windows" or "MacOS",
                "Networkinterfaces": "ipconfig1" (or other possible value from 'Networkinterfaces')
}
when 'type'=='Networkinterfaces' an example as below
"properties": {
                "virtualnetworks": "Subnet-1" (or other possible value from "virtualnetworks"),
                "Networkinterfaces": "ipconfig1",
                "addressPrefixes": a random value from IPaddress
}
when 'type'=='virtualnetworks' an example as below
"properties": {
                "provisioningState": "Succeeded" or "Failed",
                "addressPrefixes": a random value from IPaddress
                "port": a random value from Port
}
when 'type'=='networksecuritygroups' an example as below
"properties": {
                "protocol": "Any" or "TCP" or "UDP",
                "addressPrefixes": a random value from IPaddress,
                "port": a random value from Port,
                "priority": a number between 2000 to 4000
}

IPaddress = [10.0.0.1, 10.0.0.2, 10.0.0.3, 10.0.0.4, 10.0.0.5]
Port = [21, 22, 23, 24, 25, 26]

'virtualmachines' node can be connected to 'Networkinterfaces' node if they have the same value for 'Networkinterfaces' in properties.
'Networkinterfaces' node can be connected to  'virtualnetworks' node if they have the same value for 'addressPrefixes' in properties.
'virtualnetworks' node can be connected to  'networksecuritygroups' node if they have the same value for 'addressPrefixes' and 'port' in properties.

"""

numNodes = {
    'virtualmachines': 50,
    'Networkinterfaces': 50,
    'virtualnetworks': 50,
    'networksecuritygroups': 50
}

def generate_random_string(prefix, start, end):
    return prefix + str(random.randint(start, end))

def generate_mock_graph(numNodes, outFilename):
    types = ['virtualmachines', 'Networkinterfaces', 'virtualnetworks', 'networksecuritygroups']
    names = {
        'virtualmachines': [generate_random_string("GCRAZGDL", 2100, 2200) for _ in range(numNodes['virtualmachines'])],
        'Networkinterfaces': ['ipconfig' + str(i + 1) for i in range(6)],
        'virtualnetworks': ["Subnet-1", "Subnet-2", "jumptainer", "AzureBastionSubnet"],
        'networksecuritygroups': ["AllowVnetInBound", "AllowVnetOutBound", "DenyAllInBound", "DenyAllOutBound",
                                  "AllowInternetOutBound", "AllowInternetInBound"]
    }
    IPaddress = ['10.0.0.' + str(i + 1) for i in range(5)]
    Port = [21, 22, 23, 24, 25, 26]
    properties = {
        'virtualmachines': lambda: {"osType": random.choice(["Linux", "Windows", "MacOS"]),
                                    "Networkinterfaces": random.choice(names['Networkinterfaces'])},
        'Networkinterfaces': lambda: {"virtualnetworks": random.choice(names['virtualnetworks']),
                                      "Networkinterfaces": random.choice(names['Networkinterfaces']),
                                      "addressPrefixes": random.choice(IPaddress)},
        'virtualnetworks': lambda: {"provisioningState": random.choice(["Succeeded", "Failed"]),
                                    "addressPrefixes": random.choice(IPaddress),
                                    "port": random.choice(Port)},
        'networksecuritygroups': lambda: {"protocol": random.choice(["Any", "TCP", "UDP"]),
                                          "addressPrefixes": random.choice(IPaddress),
                                          "port": random.choice(Port),
                                          "priority": random.randint(2000, 4000)}
    }

    nodes = []
    nodeId = 0
    for type_ in types:
        for i in range(numNodes[type_]):
            name_ = random.choice(names[type_])
            properties_ = properties[type_]()
            nodes.append({'id': nodeId, 'type': type_, 'name': name_, 'properties': properties_})
            nodeId += 1

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node['id'], type=node['type'], name=node['name'], properties=node['properties'])

    # add edges based on node properties
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[i]['type'] == 'virtualmachines' and nodes[j]['type'] == 'Networkinterfaces':
                if nodes[i]['properties']['Networkinterfaces'] == nodes[j]['properties']['Networkinterfaces']:
                    G.add_edge(i, j)
            elif nodes[i]['type'] == 'Networkinterfaces' and nodes[j]['type'] == 'virtualnetworks':
                if nodes[i]['properties']['addressPrefixes'] == nodes[j]['properties']['addressPrefixes']:
                    G.add_edge(i, j)
            elif nodes[i]['type'] == 'virtualnetworks' and nodes[j]['type'] == 'networksecuritygroups':
                if nodes[i]['properties']['addressPrefixes'] == nodes[j]['properties']['addressPrefixes'] and \
                        nodes[i]['properties']['port'] == nodes[j]['properties']['port']:
                    G.add_edge(i, j)

    data = nx.node_link_data(G)
    with open(outFilename, 'w') as outfile:
        json.dump(data, outfile)

generate_mock_graph(numNodes, "resources.json")
