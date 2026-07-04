import json

# Sample dat
# For each prompt, please spcify the returned type you want.
prompts = [
    # 8 easy ones
    "How many nodes are in the graph? Return only the number.",
    "How many nodes and edges are in the graph? Return a list.",
    "Add a label app:prod to nodes with address prefix 15.76 and add the label app:test to nodes with address prefix 149.196. Return the networkx graph object.",
    "Show me the unique labels and the number of nodes per label. Return a table with header 'Label', 'Number of Nodes' on the first row.",
    "Remove the label 'type:VM' from all the nodes. Return the networkx graph object.",
    "What are max degree and min degree in the graph? Return a table with with header 'Max degree', 'Min degree' on the first row.",
    "Color all of the nodes with label 'app:prod' purple. Return the networkx graph object.",
    "Color the node with max degree red and min degree green. Return the networkx graph object.",

    # 8 medium ones
    "How many nodes are there that have an edge to nodes with labels app:prod or app:test and doesn't have either of those labels? Return only the number.",
    "Assign a unique color for each /16 IP address prefix and color the nodes accordingly. Return the networkx graph object.",
    "Color the size of the node with max degree green and double it's size. Return the networkx graph object.",
    "Find nodes with top 10 number of degrees, list nodes, labels and number of degrees. Return a table without headers.",
    "Color the nodes that can be connect to nodes with labels app:prod with green. Return the networkx graph object.",
    # TODO: check the ground truth again. Does this one should be create a partition that minimizes the number of edges between the parts using spectral graph partitioning.
    "Cut the graph into two parts such that the number of edges between the cuts is same. Color two parts with red and blue. Return the networkx graph object.",
    "Identify the unique labels in the graph. Create a new graph with a node for each unique label. For each edge in the old graph, identify the labels of the nodes on either side of the edge. Connect the nodes with those labels in the new graph if they are not already connected by an edge. Return the networkx graph object.",
    # TODO: Print out each cluster so I can check?
    "Calculate the total byte weight of edges incident on each node, use kmeans clustering to cluster the total byte weights into 5 clusters, apply the cluster labels as strings to the nodes and pick and assign colors to the nodes based on their cluster labels. Shape the data correctly using numpy before passing it to kmeans. Return the networkx graph object.",

    # more for experiment
    "How many nodes that contains labels 'app:prod' are in the graph? Return only the number.",
    "How many nodes that contains labels 'app:test' are in the graph? Return only the number.",
    "Find all nodes that have edges with node '149.196.217.92' without the node itself. Return the list.",
    "Find all nodes that have edges with node '149.196.174.213' without the node itself. Return the list.",
]

answers=[
    # 8 easy ones
    "\ndef ground_truth_process_graph(graph_data):\n    return_object = {\n        'type': 'text',\n        'data': str(len(graph_data.nodes))\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    num_nodes = len(graph_data.nodes())\n    num_edges = len(graph_data.edges())\n    return_object = {\n        'type': 'list',\n        'data': [num_nodes, num_edges]\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    return_object = {'type': 'graph', 'data': {}}\n    for node in graph_data.nodes():\n        ip_address = graph_data.nodes[node]['ip_address']\n        if ip_address.startswith('15.76'):\n            graph_data.nodes[node]['labels'].append('app:prod')\n        elif ip_address.startswith('149.196'):\n            graph_data.nodes[node]['labels'].append('app:test')\n    return_object['data'] = graph_data\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    labels = {}\n    for node in graph_data.nodes():\n        for label in graph_data.nodes[node]['labels']:\n            if label not in labels:\n                labels[label] = 1\n            else:\n                labels[label] += 1\n    return_object = {\n        'type': 'table',\n        'data': [['Label', 'Number of Nodes']]\n    }\n    for label, count in labels.items():\n        return_object['data'].append([label, count])\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    for node in graph_data.nodes():\n        labels = graph_data.nodes[node]['labels']\n        if 'type:VM' in labels:\n            labels.remove('type:VM')\n            graph_data.nodes[node]['labels'] = labels\n    return_object = {\n        'type': 'graph',\n        'data': graph_data\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    max_degree = 0\n    min_degree = float('inf')\n    for node in graph_data.nodes():\n        degree = graph_data.degree(node)\n        if degree > max_degree:\n            max_degree = degree\n        if degree < min_degree:\n            min_degree = degree\n    return_object = {\n        'type': 'table',\n        'data': [['Max degree', 'Min degree'], [max_degree, min_degree]]\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    for node in graph_data.nodes():\n        if 'app:prod' in graph_data.nodes[node]['labels']:\n            graph_data.nodes[node]['color'] = 'purple'\n    return_object = {\n        'type': 'graph',\n        'data': graph_data\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    max_degree = 0\n    min_degree = float('inf')\n    max_node = None\n    min_node = None\n    for node in graph_data.nodes():\n        degree = graph_data.degree(node)\n        if degree > max_degree:\n            max_degree = degree\n            max_node = node\n        if degree < min_degree:\n            min_degree = degree\n            min_node = node\n    graph_data.nodes[max_node]['color'] = 'red'\n    graph_data.nodes[min_node]['color'] = 'green'\n    return_object = {\n        'type': 'graph',\n        'data': graph_data\n    }\n    return return_object",

    # 8 medium ones
    "\ndef ground_truth_process_graph(graph_data):\n    count = 0\n    for node in graph_data.nodes():\n        if 'app:prod' not in graph_data.nodes[node]['labels'] and 'app:test' not in graph_data.nodes[node]['labels']:\n            for neighbor in graph_data.neighbors(node):\n                if 'app:prod' in graph_data.nodes[neighbor]['labels'] or 'app:test' in graph_data.nodes[neighbor]['labels']:\n                    count += 1\n                    break\n    return_object = {\n        'type': 'text',\n        'data': str(count)\n    }\n    return json.dumps(return_object)\n",
    "\ndef ground_truth_process_graph(graph_data):\n    # Create a dictionary to store the /16 IP prefixes and their corresponding colors\n    ip_prefix_colors = {}\n    # Iterate through the nodes in the graph\n    for node in graph_data.nodes():\n        # Get the IP address of the node\n        ip_address = graph_data.nodes[node]['ip_address']\n        # Get the /16 IP prefix of the node\n        ip_prefix = ip_address.split('.')[0] + '.' + ip_address.split('.')[1]\n        # Check if the /16 IP prefix is already in the dictionary\n        if ip_prefix not in ip_prefix_colors:\n            # Generate a random color for the /16 IP prefix\n            color = '#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])\n            # Add the /16 IP prefix and its corresponding color to the dictionary\n            ip_prefix_colors[ip_prefix] = color\n        # Assign the color to the node\n        graph_data.nodes[node]['color'] = ip_prefix_colors[ip_prefix]\n    # Return the graph object\n    return_object = {\n        'type': 'graph',\n        'data': graph_data\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    max_degree = 0\n    max_node = None\n    for node in graph_data.nodes():\n        if graph_data.degree(node) > max_degree:\n            max_degree = graph_data.degree(node)\n            max_node = node\n    graph_data.nodes[max_node]['color'] = 'green'\n    graph_data.nodes[max_node]['size'] *= 2\n    return_object = {\n        'type': 'graph',\n        'data': graph_data\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    degree_list = []\n    for node in graph_data.nodes():\n        degree_list.append((node, graph_data.degree(node)))\n    degree_list.sort(key=lambda x: x[1], reverse=True)\n    top_10_degree_list = degree_list[:10]\n    return_object = {\n        'type': 'table',\n        'data': [[node, graph_data.nodes[node]['labels'], degree] for node, degree in top_10_degree_list]\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    for node in graph_data.nodes():\n        if any(label.startswith('app:prod') for label in graph_data.nodes[node]['labels']):\n            for neighbor in graph_data.neighbors(node):\n                graph_data.nodes[neighbor]['color'] = 'green'\n    return_object = {\n        'type': 'graph',\n        'data': graph_data\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    # Get the IP prefixes of all nodes\n    ip_prefixes = set()\n    for node in graph_data.nodes():\n        ip_prefixes.add(graph_data.nodes[node]['ip_address'].split('.')[0])\n    \n    # Create two sets of nodes, one for each color\n    red_nodes = set()\n    blue_nodes = set()\n    for ip_prefix in ip_prefixes:\n        for node in graph_data.nodes():\n            if graph_data.nodes[node]['ip_address'].startswith(ip_prefix):\n                if len(red_nodes) <= len(blue_nodes):\n                    red_nodes.add(node)\n                else:\n                    blue_nodes.add(node)\n    \n    # Color the nodes\n    for node in red_nodes:\n        graph_data.nodes[node]['color'] = 'red'\n    for node in blue_nodes:\n        graph_data.nodes[node]['color'] = 'blue'\n    \n    # Return the graph\n    return_object = {\n        'type': 'graph',\n        'data': graph_data\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    # Create a new graph\n    new_graph = nx.Graph()\n    \n    # Get the unique labels from the graph\n    labels = set()\n    for node in graph_data.nodes():\n        labels.update(graph_data.nodes[node]['labels'])\n    \n    # Create a node for each unique label\n    for label in labels:\n        new_graph.add_node(label)\n    \n    # Connect the nodes with those labels in the new graph if they are not already connected by an edge\n    for edge in graph_data.edges():\n        source_labels = graph_data.nodes[edge[0]]['labels']\n        target_labels = graph_data.nodes[edge[1]]['labels']\n        for source_label in source_labels:\n            for target_label in target_labels:\n                if not new_graph.has_edge(source_label, target_label):\n                    new_graph.add_edge(source_label, target_label)\n    \n    # Return the networkx graph object\n    return_object = {\n        'type': 'graph',\n        'data': new_graph\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    # Calculate the total byte weight of edges incident on each node\n    node_byte_weights = {}\n    for node in graph_data.nodes():\n        node_byte_weights[node] = 0\n        for edge in graph_data.edges(node):\n            node_byte_weights[node] += graph_data.edges[edge]['byte_weight']\n    \n    # Use kmeans clustering to cluster the total byte weights into 5 clusters\n    byte_weights = np.array(list(node_byte_weights.values()))\n    byte_weights = byte_weights.reshape(-1, 1)\n    kmeans = KMeans(n_clusters=5, random_state=0).fit(byte_weights)\n    labels = kmeans.labels_\n    \n    # Apply the cluster labels as strings to the nodes\n    for i, node in enumerate(graph_data.nodes()):\n        graph_data.nodes[node]['labels'].append('cluster_label: ' + str(labels[i]))\n    \n    # Pick and assign colors to the nodes based on their cluster labels\n    colors = ['red', 'green', 'blue', 'yellow', 'orange']\n    for i, node in enumerate(graph_data.nodes()):\n        graph_data.nodes[node]['color'] = colors[labels[i]]\n    \n    # Return the networkx graph object\n    return_object = {\n        'type': 'graph',\n        'data': graph_data\n    }\n    return return_object",


    # more for experiment
    "\ndef ground_truth_process_graph(graph_data):\n    count = 0\n    for node in graph_data.nodes():\n        labels = graph_data.nodes[node]['labels']\n        if 'app:prod' in labels:\n            count += 1\n    \n    return {\n        'type': 'text',\n        'data': str(count)\n    }",
    "\ndef ground_truth_process_graph(graph_data):\n    count = 0\n    for node in graph_data.nodes():\n        labels = graph_data.nodes[node]['labels']\n        if 'app:test' in labels:\n            count += 1\n    \n    return {\n        'type': 'text',\n        'data': str(count)\n    }",
    "\ndef ground_truth_process_graph(graph_data):\n    target_ip = '149.196.217.92'\n    connected_nodes = [edge[0] for edge in graph_data.edges() if edge[1] == target_ip] + [edge[1] for edge in graph_data.edges() if edge[0] == target_ip]\n    return_object = {\n        'type': 'list',\n        'data': connected_nodes\n    }\n    return return_object",
    "\ndef ground_truth_process_graph(graph_data):\n    target_ip = '149.196.174.213'\n    connected_nodes = [edge[0] for edge in graph_data.edges() if edge[1] == target_ip] + [edge[1] for edge in graph_data.edges() if edge[0] == target_ip]\n    return_object = {\n        'type': 'list',\n        'data': connected_nodes\n    }\n    return return_object",

]


# File to save the jsonl
output_file = 'traffic-analysis-finetune.jsonl'

# Prepare data in the required format
data = []
for prompt, answer in zip(prompts, answers):
    entry = {
        "messages": [
            {"role": "system", "content": "You are a system to help network management"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
            {"role": "system", "content": "You are a system to help network management"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
    }
    data.append(entry)

# Write data to jsonl file
with open(output_file, 'w') as f:
    for entry in data:
        f.write(json.dumps(entry) + '\n')

print(f"Data successfully written to {output_file}")
