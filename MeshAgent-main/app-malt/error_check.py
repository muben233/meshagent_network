import sys
import json
import traceback
import re
import numpy as np
import pandas as pd
import networkx as nx


class MyChecker():
    def __init__(self, ret_graph=None, ret_list=None):
        if ret_graph:
            self.graph = ret_graph
        else:
            self.graph = None
        if ret_list:
            self.output_list = ret_list
        else:
            self.output_list = None

    def evaluate_all(self):
        if self.graph:
            graph_checks = [self.verify_node_format_and_type,
                            self.verify_edge_format_and_type,
                            self.verify_node_hierarchy,
                            self.verify_no_isolated_nodes,]
            for check in graph_checks:
                try:
                    check()
                except Exception as e:
                    print("Check failed:", e)
                    print(traceback.format_exc())
                    return False, e
            return True, ""

        if self.output_list:
            list_checks = [self.verify_bandwidth]
            for check in list_checks:
                try:
                    check()
                except Exception as e:
                    print("Check failed:", e)
                    print(traceback.format_exc())
                    return False, e
            return True, ""

    def verify_node_format_and_type(self):
        """
        Graph check: verify node type and format
        """
        valid_types = ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN']

        for node in self.graph.nodes():
            # Check if the node has a 'type' attribute
            if self.graph.nodes[node].get('type'):
                node_types = self.graph.nodes[node]['type']
                for node_type in node_types:
                    if node_type not in valid_types:
                        raise Exception(f"verify_node_types failed at node: {node} with type: {node_type}")
            else:
                raise Exception(f"verify_node_types failed at node: {node}, there is no node type on it.")

        return True, ""

    def verify_edge_format_and_type(self):
        """
        Graph check: verify_edge_format_and_type
        """
        valid_edge_types = ["RK_CONTAINS", "RK_CONTROLS"]

        for edge in self.graph.edges(data=True):
            # Check if the edge has a 'type' attribute
            if 'type' not in edge[2]:
                return False  # Edge does not have a 'type' attribute
            # Check if the edge's type is in the valid_edge_types list
            if not any(edge_type in edge[2]['type'] for edge_type in valid_edge_types):
                raise Exception(f"verify_edge_format_and_type failed at edge: {edge} with type: {edge[2]['type']}")
        return True, ""

    def verify_node_hierarchy(self):
        """
        Graph check: verify_node_hierarchy
        """
        hierarchy = {
            "EK_JUPITER": ["EK_SPINEBLOCK", "EK_SUPERBLOCK"],
            "EK_SPINEBLOCK": ["EK_PACKET_SWITCH"],
            "EK_SUPERBLOCK": ["EK_AGG_BLOCK"],
            "EK_AGG_BLOCK": ["EK_PACKET_SWITCH"],
            "EK_CHASSIS": ["EK_CONTROL_POINT", "EK_PACKET_SWITCH"],
            "EK_CONTROL_POINT": ["EK_PACKET_SWITCH"],
            "EK_RACK": ["EK_CHASSIS"],
            "EK_PACKET_SWITCH": ["EK_PORT"],
            "EK_CONTROL_DOMAIN": ["EK_CONTROL_POINT"]
        }

        for edge in self.graph.edges(data=True):
            if 'RK_CONTAINS' in edge[2]['type']:
                source_node_types = self.graph.nodes[edge[0]]['type']
                target_node_types = self.graph.nodes[edge[1]]['type']
                for source_type in source_node_types:
                    if source_type in hierarchy and any(target_type in hierarchy[source_type] for target_type in target_node_types):
                        return True, ""

        raise Exception("verify_node_hierarchy failed at edge: " + str(edge))

    def verify_no_isolated_nodes(self):
        """
        Graph check: verify_no_isolated_nodes
        """
        # An isolated node is a node with degree 0, i.e., no edges.
        isolated_nodes = list(nx.isolates(self.graph))

        if len(isolated_nodes) == 0:
            return True, ""  # There are no isolated nodes in the graph.
        else:
            raise Exception("verify_no_isolated_nodes failed at node: " + str(isolated_nodes))


    def verify_no_isolated_nodes(self):
        """
        Graph check: verify_no_isolated_nodes
        """
        # An isolated node is a node with degree 0, i.e., no edges.
        isolated_nodes = list(nx.isolates(self.graph))

        if len(isolated_nodes) == 0:
            return True, ""  # There are no isolated nodes in the graph.
        else:
            raise Exception("verify_no_isolated_nodes failed at node: " + str(isolated_nodes))

    def verify_bandwidth(self):
        """
        Verify if the "Bandwidth" column in a given table is never 0.

        Args:
            data (list): A list of lists representing a table.

        Returns:
            bool: True if the "Bandwidth" column is never 0 or doesn't exist, False otherwise.
        """
        # Check if the input is a table (list of lists)
        if isinstance(self.output_list, list) and all(isinstance(row, list) for row in self.output_list):
            # Check if the first row (header) exists and contains "Bandwidth"
            if self.output_list and "Bandwidth" in self.output_list[0]:
                # Find the index of the "Bandwidth" column
                bandwidth_index = self.output_list[0].index("Bandwidth")

                # Iterate over the table
                for row in self.output_list[1:]:
                    # Check if the "Bandwidth" column is 0
                    if row[bandwidth_index] == 0.0:
                        raise Exception("verify bandwidth failed: Bandwidth should be more than 0.")
        return True, ""

    def verify_port_count(self):
        # TODO: check does this work?
        """
        Verify if the "Port Count" column in a given table is never 0.

        Args:
            data (list): A list of lists representing a table.

        Returns:
            bool: True if the "Port Count" column is never 0 or doesn't exist, False otherwise.
        """
        # Check if the input is a table (list of lists)
        if isinstance(self.output_list, list) and all(isinstance(row, list) for row in self.output_list):
            # Check if the first row (header) exists and contains "Port Count"
            if self.output_list and "Port Count" in self.output_list[0]:
                # Find the index of the "Port Count" column
                port_count_index = self.output_list[0].index("Port Count")

                # Iterate over the table
                for row in self.output_list[1:]:
                    # Check if the "Port Count" column is 0
                    if row[port_count_index] == 0:
                        raise Exception("verify port nodes number failed: Port count should be more than 0.")

        return True, ""

    ##TODO: add each port can only be connected with one packet switch
