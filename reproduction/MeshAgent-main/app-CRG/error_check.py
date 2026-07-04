import sys
import json
import traceback
import re
import numpy as np
import pandas as pd
import networkx as nx
import ipaddress

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
            graph_checks = [self.verify_node_type]
            for check in graph_checks:
                try:
                    check()
                except Exception as e:
                    print("Check failed:", e)
                    print(traceback.format_exc())
                    return False, e
            return True, ""

        if self.output_list:
            return True, ""

    def verify_node_type(self):
        """
        Verify if each node's 'type' is one of the four allowed types.

        Args:
            graph (networkx.Graph): The graph to verify.

        Returns:
            bool: True if all types are valid, False otherwise.
        """
        allowed_types = set(['virtualmachines', 'Networkinterfaces', 'virtualnetworks', 'networksecuritygroups'])

        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type')
            if node_type:
                if node_type not in allowed_types:
                    print(f"Invalid type at node: {node} with type: {node_type}")
                    return False
        return True



