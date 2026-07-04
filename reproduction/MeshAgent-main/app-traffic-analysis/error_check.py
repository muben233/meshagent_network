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
            graph_checks = [self.verify_ip_addresses]
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

    def verify_ip_addresses(self):
        """
        Verify if each node's 'ip_address' is a valid IP address.

        Args:
            graph (networkx.Graph): The graph to verify.

        Returns:
            bool: True if all IP addresses are valid, False otherwise.
        """
        for node in self.graph.nodes():
            ip_address = self.graph.nodes[node].get('ip_address')
            if ip_address:
                try:
                    ipaddress.ip_address(ip_address)
                except ValueError:
                    print(f"Invalid IP address at node: {node} with IP: {ip_address}")
                    return False
        return True

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

