import json
import traceback
from dotenv import load_dotenv
import openai
import copy
import pandas as pd
from prototxt_parser.prototxt import parse
from collections import Counter
import os
from ai_models_cot import summary_gen_chain, cot_only_chain, pySelfDebugger
from helper import getGraphData, extract_constraints, clean_up_llm_output_func, check_list_equal, node_attributes_are_equal, clean_up_output_graph_data
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
from azure.search.documents.models import Vector

# Load environ variables from .env, will not override existing environ variables
load_dotenv()
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
constraint_index_name = os.getenv("RAG_MALT_CONSTRAINT")
tool_index_name = os.getenv("RAG_MALT_TOOL")
azure_search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
credential = AzureKeyCredential(azure_search_key)

EACH_PROMPT_RUN_TIME = 1
OUTPUT_JSONL_PATH = 'logs/debug/baseline_static.jsonl'
DEBUG_LOOP_TOTAL = 3

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine="text-embedding-ada-002")
    embeddings = response['data'][0]['embedding']
    return embeddings


def rag_vector_search(query, num_extraction=10):
    '''
    With given query, use pure vector search to find the most related items from RAG.
    It assume index is already created and uploaded.
    '''
    # Pure Vector Search
    search_client = SearchClient(service_endpoint, constraint_index_name, AzureKeyCredential(azure_search_key))

    results = search_client.search(
        search_text="",
        vector=Vector(value=generate_embeddings(query), k=num_extraction, fields="constraintVector"),
        select=["label", "constraint"]
    )

    return extract_constraints(results)


def self_debug_process_loop(requestData, constraints_found, code, error_details, debug_status_msg, loop_time_index):
    """
    Return the error back to LLM with related info and ask LLM to fix the code.
    :param requestData: input query requestData['query']
    :param constraints_found: RAG based constraints
    :param code: LLM generated code from last time
    :param error_details: error_details
    :param loop_time_index: the number of the self-debug time
    :return: debugged_code
    """
    print(debug_status_msg)

    self_debug_answer = pySelfDebugger.run({'input': requestData['query'],
                                            'constraints': constraints_found,
                                            'code': code,
                                            'error': error_details})
    debugged_code = clean_up_llm_output_func(self_debug_answer)
    print("Debugged code for time: ", loop_time_index)
    print(debugged_code)

    return debugged_code


def extract_final_code(first_step_code, second_step_code, third_step_code):
    """
    Each of input is a string. Find the last one that contains a python code block (```python ```) as the final code.
    """
    # Put them in a list
    step_codes = [first_step_code, second_step_code, third_step_code]

    # Initialize the last code block to None
    last_python_code_block = None

    # Iterate over the step codes in reverse order
    for step_code in reversed(step_codes):
        # Check if the step code contains a Python code block
        if re.search('```python.*?```', step_code, re.DOTALL):
            # If it does, assign it to last_python_code_block and break the loop
            last_python_code_block = step_code
            break

    if last_python_code_block is not None:
        print("The last Python code block is:", last_python_code_block)
    else:
        print("No Python code block found.")

    return last_python_code_block


def userQuery(prompt_list):
    # Load the existing prompt and golden answers from Json
    golden_answer_filename = 'golden_answer_generator/prompt_golden_ans.json'
    with open(golden_answer_filename, "r") as fa:
        allAnswer = json.load(fa)


    # for each prompt in the prompt_list, append it as the value of {'query': prompt}
    for each_prompt in prompt_list:
        print("Query: ", each_prompt)
        requestData = {'query': each_prompt}

        constraints_found = rag_vector_search(each_prompt)
        print("Constraints: ", constraints_found)
        prompt_accu = 0

        _, G = getGraphData()

        # Reset ret when it's a new test
        ret = None
        ground_truth_ret = None

        # Run each prompt for 10 times
        for i in range(EACH_PROMPT_RUN_TIME):
            if requestData['query'] not in allAnswer.keys():
                # terminate the code with error message
                raise SystemExit('Un-support ground truth for the current prompt.')

            print("Find the prompt in the list.")

            print("Calling model")
            summary_output = summary_gen_chain.invoke({"input": requestData['query']})
            # process the AIMessage output
            step_summary = summary_output.to_json()['kwargs']['content']

            # Use regular expressions to split the string by 'Step X:' where X is a digit
            steps = re.split('Step \d+: ', step_summary)
            # Remove the first element which is empty due to the split
            steps = steps[1:]

            first_step_llm = cot_only_chain.invoke({"input": requestData['query'],
                                                   "constraints": constraints_found,
                                                   "step": steps[0],
                                                   "code": "None", })
            first_step_code = first_step_llm.to_json()['kwargs']['content']
            print("Step 1: ", steps[0])
            print("Code generated: ", first_step_code)
            time.sleep(2)

            second_step_llm = cot_only_chain.invoke({"input": requestData['query'],
                                                    "constraints": constraints_found,
                                                    "step": steps[1],
                                                    "code": first_step_code, })
            second_step_code = second_step_llm.to_json()['kwargs']['content']
            print("Step 2: ", steps[1])
            print("Code generated: ", second_step_code)
            time.sleep(2)

            third_step_llm = cot_only_chain.invoke({"input": requestData['query'],
                                                   "constraints": constraints_found,
                                                   "step": steps[2],
                                                   "code": second_step_code, })
            third_step_code = third_step_llm.to_json()['kwargs']['content']
            print("Step 3: ", steps[2])
            print("Code generated: ", third_step_code)

            answer = extract_final_code(first_step_code, second_step_code, third_step_code)
            llm_output_token_count = 0
            # if code contains python package import, remove all lines related to it
            code = clean_up_llm_output_func(answer)
            print(code)
            try:
                exec(code)
                ret = eval("process_graph(G)")
            except Exception:
                # Got the detailed error from exec(code)
                exc_type, ex, tb = sys.exc_info()
                imported_tb_info = traceback.extract_tb(tb)[-1]
                line_number = imported_tb_info[1]
                print_format = '{}: Exception in line: {}. Message: {}'
                error_details = print_format.format(exc_type.__name__, line_number, ex)
                print("Fail due to errors:", error_details)

                # Add self-debug here
                for i in range(DEBUG_LOOP_TOTAL):  # 3 times self-debug loop
                    debugged_code = self_debug_process_loop(requestData,
                                                            constraints_found,
                                                            code,
                                                            error_details,
                                                            debug_status_msg="================= Error reduce: start self-debugging =================",
                                                            loop_time_index=i)

                    try:
                        _, G = getGraphData()
                        exec(debugged_code)
                        ret = eval("process_graph(G)")
                        print("================= Error reduce progress + 1: Code can run! =================")
                        break   # if the code successfully executed, break the loop
                    except Exception as e:
                        # Got the detailed error from exec(code)
                        exc_type, ex, tb = sys.exc_info()
                        imported_tb_info = traceback.extract_tb(tb)[-1]
                        line_number = imported_tb_info[1]
                        print_format = '{}: Exception in line: {}. Message: {}'
                        error_details = print_format.format(exc_type.__name__, line_number, ex)
                        print("Fail due to errors:", error_details)

                        if i == DEBUG_LOOP_TOTAL-1:
                            print("Fail the test, the code cannot run.")
                            # if it still fails due to execution error, log it differently
                            with jsonlines.open(OUTPUT_JSONL_PATH, mode='a') as writer:
                                writer.write(requestData)
                                writer.write({"Result": "Fail, code cannot run"})
                                writer.write({"LLM code": debugged_code})
                                writer.write({"Error": str(e)})
                            # break from the current for loop
                            continue
            # if the type of ret is string, turn it into a json object
            if isinstance(ret, str):
                ret = json.loads(ret)

            if ret['type'] == 'graph':
                ret_graph_copy = clean_up_output_graph_data(ret)

            goldenAnswerCode = allAnswer[requestData['query']]

            # ground truth answer should already be checked to ensure it can run successfully
            exec(goldenAnswerCode)
            ground_truth_ret = eval("ground_truth_process_graph(G)")
            # if the type of ground_truth_ret is string, turn it into a json object
            if isinstance(ground_truth_ret, str):
                ground_truth_ret = json.loads(ground_truth_ret)

            ground_truth_ret['reply'] = goldenAnswerCode
            ret['reply'] = code

            # check type "text", "list", "table", "graph" separately.
            if ground_truth_ret['type'] == 'text':
                # if ret['data'] type is int, turn it into string
                if isinstance(ret['data'], int):
                    ret['data'] = str(ret['data'])
                if isinstance(ground_truth_ret['data'], int):
                    ground_truth_ret['data'] = str(ground_truth_ret['data'])

                if ground_truth_ret['data'] == ret['data']:
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret,
                                                          llm_output_token_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count)

            elif ground_truth_ret['type'] == 'list':
                # Use Counter to check if two lists contain the same items, including duplicate items.
                if check_list_equal(ground_truth_ret['data'], ret['data']):
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret,
                                                          llm_output_token_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count)

            elif ground_truth_ret['type'] == 'table':
                if ground_truth_ret['data'] == ret['data']:
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret,
                                                          llm_output_token_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count)

            elif ground_truth_ret['type'] == 'graph':
                # Undirected graphs will be converted to a directed graph
                # with two directed edges for each undirected edge.
                ground_truth_graph = nx.Graph(ground_truth_ret['data'])
                # TODO: fix ret_graph_copy reference possible error, when it's not created.
                ret_graph = nx.Graph(ret_graph_copy)

                # Check if two graphs are identical, no weights considered
                if nx.is_isomorphic(ground_truth_graph, ret_graph, node_match=node_attributes_are_equal):
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, llm_output_token_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count)

            # sleep for 60 seconds, to avoid the API call limit
            time.sleep(10)

        print("=========Current query process is done!=========")
        print(requestData)
        print("Total test times: ", EACH_PROMPT_RUN_TIME)
        print("Testing accuracy: ", prompt_accu/EACH_PROMPT_RUN_TIME)

    return ret


def ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count):
    print("Fail the test, and here is more info: ")
    if ground_truth_ret['type'] == 'graph':
        print("Two graph are not identical.")
    else:
        print("ground truth: ", ground_truth_ret['data'])
        print("model output: ", ret['data'])

    # Save requestData, code, ground_truth_ret['data'] into a JsonLine file
    with jsonlines.open(OUTPUT_JSONL_PATH, mode='a') as writer:
        writer.write(requestData)
        writer.write({"Result": "Fail"})
        writer.write({"Ground truth code": ground_truth_ret['reply']})
        writer.write({"LLM code": ret['reply']})
        if ground_truth_ret['type'] != 'graph':
            writer.write({"Ground truth exec": ground_truth_ret['data']})
            writer.write({"LLM code exec": ret['data']})
    return None

def ground_truth_check_accu(count, requestData, ground_truth_ret, ret, llm_output_token_count):
    print("Pass the test!")
    count += 1
    # Save requestData, code, ground_truth_ret['data'] into a JsonLine file
    with jsonlines.open(OUTPUT_JSONL_PATH, mode='a') as writer:
        writer.write(requestData)
        writer.write({"Result": "Pass"})
        writer.write({"Ground truth code": ground_truth_ret['reply']})
        writer.write({"LLM code": ret['reply']})
        if ground_truth_ret['type'] != 'graph':
            writer.write({"Ground truth exec": ground_truth_ret['data']})
            writer.write({"LLM code exec": ret['data']})
    return count

def main():
    # create 'output.jsonl' file if it does not exist
    if not os.path.exists(OUTPUT_JSONL_PATH):
        with open(OUTPUT_JSONL_PATH, 'w') as f:
            pass

    prompt_list = [
        # 3 easy ones
        # "List all ports contained in packet switch ju1.a1.m1.s2c1. Return a list.",
        # "Add a new packet_switch 'ju1.a1.m1.s4c7' on jupiter 1, aggregation block 1, domain 1, with 5 ports. Return the new graph.",
        # "Update the physical_capacity_bps from 1000 Mbps to 4000 Mbps on node ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",
        # 2nd turn
        # "Identify all CONTROL_POINT nodes that are also PACKET_SWITCH type within the AGG_BLOCK type node ju1.a4.m4. Return a list.",
        # "Display all CONTROL_DOMAIN that contains at least 3 CONTROL_POINT. Return a list.",
        # "Update all PACKET_SWITCH with node attr packet_switch_attr{switch_loc {stage: 3}} to packet_switch_attr{switch_loc {stage: 5}}. Return the graph.",
        # "Find the number of CHASSIS nodes contained in each RACK node? Return a table with headers 'RACK', 'CHASSIS Count'.",

        # 3 medium one
        "What is the bandwidth on packet switch ju1.a2.m1.s2c2? Output bandwidth unit should be in Mbps. Return only the number.",
        "What is the bandwidth on each AGG_BLOCK? Output bandwidth unit should be in Mbps. Return a table with header 'AGG_BLOCK', 'Bandwidth' on the first row.",
        "Find the first and the second largest Chassis by capacity on 'ju1.a1.m1'. Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",
        # 2nd turn
        "Show the average physical_capacity_bps for all PORT in all PACKET_SWITCH. Return a number in string.",
        "For each AGG_BLOCK, list the number of PACKET_SWITCH and PORT it contains. Return a table with headers 'AGG_BLOCK', 'Switch Count', 'Port Count'.",
        "Identify all PACKET_SWITCH nodes contains in AGG_BLOCK node ju1.a1.m1 and calculate their average physical_capacity_bps (on PORT) in bps. Return a table with headers 'Packet Switch', Average Capacity (bps)', sort by highest average capacity.",
        "Find all PACKET_SWITCH nodes that have capacity more than the average. Return a list of nodes.",

        # 3 hard ones
        "Provide a graph that contains all SUPERBLOCK and AGG_BLOCK. Create the new graph.",
        "Remove packet switch 'ju1.a1.m1.s2c4' out from Chassis c4, how to balance the capacity between Chassis? Return the balanced graph.",
        "Remove five PORT nodes from each PACKET_SWITCH node ju1.a1.m1.s2c1, ju1.a1.m1.s2c2, ju1.a1.m1.s2c3, ju1.a1.m1.s2c4, ju1.a1.m1.s2c5. Make sure after the removal the capacity between switches is still balanced. Return the list of ports that will be moved.",
        # 2nd turn
        "Identify all paths from the CONTROL_DOMAIN type node ju1.a1.dom to PORT node ju1.a1.m1.s2c1.p1, and rank them based on the lowest number of hops.",
        "Analyze the redundancy level of each SUPERBLOCK node, by calculating the number of alternative paths between pairs of CHASSIS nodes contains in SUPERBLOCK.",
        "Optimize the current network topology by identifying PACKET_SWITCH nodes that can be removed without affecting the connectivity between CONTROL_DOMAIN nodes. Return a list.",
        "Determine the optimal placement of a new PACKET_SWITCH node ju1.a1.m1.s2c9 with 5 PORT nodes in the format ju1.a1.m1.s2c9.p{i} (each has physical_capacity_bps 1000000000). Consider the current physical_capacity_bps distribution. The goal is to balance average capacity between AGG_BLOCK. Return the networkx graph.",
    ]

    userQuery(prompt_list)


if __name__=="__main__":
    main()
