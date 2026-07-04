import json
import traceback
from dotenv import load_dotenv
import openai
import copy
import pandas as pd
from collections import Counter
from prototxt_parser.prototxt import parse
import os
from ai_models_cot import summary_gen_chain, cot_plus_tool_chain, pySelfDebugger
from helper import getGraphData, extract_constraints, extract_tools, clean_up_llm_output_func, check_list_equal, node_attributes_are_equal, clean_up_output_graph_data
from error_check import MyChecker
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
OUTPUT_JSONL_PATH = 'logs/gpt4/srikanth_queries_2.jsonl'
DEBUG_LOOP_TOTAL = 3

# MODEL_SOURCE = "GOOGLE"
MODEL_SOURCE = "OPENAI"

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine="text-embedding-ada-002")
    embeddings = response['data'][0]['embedding']
    return embeddings


def rag_constraint_search(query, num_extraction=13):
    '''
    With given query, use hybrid search to find the most related constraints from RAG.
    It assumes index is already created and uploaded.
    '''
    # Pure Vector Search
    search_client = SearchClient(service_endpoint, constraint_index_name, AzureKeyCredential(azure_search_key))

    results = search_client.search(
        search_text='',
        vector=Vector(value=generate_embeddings(query), k=num_extraction, fields="constraintVector"),
        select=["label", "constraint"]
    )

    return extract_constraints(results)

def rag_tool_search(query, num_extraction=1):
    '''
    With given query, use vector search to find the most related tools from RAG.
    It assumes index is already created and uploaded.
    '''
    # Pure Vector Search
    search_client = SearchClient(service_endpoint, tool_index_name, AzureKeyCredential(azure_search_key))

    results = search_client.search(
        search_text='',
        vector=Vector(value=generate_embeddings(query), k=num_extraction, fields="descriptionVector"),
        select=["description", "tool"]
    )

    return extract_tools(results)

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
        if re.search('process_graph', step_code, re.DOTALL):
            # If it does, assign it to last_python_code_block and break the loop
            last_python_code_block = step_code
            break

    if last_python_code_block is None:
        print("No Python code block found.")

    return last_python_code_block

def error_reduce_verify(constraints_found, requestData, code, ret_graph=None, ret_list=None):
    error_reduce_self_debug_count = 0
    # Run the invariants checker on the modified graph
    print("================= Start verifying invariant constraints =================")
    verifier = MyChecker(ret_graph, ret_list)
    verifier_results, verifier_error = verifier.evaluate_all()
    if verifier_results:
        print("================= Congrats, verifiers all passed! =================")
    else:
        print("================= Start self-debugging for verifier errors =================")
        # TODO: do the RAG again based on error message
        # Self-debug for verifier errors here
        verifier_constraints_rag = rag_constraint_search(str(verifier_error), num_extraction=2)
        debug_constraints = constraints_found + verifier_constraints_rag
        print("Verifier RAG extract constraints: ", verifier_constraints_rag)

        for i in range(DEBUG_LOOP_TOTAL):  # times of self-debug loop
            error_reduce_self_debug_count += 1
            debugged_code = self_debug_process_loop(requestData,
                                                    debug_constraints,
                                                    code,
                                                    verifier_error,
                                                    debug_status_msg="================= Verifier: start self-debugging =================",
                                                    loop_time_index=i)
            try:
                _, G = getGraphData()
                exec(debugged_code)
                ret = eval("process_graph(G)")
                ret_graph_copy = clean_up_output_graph_data(ret)
                verifier = MyChecker(ret_graph_copy, ret_list)
                verifier_results, verifier_error = verifier.evaluate_all()
                if verifier_results:
                    print("================= Congrats, verifiers all passed after self-debugging! =================")
                    return verifier_results, debugged_code, error_reduce_self_debug_count
                    # break  # if the code successfully executed, break the loop
                else:
                    verifier_constraints_rag = rag_constraint_search(str(verifier_error), num_extraction=2)
                    debug_constraints = constraints_found + verifier_constraints_rag
                    print("Verifier RAG extract constraints: ", verifier_constraints_rag)
                    if i == DEBUG_LOOP_TOTAL - 1:
                        print("Fail the test, the code cannot pass all verifiers.")
                        # if it still fails due to verifier error, log it differently
                        with jsonlines.open(OUTPUT_JSONL_PATH, mode='a') as writer:
                            writer.write(requestData)
                            writer.write({"Result": "Fail, code cannot pass all verifiers"})
                            writer.write({"LLM code": debugged_code})
                            writer.write({"Error": str(verifier_error)})
                        # break from the current for loop
                        continue

            except Exception as e:
                print(e)
                print("Fail, verifier debugged code cannot run.")
                # break from the current for loop
                continue

    return verifier_results, None, error_reduce_self_debug_count

def self_debug_execution_error(code, requestData, constraints_found):
    execution_error_self_debug = 0

    # Got the detailed error from exec(code)
    exc_type, ex, tb = sys.exc_info()
    imported_tb_info = traceback.extract_tb(tb)[-1]
    line_number = imported_tb_info[1]
    print_format = '{}: Exception in line: {}. Message: {}'
    error_details = print_format.format(exc_type.__name__, line_number, ex)
    print("Fail due to errors:", error_details)

    # Add self-debug here
    for i in range(DEBUG_LOOP_TOTAL):  # 3 times self-debug loop
        execution_error_self_debug += 1
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
            return debugged_code, ret, execution_error_self_debug
        except Exception as e:
            # Got the detailed error from exec(code)
            exc_type, ex, tb = sys.exc_info()
            imported_tb_info = traceback.extract_tb(tb)[-1]
            line_number = imported_tb_info[1]
            print_format = '{}: Exception in line: {}. Message: {}'
            error_details = print_format.format(exc_type.__name__, line_number, ex)
            print("Fail due to errors:", error_details)

            if i == DEBUG_LOOP_TOTAL - 1:
                print("Fail the test, the code cannot run.")
                # if it still fails due to execution error, log it differently
                with jsonlines.open(OUTPUT_JSONL_PATH, mode='a') as writer:
                    writer.write(requestData)
                    writer.write({"Result": "Fail, code cannot run"})
                    writer.write({"LLM code": debugged_code})
                    writer.write({"Error": str(e)})
                # break from the current for loop
                continue

    return None, None, execution_error_self_debug

def diff_model_source_output_format(model_output):
    if MODEL_SOURCE == "OPENAI":
        model_output_clean = model_output.to_json()['kwargs']['content']

    if MODEL_SOURCE == "GOOGLE":
        model_output_clean = model_output

    return model_output_clean

def userQuery(prompt_list):
    # Load the existing prompt and golden answers from Json
    golden_answer_filename = 'golden_answer_generator/prompt_golden_ans.json'
    with open(golden_answer_filename, "r") as fa:
        allAnswer = json.load(fa)


    # for each prompt in the prompt_list, append it as the value of {'query': prompt}
    for each_prompt in prompt_list:
        print("Query: ", each_prompt)
        requestData = {'query': each_prompt}

        constraints_found = rag_constraint_search(each_prompt)
        print("Constraints: ", constraints_found)
        tool_found = rag_tool_search(each_prompt)
        print("Tools: ", tool_found)

        prompt_accu = 0
        _, G = getGraphData()

        # Reset ret when it's a new test
        ret = None
        # init
        first_step_execution_debug_count = 0
        second_step_execution_debug_count = 0
        third_step_execution_debug_count = 0
        first_step_verifier_debug_count = 0
        second_step_verifier_debug_count = 0
        third_step_verifier_debug_count = 0

        # Run each prompt for 10 times
        for i in range(EACH_PROMPT_RUN_TIME):
            if requestData['query'] not in allAnswer.keys():
                # terminate the code with error message
                raise SystemExit('Un-support ground truth for the current prompt.')

            print("Find the prompt in the list.")

            print("Calling model")
            summary_output = summary_gen_chain.invoke({"input": requestData['query']})
            # import pdb; pdb.set_trace()
            # process the AIMessage output
            step_summary = diff_model_source_output_format(summary_output)

            # Use regular expressions to split the string by 'Step X:' where X is a digit
            steps = re.split('Step \d+: ', step_summary)
            # Remove the first element which is empty due to the split
            steps = steps[1:]

            first_step_llm = cot_plus_tool_chain.invoke({"input": requestData['query'],
                                                           "constraints": constraints_found,
                                                           "step": steps[0],
                                                           "code": "None",
                                                           "tool": tool_found})
            first_step_code = diff_model_source_output_format(first_step_llm)
            first_step_code = clean_up_llm_output_func(first_step_code)
            print("Step 1: ", steps[0])
            print("Code generated: ", first_step_code)
            try:
                exec(first_step_code)
                first_step_ret = eval("process_graph(G)")
            except Exception:
                self_debugged_code_1, first_step_ret, first_step_execution_debug_count = self_debug_execution_error(first_step_code, requestData, constraints_found)
                if self_debugged_code_1:
                    first_step_code = self_debugged_code_1

            # if the type of ret is string, turn it into a json object
            if isinstance(first_step_ret, str):
                first_step_ret = json.loads(first_step_ret)
            if first_step_ret['type'] == 'graph':
                first_step_ret_graph = clean_up_output_graph_data(first_step_ret)
                # Run the error reducer
                first_step_verify_result, cot_first_step_debugged_code, first_step_verifier_debug_count = error_reduce_verify(constraints_found, requestData, first_step_code, ret_graph=first_step_ret_graph, ret_list=None)
            else:
                first_step_verify_result, cot_first_step_debugged_code, first_step_verifier_debug_count = error_reduce_verify(constraints_found, requestData, first_step_code, ret_graph=None, ret_list=first_step_ret)

            print("first_step_verify_result: ", first_step_verify_result)
            if cot_first_step_debugged_code:
                # if the debugged code is not none, replace the original code with the debugged code
                first_step_code = cot_first_step_debugged_code

            second_step_llm = cot_plus_tool_chain.invoke({"input": requestData['query'],
                                                            "constraints": constraints_found,
                                                            "step": steps[1],
                                                            "code": first_step_code,
                                                            "tool": tool_found})
            second_step_code = diff_model_source_output_format(second_step_llm)
            second_step_code = clean_up_llm_output_func(second_step_code)
            print("Step 2: ", steps[1])
            print("Code generated: ", second_step_code)

            try:
                exec(second_step_code)
                second_step_ret = eval("process_graph(G)")
            except Exception:
                self_debugged_code_2, second_step_ret, second_step_execution_debug_count = self_debug_execution_error(second_step_code, requestData,
                                                                           constraints_found)
                if self_debugged_code_2:
                    second_step_code = self_debugged_code_2

            # if the type of ret is string, turn it into a json object
            if isinstance(second_step_ret, str):
                second_step_ret = json.loads(second_step_ret)
            if second_step_ret['type'] == 'graph':
                second_step_ret_graph = clean_up_output_graph_data(second_step_ret)
                # Run the error reducer
                second_step_verify_result, cot_second_step_debugged_code, second_step_verifier_debug_count = error_reduce_verify(constraints_found, requestData, second_step_code,
                                                               ret_graph=second_step_ret_graph, ret_list=None)
            else:
                second_step_verify_result, cot_second_step_debugged_code, second_step_verifier_debug_count = error_reduce_verify(constraints_found, requestData, second_step_code,
                                                               ret_graph=None, ret_list=second_step_ret)

            print("second_step_verify_result: ", second_step_verify_result)
            if cot_second_step_debugged_code:
                # if the debugged code is not none, replace the original code with the debugged code
                second_step_code = cot_second_step_debugged_code

            third_step_llm = cot_plus_tool_chain.invoke({"input": requestData['query'],
                                                           "constraints": constraints_found,
                                                           "step": steps[2],
                                                           "code": second_step_code,
                                                           "tool": tool_found})

            third_step_code = diff_model_source_output_format(third_step_llm)
            if "```python" in third_step_code:
                third_step_code = clean_up_llm_output_func(third_step_code)
                print("Step 3: ", steps[2])
                print("Code generated: ", third_step_code)

                try:
                    exec(third_step_code)
                    third_step_ret = eval("process_graph(G)")
                except Exception:
                    self_debugged_code_3, third_step_ret, third_step_execution_debug_count = self_debug_execution_error(third_step_code, requestData,
                                                                               constraints_found)
                    if self_debugged_code_3:
                        third_step_code = self_debugged_code_3

                # if the type of ret is string, turn it into a json object
                if isinstance(third_step_ret, str):
                    third_step_ret = json.loads(third_step_ret)
                if third_step_ret['type'] == 'graph':
                    third_step_ret_graph = clean_up_output_graph_data(third_step_ret)
                    # Run the error reducer
                    third_step_verify_result, cot_third_step_debugged_code, third_step_verifier_debug_count = error_reduce_verify(constraints_found, requestData, third_step_code,
                                                                                                 ret_graph=third_step_ret_graph, ret_list=None)
                else:
                    third_step_verify_result, cot_third_step_debugged_code, third_step_verifier_debug_count = error_reduce_verify(constraints_found, requestData, third_step_code,
                                                                                                 ret_graph=None, ret_list=second_step_ret)
                print("third_step_verify_result: ", third_step_verify_result)
                if cot_third_step_debugged_code:
                    # if the debugged code is not none, replace the original code with the debugged code
                    third_step_code = cot_third_step_debugged_code

            # code = third_step_code
            code = extract_final_code(first_step_code,
                                        second_step_code,
                                        third_step_code)
            total_execution_debug_count = first_step_execution_debug_count + second_step_execution_debug_count + third_step_execution_debug_count
            total_verifier_debug_count = first_step_verifier_debug_count + second_step_verifier_debug_count + third_step_verifier_debug_count

            llm_output_token_count = 0
            # if code contains python package import, remove all lines related to it
            # code = clean_up_llm_output_func(answer)
            print(code)
            try:
                exec(code)
                ret = eval("process_graph(G)")
            except Exception:
                debugged_code, ret, _ = self_debug_execution_error(code, requestData, constraints_found)
            # TODO: fix misleading print
            # print("================= Error reduce progress + 1: Code can run! =================")
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
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, llm_output_token_count, total_execution_debug_count, total_verifier_debug_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count, total_execution_debug_count, total_verifier_debug_count)

            elif ground_truth_ret['type'] == 'list':
                # Use Counter to check if two lists contain the same items, including duplicate items.
                if check_list_equal(ground_truth_ret['data'], ret['data']):
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, llm_output_token_count, total_execution_debug_count, total_verifier_debug_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count, total_execution_debug_count, total_verifier_debug_count)

            elif ground_truth_ret['type'] == 'table':
                if ground_truth_ret['data'] == ret['data']:
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, llm_output_token_count, total_execution_debug_count, total_verifier_debug_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count, total_execution_debug_count, total_verifier_debug_count)

            elif ground_truth_ret['type'] == 'graph':
                # Undirected graphs will be converted to a directed graph
                # with two directed edges for each undirected edge.
                ground_truth_graph = nx.Graph(ground_truth_ret['data'])
                # TODO: fix ret_graph_copy reference possible error, when it's not created.
                ret_graph = nx.Graph(ret_graph_copy)

                # Check if two graphs are identical, no weights considered
                if nx.is_isomorphic(ground_truth_graph, ret_graph, node_match=node_attributes_are_equal):
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, llm_output_token_count, total_execution_debug_count, total_verifier_debug_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count, total_execution_debug_count, total_verifier_debug_count)

            # sleep for 60 seconds, to avoid the API call limit
            time.sleep(5)

        print("=========Current query process is done!=========")
        print(requestData)
        print("Total test times: ", EACH_PROMPT_RUN_TIME)
        print("Testing accuracy: ", prompt_accu/EACH_PROMPT_RUN_TIME)

    return ret


def ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count, execution_debug_count, verifier_debug_count):
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
        writer.write({"Execution debug count": execution_debug_count})
        writer.write({"Verifier debug count": verifier_debug_count})
        writer.write({"Ground truth code": ground_truth_ret['reply']})
        writer.write({"LLM code": ret['reply']})
        if ground_truth_ret['type'] != 'graph':
            writer.write({"Ground truth exec": ground_truth_ret['data']})
            writer.write({"LLM code exec": ret['data']})
    return None

def ground_truth_check_accu(count, requestData, ground_truth_ret, ret, llm_output_token_count, execution_debug_count, verifier_debug_count):
    print("Pass the test!")
    count += 1
    # Save requestData, code, ground_truth_ret['data'] into a JsonLine file
    with jsonlines.open(OUTPUT_JSONL_PATH, mode='a') as writer:
        writer.write(requestData)
        writer.write({"Result": "Pass"})
        writer.write({"Execution debug count": execution_debug_count})
        writer.write({"Verifier debug count": verifier_debug_count})
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
        # # 3 easy ones
        # "List all ports contained in packet switch ju1.a1.m1.s2c1. Return a list.",
        # "Add a new packet_switch 'ju1.a1.m1.s4c7' on jupiter 1, aggregation block 1, domain 1, with 5 ports, each port has physical_capacity_bps as 1000. Add node type and edges too. Return the new graph.",
        # "Update the physical_capacity_bps from 1000 Mbps to 4000 Mbps on node ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",
        # # 2nd turn
        # "Identify all CONTROL_POINT nodes that are also PACKET_SWITCH type within the AGG_BLOCK type node ju1.a4.m4. Return a list.",
        # "Display all CONTROL_DOMAIN that contains at least 3 CONTROL_POINT. Return a list.",
        # "Update all PACKET_SWITCH with node attr packet_switch_attr{switch_loc {stage: 3}} to packet_switch_attr{switch_loc {stage: 5}}. Return the graph.",
        # "Find the number of CHASSIS nodes contained in each RACK node? Return a table with headers 'RACK', 'CHASSIS Count'.",
        #
        # # 3 medium one
        # "What is the bandwidth on packet switch ju1.a2.m1.s2c2? Output bandwidth unit should be in Mbps. Return only the number.",
        # "What is the bandwidth on each AGG_BLOCK? Output bandwidth unit should be in Mbps. Return a table with header 'AGG_BLOCK', 'Bandwidth' on the first row.",
        # "Find the first and the second largest Chassis by capacity on 'ju1.a1.m1'. Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",
        # # 2nd turn
        # "Show the average physical_capacity_bps for all PORT in all PACKET_SWITCH. Return a number in string.",
        # "For each AGG_BLOCK, list the number of PACKET_SWITCH and PORT it contains. Return a table with headers 'AGG_BLOCK', 'Switch Count', 'Port Count'.",
        # "Identify all PACKET_SWITCH nodes contains in AGG_BLOCK node ju1.a1.m1 and calculate their average physical_capacity_bps (on PORT) in bps. Return a table with headers 'Packet Switch', Average Capacity (bps)', sort by highest average capacity.",
        # "Find all PACKET_SWITCH nodes that have capacity more than the average. Return a list of nodes.",
        #
        # # 3 hard ones
        # "Provide a graph that contains all SUPERBLOCK and AGG_BLOCK. Create the new graph.",
        # "Remove packet switch 'ju1.a1.m1.s2c4' out from Chassis c4, how to balance the capacity between Chassis? Return the balanced graph.",
        # "Remove five PORT nodes (start from p1) from each PACKET_SWITCH node ju1.a1.m1.s2c1, ju1.a1.m1.s2c2, ju1.a1.m1.s2c3, ju1.a1.m1.s2c4, ju1.a1.m1.s2c5. Make sure after the removal the capacity between switches is still balanced. Return the list of ports that will be moved.",
        # # 2nd turn
        # "Identify all paths from the CONTROL_DOMAIN type node ju1.a1.dom to PORT node ju1.a1.m1.s2c1.p1, and rank them based on the lowest number of hops.",
        # "Analyze the redundancy level of each SUPERBLOCK node, by calculating the number of alternative paths between pairs of CHASSIS nodes contains in SUPERBLOCK.",
        # "Optimize the current network topology by identifying PACKET_SWITCH nodes that can be removed without affecting the connectivity between CONTROL_DOMAIN nodes. Return a list.",
        # "Determine the optimal placement of a new PACKET_SWITCH node ju1.a1.m1.s2c9 with 5 PORT nodes in the format ju1.a1.m1.s2c9.p{i} (each has physical_capacity_bps 1000000000). Consider the current physical_capacity_bps distribution. The goal is to balance average capacity between AGG_BLOCK. Return the networkx graph.",
        #
        # ## More for experiments
        # "Find all ports contained in packet switch ju1.a1.m1.s2c3. Return a list.",
        # "Find all ports contained in packet switch ju1.a1.m1.s2c7. Return a list.",
        # "Find all ports contained in packet switch ju1.a1.m1.s2c4. Return a list.",
        # "Find all ports contained in packet switch ju1.a1.m1.s2c5. Return a list.",
        # "Find all ports contained in packet switch ju1.a1.m1.s3c3. Return a list.",
        # "Find all ports contained in packet switch ju1.a1.m1.s3c6. Return a list.",
        #
        # "Update the physical_capacity_bps to 2000 Mbps on node ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",
        # "Update the physical_capacity_bps to 5000 Mbps on node ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",
        # "Update the physical_capacity_bps to 6000 Mbps on node ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",
        # "Update the physical_capacity_bps to 3000 Mbps on node ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",
        # "Update the physical_capacity_bps to 7000 Mbps on node ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",
        # "Update the physical_capacity_bps to 8000 Mbps on node ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",
        #
        # "Identify all CONTROL_POINT nodes that are also PACKET_SWITCH type within the AGG_BLOCK type node ju1.a1.m1. Return a list.",
        # "Identify all CONTROL_POINT nodes that are also PACKET_SWITCH type within the AGG_BLOCK type node ju1.a1.m2. Return a list.",
        # "Identify all CONTROL_POINT nodes that are also PACKET_SWITCH type within the AGG_BLOCK type node ju1.a2.m3. Return a list.",
        # "Identify all CONTROL_POINT nodes that are also PACKET_SWITCH type within the AGG_BLOCK type node ju1.a2.m4. Return a list.",
        # "Identify all CONTROL_POINT nodes that are also PACKET_SWITCH type within the AGG_BLOCK type node ju1.a3.m1. Return a list.",
        # "Identify all CONTROL_POINT nodes that are also PACKET_SWITCH type within the AGG_BLOCK type node ju1.a3.m2. Return a list.",
        #
        # "Update all PACKET_SWITCH with node attr packet_switch_attr{switch_loc {stage: 2}} to packet_switch_attr{switch_loc {stage: 5}}. Return the graph.",
        # "Update all PACKET_SWITCH with node attr packet_switch_attr{switch_loc {stage: 2}} to packet_switch_attr{switch_loc {stage: 6}}. Return the graph.",
        # "Update all PACKET_SWITCH with node attr packet_switch_attr{switch_loc {stage: 3}} to packet_switch_attr{switch_loc {stage: 6}}. Return the graph.",
        # "Update all PACKET_SWITCH with node attr packet_switch_attr{switch_loc {stage: 3}} to packet_switch_attr{switch_loc {stage: 7}}. Return the graph.",
        # "Update all PACKET_SWITCH with node attr packet_switch_attr{switch_loc {stage: 2}} to packet_switch_attr{switch_loc {stage: 1}}. Return the graph.",
        # "Update all PACKET_SWITCH with node attr packet_switch_attr{switch_loc {stage: 3}} to packet_switch_attr{switch_loc {stage: 1}}. Return the graph.",
        #
        # "What is the bandwidth on packet switch ju1.a2.m1.s3c1? Output bandwidth unit should be in Mbps. Return only the number.",
        # "What is the bandwidth on packet switch ju1.a2.m1.s2c7? Output bandwidth unit should be in Mbps. Return only the number.",
        # "What is the bandwidth on packet switch ju1.a2.m1.s3c4? Output bandwidth unit should be in Mbps. Return only the number.",
        # "What is the bandwidth on packet switch ju1.a2.m1.s3c6? Output bandwidth unit should be in Mbps. Return only the number.",
        # "What is the bandwidth on packet switch ju1.a2.m1.s2c3? Output bandwidth unit should be in Mbps. Return only the number.",
        # "What is the bandwidth on packet switch ju1.a2.m1.s2c4? Output bandwidth unit should be in Mbps. Return only the number.",
        #
        # "Find the first and the second largest Chassis by capacity on 'ju1.a1.m2' (do not check its type). Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",
        # "Find the first and the second largest Chassis by capacity on 'ju1.a2.m2' (do not check its type). Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",
        # "Find the first and the second largest Chassis by capacity on 'ju1.a3.m2' (do not check its type). Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",
        # "Find the first and the second largest Chassis by capacity on 'ju1.a4.m2' (do not check its type). Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",
        # "Find the first and the second largest Chassis by capacity on 'ju1.a3.m3' (do not check its type). Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",
        # "Find the first and the second largest Chassis by capacity on 'ju1.a4.m3' (do not check its type). Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",
        #
        # "Identify all PACKET_SWITCH nodes contains in AGG_BLOCK node ju1.a1.m2 and calculate their average physical_capacity_bps (on PORT) in bps. Return a table with headers 'Packet Switch', Average Capacity (bps)', sort by highest average capacity.",
        # "Identify all PACKET_SWITCH nodes contains in AGG_BLOCK node ju1.a2.m2 and calculate their average physical_capacity_bps (on PORT) in bps. Return a table with headers 'Packet Switch', Average Capacity (bps)', sort by highest average capacity.",
        # "Identify all PACKET_SWITCH nodes contains in AGG_BLOCK node ju1.a3.m2 and calculate their average physical_capacity_bps (on PORT) in bps. Return a table with headers 'Packet Switch', Average Capacity (bps)', sort by highest average capacity.",
        # "Identify all PACKET_SWITCH nodes contains in AGG_BLOCK node ju1.a4.m2 and calculate their average physical_capacity_bps (on PORT) in bps. Return a table with headers 'Packet Switch', Average Capacity (bps)', sort by highest average capacity.",
        # "Identify all PACKET_SWITCH nodes contains in AGG_BLOCK node ju1.a4.m1 and calculate their average physical_capacity_bps (on PORT) in bps. Return a table with headers 'Packet Switch', Average Capacity (bps)', sort by highest average capacity.",
        # "Identify all PACKET_SWITCH nodes contains in AGG_BLOCK node ju1.a4.m3 and calculate their average physical_capacity_bps (on PORT) in bps. Return a table with headers 'Packet Switch', Average Capacity (bps)', sort by highest average capacity.",
        #
        # "Remove packet switch 'ju1.a1.m1.s2c1' out from Chassis c1, how to balance the capacity between Chassis? Return the balanced graph.",
        # "Remove packet switch 'ju1.a1.m1.s2c2' out from Chassis c2, how to balance the capacity between Chassis? Return the balanced graph.",
        # "Remove packet switch 'ju1.a1.m1.s2c3' out from Chassis c3, how to balance the capacity between Chassis? Return the balanced graph.",
        # "Remove packet switch 'ju1.a1.m1.s2c5' out from Chassis c5, how to balance the capacity between Chassis? Return the balanced graph.",
        # "Remove packet switch 'ju1.a1.m1.s3c1' out from Chassis c5, how to balance the capacity between Chassis? Return the balanced graph.",
        # "Remove packet switch 'ju1.a1.m1.s3c2' out from Chassis c5, how to balance the capacity between Chassis? Return the balanced graph.",
        #
        # "Identify all paths from the CONTROL_DOMAIN type node ju1.a1.dom to PORT node ju1.a1.m1.s2c1.p2, and rank them based on the lowest number of hops.",
        # "Identify all paths from the CONTROL_DOMAIN type node ju1.a1.dom to PORT node ju1.a1.m1.s2c1.p3, and rank them based on the lowest number of hops.",
        # "Identify all paths from the CONTROL_DOMAIN type node ju1.a1.dom to PORT node ju1.a1.m1.s2c1.p4, and rank them based on the lowest number of hops.",
        # "Identify all paths from the CONTROL_DOMAIN type node ju1.a1.dom to PORT node ju1.a1.m1.s2c1.p5, and rank them based on the lowest number of hops.",
        # "Identify all paths from the CONTROL_DOMAIN type node ju1.a2.dom to PORT node ju1.a1.m1.s2c1.p5, and rank them based on the lowest number of hops.",
        # "Identify all paths from the CONTROL_DOMAIN type node ju1.a3.dom to PORT node ju1.a1.m1.s2c1.p5, and rank them based on the lowest number of hops.",
        #
        # "Display all CONTROL_DOMAIN that contains at least 4 CONTROL_POINT. Return a list.",
        # "Display all CONTROL_DOMAIN that contains at least 5 CONTROL_POINT. Return a list.",
        # "Display all CONTROL_DOMAIN that contains at least 6 CONTROL_POINT. Return a list.",
        # "Display all CONTROL_DOMAIN that contains at least 7 CONTROL_POINT. Return a list.",
        # "Display all CONTROL_DOMAIN that contains at least 8 CONTROL_POINT. Return a list.",

        # From srikanth
        "What is the total capacity of all the ports of the packet switch ju1.a1.m1.s2c1? Return only the number",
        "How many packet switches have more capacity than ju1.a2.m3.s2c3? Can you give me an example of such a switch? Return the text",
        "How many packet switches are there in a typical chassis? Can you find me an example of a chassis with more or fewer switches than normal?",
        "How many ports in the graph? What type of children nodes does it have (count the average number)? What type of parent nodes does it have (count the average number)?"
        "How many agg blocks are there in a common superblock? Also find the min number and max number of agg blocks.",
        "Does the graph have any information about power usage or physical location of the switches? Return 'Yes' or 'No'.",
        "What are the other ports on the switch that has the port ju1.a1.m1.s2c1.p10? Return a list.",
        "For each superblock, tell me the numbers of agg blocks, packet switches and ports. Return a list.",
        "Attach an attribute called capacity fraction to each of the port type node in the graph whose value is the fraction of their capacity relative to the total capacity of all ports in that switch. Return the updated graph.",
        "Divide the nodes into clusters using louvain communities algorithm. Add the cluster index as a node attrbute called 'Louvain Communities ClusterID'. Return the updated graph.",

    ]

    userQuery(prompt_list)


if __name__=="__main__":
    main()
