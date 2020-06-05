from aser.database.db_API import KG_Connection
import time
from tqdm import tqdm
# import aser
import ujson as json
from multiprocessing import Pool
import spacy
import random
import pandas
import numpy as np
from itertools import combinations
from scipy import spatial
import os

def get_ConceptNet_info(file_path):
    tmp_collection = dict()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_words = line[:-1].split('\t')
            if tmp_words[3] == '0':
                continue
            if tmp_words[0] not in tmp_collection:
                tmp_collection[tmp_words[0]] = list()
            tmp_collection[tmp_words[0]].append((tmp_words[1], tmp_words[2]))
    return tmp_collection


def load_jsonlines(file_name):
    extracted_file = list()
    with open(file_name, 'r') as f:
        for line in f:
            tmp_info = json.loads(line)
            extracted_file.append(tmp_info)
    return extracted_file


def chunks(l, group_number):
    if len(l) < 10:
        return [l]
    group_size = int(len(l) / group_number)
    final_data_groups = list()
    for i in range(0, len(l), group_size):
        final_data_groups.append(l[i:i+group_size])
    return final_data_groups


def match_commonsense_and_aser(sample_pairs, ASER):
    tmp_dict = dict()
    for tmp_tuple in sample_pairs:
        head_words = tmp_tuple['head'].split(' ')
        tail_words = tmp_tuple['tail'].split(' ')
        all_words = head_words + tail_words
        tmp_key = tmp_tuple['head'] + '$$' + tmp_tuple['tail']
        matched_eventualities = list()
        for tmp_event in ASER:
            is_match = True
            for w in all_words:
                if w not in tmp_event['words'].split(' '):
                    is_match = False
                    break
            if is_match:
                matched_eventualities.append(tmp_event)
        tmp_dict[tmp_key] = matched_eventualities
    return tmp_dict


def match_commonsense_and_aser_edge(sample_pairs, ASER):
    tmp_dict = dict()
    for tmp_tuple in sample_pairs:
        head_words = tmp_tuple['head'].split(' ')
        tail_words = tmp_tuple['tail'].split(' ')
        all_words = head_words + tail_words
        tmp_key = tmp_tuple['head'] + '$$' + tmp_tuple['tail']
        matched_eventualities = list()
        for tmp_event in ASER:
            is_match = True
            edge_words = tmp_event['event_1_words'].split(' ') + tmp_event['event_2_words'].split(' ')
            for w in all_words:
                if w not in edge_words:
                    is_match = False
                    break
            if is_match:
                matched_eventualities.append(tmp_event)
        tmp_dict[tmp_key] = matched_eventualities
    return tmp_dict


def find_head_tail_position_from_graph(graph, pattern_keywords, direction, loop):
    if loop == 5:
        print('Current loop is 5, we need to stop, something is wrong, please check.')
        return []
    if len(pattern_keywords) == 0:
        return []
    if direction == '<':
        if len(pattern_keywords) == 3:
            potential_links = list()
            for edge in graph:
                if edge[1] == pattern_keywords[1]:
                    if pattern_keywords[0] == 'head':
                        potential_links.append([edge[2], edge[0]])
                    else:
                        if pattern_keywords[0] == edge[2][0]:
                            potential_links.append([edge[0]])
            return potential_links
        else:
            potential_links = list()
            for edge in graph:
                if edge[1] == pattern_keywords[1]:
                    if pattern_keywords[0] == 'head':
                        tmp_link = [edge[2], edge[0]]
                        new_pattern_keywords = pattern_keywords[2:]
                        rest_links = find_head_tail_position_from_graph(graph, new_pattern_keywords, direction, loop+1)
                        for tmp_rest_link in rest_links:
                            tmp_link = tmp_link + tmp_rest_link
                            potential_links.append(tmp_link)
                    else:
                        if pattern_keywords[0] == edge[2][0]:
                            tmp_link = [edge[0]]
                            new_pattern_keywords = pattern_keywords[2:]
                            rest_links = find_head_tail_position_from_graph(graph, new_pattern_keywords, direction,
                                                                            loop + 1)
                            for tmp_rest_link in rest_links:
                                tmp_link = tmp_link + tmp_rest_link
                                potential_links.append(tmp_link)
            return potential_links
    else:
        if len(pattern_keywords) == 3:
            potential_links = list()
            for edge in graph:
                if edge[1] == pattern_keywords[1]:
                    if pattern_keywords[0] == 'head':
                        potential_links.append([edge[0], edge[2]])
                    else:
                        if pattern_keywords[0] == edge[0][0]:
                            potential_links.append([edge[2]])
            return potential_links
        else:
            potential_links = list()
            for edge in graph:
                if edge[1] == pattern_keywords[1]:
                    if pattern_keywords[0] == 'head':
                        tmp_link = [edge[0], edge[2]]
                        new_pattern_keywords = pattern_keywords[2:]
                        rest_links = find_head_tail_position_from_graph(graph, new_pattern_keywords, direction, loop+1)
                        for tmp_rest_link in rest_links:
                            tmp_link = tmp_link + tmp_rest_link
                            potential_links.append(tmp_link)
                    else:
                        if pattern_keywords[0] == edge[0][0]:
                            tmp_link = [edge[2]]
                            new_pattern_keywords = pattern_keywords[2:]
                            rest_links = find_head_tail_position_from_graph(graph, new_pattern_keywords, direction,
                                                                            loop + 1)
                            for tmp_rest_link in rest_links:
                                tmp_link = tmp_link + tmp_rest_link
                                potential_links.append(tmp_link)
            return potential_links


def extract_knowledge_with_focused_position(graph, pattern_keywords, focused_position):
    if len(pattern_keywords) == 0:
        return focused_position[0]
    else:
        extracted_pattern = list()
        extracted_nodes = [focused_position]
        while len(extracted_pattern) != len(pattern_keywords):
            found_new_node = False
            for edge in graph:
                if edge[1] in pattern_keywords and edge[1] not in extracted_pattern:
                    if edge[0] in extracted_nodes:
                        extracted_nodes.append(edge[2])
                        found_new_node = True
                        extracted_pattern.append(edge[1])
                    elif edge[2] in extracted_nodes:
                        extracted_nodes.append(edge[0])
                        found_new_node = True
                        extracted_pattern.append(edge[1])
            if not found_new_node:
                break
        if len(extracted_pattern) == len(pattern_keywords):
            sorted_nodes = sorted(extracted_nodes, key=lambda x: x[1])
            tmp_knowledge = ''
            for w in sorted_nodes:
                tmp_knowledge += w[0]
                tmp_knowledge += ' '
            return tmp_knowledge[:-1]
        else:
            return None


def extract_knowledge_from_graph_with_knowledge(graph, pattern):
    head_pattern = pattern.split(')')[0][1:]
    if head_pattern == '':
        head_keywords = []
    else:
        head_keywords = head_pattern.split('-')[1:-1]
    internal_pattern = pattern.split(')')[1].split('(')[0]
    tail_pattern = pattern.split('(')[2][:-1]
    if tail_pattern == '':
        tail_keywords = []
    else:
        tail_keywords = tail_pattern.split('-')[1:-1]
    focus_nodes = list()

    # We need to detect double direction
    if '<-' in internal_pattern and '->' in internal_pattern:
        all_paths = list()
        # we find a double direction
        if internal_pattern[0] == '<':
            middle_word = internal_pattern.split('<-')[-1].split('->')[0]
            first_half_pattern = internal_pattern.split(middle_word)[0]
            first_half_keywords = first_half_pattern.split('<-')
            first_half_keywords[0] = 'head'
            first_half_keywords[-1] = 'tail'
            first_half_paths = find_head_tail_position_from_graph(graph=graph, pattern_keywords=first_half_keywords, direction='<', loop=0)
            second_half_pattern = internal_pattern.split(middle_word)[1]
            second_half_keywords = second_half_pattern.split('->')
            second_half_keywords[0] = 'head'
            second_half_keywords[-1] = 'tail'
            second_half_paths = find_head_tail_position_from_graph(graph=graph, pattern_keywords=second_half_keywords,
                                                                  direction='>', loop=0)
            for tmp_first_half_path in first_half_paths:
                for tmp_second_half_path in second_half_paths:
                    if tmp_first_half_path[-1] == tmp_second_half_path[0] and tmp_first_half_path[-1][0] == middle_word:
                        all_paths.append((tmp_first_half_path[0], tmp_second_half_path[-1]))
        else:
            middle_word = internal_pattern.split('->')[-1].split('<-')[0]
            first_half_pattern = internal_pattern.split(middle_word)[0]
            first_half_keywords = first_half_pattern.split('->')
            first_half_keywords[0] = 'head'
            first_half_keywords[-1] = 'tail'
            first_half_paths = find_head_tail_position_from_graph(graph=graph, pattern_keywords=first_half_keywords,
                                                                  direction='>', loop=0)
            second_half_pattern = internal_pattern.split(middle_word)[1]
            second_half_keywords = second_half_pattern.split('<-')
            second_half_keywords[0] = 'head'
            second_half_keywords[-1] = 'tail'
            second_half_paths = find_head_tail_position_from_graph(graph=graph, pattern_keywords=second_half_keywords,
                                                                   direction='<', loop=0)
            for tmp_first_half_path in first_half_paths:
                for tmp_second_half_path in second_half_paths:
                    if tmp_first_half_path[-1] == tmp_second_half_path[0] and tmp_first_half_path[-1][0] == middle_word:
                        all_paths.append((tmp_first_half_path[0], tmp_second_half_path[-1]))
    else:
        if internal_pattern[0] == '<':
            pattern_keywords = internal_pattern.split('<-')
        else:
            pattern_keywords = internal_pattern.split('->')
        pattern_keywords[0] = 'head'
        pattern_keywords[-1] = 'tail'
        all_paths = find_head_tail_position_from_graph(graph=graph, pattern_keywords=pattern_keywords, direction=internal_pattern[0], loop=0)

    extracted_knowledge_list = list()
    for tmp_path in all_paths:
        head_knowledge = extract_knowledge_with_focused_position(graph, head_keywords, tmp_path[0])
        tail_knowledge = extract_knowledge_with_focused_position(graph, tail_keywords, tmp_path[-1])
        if head_knowledge and tail_knowledge:
            extracted_knowledge_list.append(head_knowledge + '$$' + tail_knowledge)
    return extracted_knowledge_list


def extract_knowledge_from_eventuality_set(patterns, eventuality_set):
    tmp_eventuality_dict = dict()
    tmp_extracted_knowledge = dict()
    for r in patterns:
        tmp_extracted_knowledge[r] = dict()
        for tmp_pattern in patterns[r]:
            tmp_extracted_knowledge[r][tmp_pattern[0]] = dict()
    for i, tmp_e in enumerate(eventuality_set):
        doc = nlp(tmp_e['words'])
        all_dependency_edges = list()
        for word in doc:
            all_dependency_edges.append(((word.head.norm_, word.head.i), word.dep_, (word.norm_, word.i)))
        for r in patterns:
            for pattern in patterns[r]:
                tmp_knowledge_list = extract_knowledge_from_graph_with_knowledge(all_dependency_edges, pattern[0])
                for tmp_knowledge in tmp_knowledge_list:
                    if tmp_knowledge not in tmp_extracted_knowledge[r][pattern[0]]:
                        tmp_extracted_knowledge[r][pattern[0]][tmp_knowledge] = 0
                    tmp_extracted_knowledge[r][pattern[0]][tmp_knowledge] += tmp_e['frequency']
                    if tmp_knowledge not in tmp_eventuality_dict:
                        tmp_eventuality_dict[tmp_knowledge] = list()
                    tmp_e['graph'] = all_dependency_edges
                    tmp_eventuality_dict[tmp_knowledge].append(tmp_e)
        if i % 1000 == 0:
            print('finished:', i, '/', len(eventuality_set))
    return tmp_extracted_knowledge, tmp_eventuality_dict


def eventuality_to_graph(tmp_eventuality):
    doc = nlp(tmp_eventuality['words'])
    all_dependency_edges = list()
    for word in doc:
        all_dependency_edges.append(((word.head.norm_, word.head.i), word.dep_, (word.norm_, word.i)))
    return all_dependency_edges


def eventuality_set_to_graph_set(eventuality_set):
    tmp_event_id_to_graph = dict()
    for i, tmp_eventuality in enumerate(eventuality_set):
        tmp_graph = eventuality_to_graph(tmp_eventuality)
        tmp_event_id_to_graph[tmp_eventuality['id']] = tmp_graph
        if i % 10000 == 0:
            print(i, '/', len(eventuality_set))
    return tmp_event_id_to_graph


def extract_knowledge_from_edge_set(patterns, edge_set):
    tmp_edge_dict = dict()
    tmp_extracted_knowledge = dict()
    for r in patterns:
        tmp_extracted_knowledge[r] = dict()
        for tmp_pattern in patterns[r]:
            tmp_extracted_knowledge[r][tmp_pattern[0]] = dict()
    for i, tmp_edge in enumerate(edge_set):
        parsed_eventuality1_words = list()
        doc = nlp(tmp_edge['event_1_words'])
        event1_dependency_edges = list()
        event1_verb = []
        for word in doc:
            event1_dependency_edges.append(((word.head.norm_, word.head.i), word.dep_, (word.norm_, word.i)))
            parsed_eventuality1_words.append(word.text)
            if word.dep_ == 'ROOT':
                event1_verb = (word.norm_, word.i)

        doc = nlp(tmp_edge['event_2_words'])
        event2_dependency_edges = list()
        event2_verb = []
        for word in doc:
            event2_dependency_edges.append(((word.head.norm_, word.head.i + len(parsed_eventuality1_words)), word.dep_,
                                            (word.norm_, word.i + len(parsed_eventuality1_words))))
            if word.dep_ == 'ROOT':
                event2_verb = (word.norm_, word.i + len(parsed_eventuality1_words))
        all_dependency_edges = event1_dependency_edges + event2_dependency_edges
        all_dependency_edges.append((event1_verb, tmp_edge['connective'], event2_verb))
        for r in patterns:
            for pattern in patterns[r]:
                tmp_knowledge_list = extract_knowledge_from_graph_with_knowledge(all_dependency_edges, pattern[0])
                for tmp_knowledge in tmp_knowledge_list:
                    if tmp_knowledge not in tmp_extracted_knowledge[r][pattern[0]]:
                        tmp_extracted_knowledge[r][pattern[0]][tmp_knowledge] = 0
                    tmp_extracted_knowledge[r][pattern[0]][tmp_knowledge] += tmp_edge['frequency']
                    if tmp_knowledge not in tmp_edge_dict:
                        tmp_edge_dict[tmp_knowledge] = list()
                    tmp_edge['graph'] = all_dependency_edges
                    tmp_edge_dict[tmp_knowledge].append(tmp_edge)
        if i % 1000 == 0:
            print('finished:', i, '/', len(edge_set))
    return tmp_extracted_knowledge, tmp_edge_dict


def edge_to_graph(tmp_edge):
    parsed_eventuality1_words = list()
    doc = nlp(tmp_edge['event_1_words'])
    event1_dependency_edges = list()
    event1_verb = []
    for word in doc:
        event1_dependency_edges.append(((word.head.norm_, word.head.i), word.dep_, (word.norm_, word.i)))
        parsed_eventuality1_words.append(word.text)
        if word.dep_ == 'ROOT':
            event1_verb = (word.norm_, word.i)

    doc = nlp(tmp_edge['event_2_words'])
    event2_dependency_edges = list()
    event2_verb = []
    for word in doc:
        event2_dependency_edges.append(((word.head.norm_, word.head.i + len(parsed_eventuality1_words)), word.dep_,
                                        (word.norm_, word.i + len(parsed_eventuality1_words))))
        if word.dep_ == 'ROOT':
            event2_verb = (word.norm_, word.i + len(parsed_eventuality1_words))
    all_dependency_edges = event1_dependency_edges + event2_dependency_edges
    all_dependency_edges.append((event1_verb, tmp_edge['connective'], event2_verb))
    return all_dependency_edges


def merge_extracted_knowledge_from_multi_core(all_extracted_knowledge):
    merged_knowledge = dict()
    for r in selected_patterns:
        merged_knowledge[r] = dict()
        for tmp_pattern in selected_patterns[r]:
            merged_knowledge[r][tmp_pattern[0]] = dict()
    for tmp_extracted_knowledge in tqdm(all_extracted_knowledge):
        for r in tmp_extracted_knowledge:
            for tmp_pattern in tmp_extracted_knowledge[r]:
                for tmp_k in tmp_extracted_knowledge[r][tmp_pattern]:
                    if tmp_k not in merged_knowledge[r][tmp_pattern]:
                        merged_knowledge[r][tmp_pattern][tmp_k] = tmp_extracted_knowledge[r][tmp_pattern][tmp_k]
                    else:
                        merged_knowledge[r][tmp_pattern][tmp_k] += tmp_extracted_knowledge[r][tmp_pattern][tmp_k]
    return merged_knowledge

nlp = spacy.load('en_core_web_sm')

try:
    with open('selected_patterns.json', 'r') as f:
        selected_patterns = json.load(f)
        print('Finish loading the patterns')
except:
    pass

Connectives = ['Precedence', 'Succession', 'Synchronous', 'Reason', 'Result', 'Condition', 'Contrast', 'Concession', 'Conjunction', 'Instantiation', 'Restatement', 'ChosenAlternative', 'Alternative', 'Exception']

