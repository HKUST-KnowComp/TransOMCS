from util import *

KG_path = 'KG_v0.1.0.db'

st = time.time()
kg_conn = KG_Connection(db_path=KG_path, mode='memory')
print('Finished in {:.2f} seconds'.format(time.time() - st))

print('We are collecting eventualities from ASER...')
selected_eventuality_kg = list()
eventuality_id_to_graph = dict()

for tmp_key, tmp_eventuality in tqdm(kg_conn.event_cache.items()):
    tmp = dict()
    tmp['type'] = 'eventuality'
    tmp['id'] = tmp_eventuality['_id']
    tmp['verbs'] = tmp_eventuality['verbs']
    tmp['words'] = tmp_eventuality['words']
    tmp['frequency'] = tmp_eventuality['frequency']
    selected_eventuality_kg.append(tmp)
    eventuality_id_to_graph[tmp['id']] = eventuality_to_graph(tmp)

print('We are collecting edges from ASER...')

selected_edge_kg = list()
edge_id_to_graph = dict()
for tmp_key, tmp_edge in tqdm(kg_conn.relation_cache.items()):
    tmp = dict()
    event_1 = kg_conn.event_cache[tmp_edge['event1_id']]
    event_2 = kg_conn.event_cache[tmp_edge['event2_id']]
    tmp['id'] = tmp_edge['_id']
    tmp['type'] = 'edge'
    tmp['event_1_verbs'] = event_1['verbs']
    tmp['event_1_words'] = event_1['words']
    tmp['event_2_verbs'] = event_2['verbs']
    tmp['event_2_words'] = event_2['words']
    tmp['frequency'] = tmp_edge['Co_Occurrence']
    tmp['connective'] = 'Co_Occurrence'
    for tmp_connective in Connectives:
        if tmp_edge[tmp_connective] > 0:
            tmp['frequency'] = tmp_edge[tmp_connective]
            tmp['connective'] = tmp_connective
            selected_edge_kg.append(tmp)
    edge_id_to_graph[tmp['id']] = edge_to_graph(tmp)

number_of_worker = 35
print('Start to collect knowledge from eventualities')
chunked_ASER = chunks(selected_eventuality_kg, number_of_worker)
workers = Pool(number_of_worker)
all_extracted_knowledge = list()
all_eventuality_match = list()
all_results = list()
for aser_subset in chunked_ASER:
    tmp_result = workers.apply_async(extract_knowledge_from_eventuality_set, args=(selected_patterns, aser_subset,))
    all_results.append(tmp_result)
workers.close()
workers.join()
all_results = [tmp_result.get() for tmp_result in all_results]
for tmp_result in all_results:
    all_extracted_knowledge.append(tmp_result[0])
    all_eventuality_match.append(tmp_result[1])

print('Start to merge eventuality knowledge')
extracted_eventuality_knowledge = merge_extracted_knowledge_from_multi_core(all_extracted_knowledge)

print('Start to merge eventuality matches')
merged_eventuality_match = dict()
for tmp_match_list in all_eventuality_match:
    for tmp_k in tmp_match_list:
        if tmp_k not in merged_eventuality_match:
            merged_eventuality_match[tmp_k] = list()
        merged_eventuality_match[tmp_k] += tmp_match_list[tmp_k]

chunked_ASER = chunks(selected_edge_kg, number_of_worker)
workers = Pool(number_of_worker)
all_extracted_knowledge = list()
all_edge_match = list()
all_results = list()
for aser_subset in chunked_ASER:
    tmp_result = workers.apply_async(extract_knowledge_from_edge_set, args=(selected_patterns, aser_subset,))
    all_results.append(tmp_result)
workers.close()
workers.join()
all_results = [tmp_result.get() for tmp_result in all_results]
for tmp_result in all_results:
    all_extracted_knowledge.append(tmp_result[0])
    all_edge_match.append(tmp_result[1])

print('Start to merge edge knowledge')
extracted_edge_knowledge = merge_extracted_knowledge_from_multi_core(all_extracted_knowledge)

print('Start to merge edge matches')
merged_edge_match = dict()
for tmp_match_list in all_edge_match:
    for tmp_k in tmp_match_list:
        if tmp_k not in merged_edge_match:
            merged_edge_match[tmp_k] = list()
        merged_edge_match[tmp_k] += tmp_match_list[tmp_k]

print('We are loading all words...')
all_words = list()
with open('words_alpha.txt', 'r') as f:
    for line in f:
        all_words.append(line[:-1])

all_words = set(all_words)

print('start to merge knowledge...')
extracted_knowledge = dict()
for r in extracted_eventuality_knowledge:
    extracted_knowledge[r] = dict()
    for p in extracted_eventuality_knowledge[r]:
        for tmp_triplet in extracted_eventuality_knowledge[r][p]:
            if tmp_triplet in extracted_knowledge[r]:
                extracted_knowledge[r][tmp_triplet] += extracted_eventuality_knowledge[r][p][tmp_triplet]
            else:
                extracted_knowledge[r][tmp_triplet] = extracted_eventuality_knowledge[r][p][tmp_triplet]
    for p in extracted_edge_knowledge[r]:
        for tmp_triplet in extracted_edge_knowledge[r][p]:
            if tmp_triplet in extracted_knowledge[r]:
                extracted_knowledge[r][tmp_triplet] += extracted_edge_knowledge[r][p][tmp_triplet]
            else:
                extracted_knowledge[r][tmp_triplet] = extracted_edge_knowledge[r][p][tmp_triplet]

filtered_knowledge = dict()
# Check if all the extracted words are English words to filter out other languages.
for r in extracted_knowledge:
    print('We are filtering knowledge for relation:', r)
    filtered_knowledge[r] = dict()
    for tmp_k in tqdm(extracted_knowledge[r]):
        head_words = tmp_k.split('$$')[0].split(' ')
        tail_words = tmp_k.split('$$')[1].split(' ')
        found_invalid_words = False
        for w in head_words:
            if w not in all_words:
                found_invalid_words = True
        for w in tail_words:
            if w not in all_words:
                found_invalid_words = True
        if found_invalid_words:
            continue
        filtered_knowledge[r][tmp_k] = extracted_knowledge[r][tmp_k]


# Store extracted knowledge based on relations for next step plausibility prediction
if not os.path.isdir('extracted_knowledge'):
    os.mkdir('extracted_knowledge')

# full_dataset = dict()
missing_count = 0
matched_count = 0
for r in filtered_knowledge:
    print('We are working on relation:', r)
    tmp_dataset = list()
    for tmp_k in tqdm(filtered_knowledge[r]):
        tmp_example = dict()
        if tmp_k in merged_eventuality_match:
            eventuality_observations = list()
            for tmp_eventuality in merged_eventuality_match[tmp_k]:
                tmp_graph = eventuality_id_to_graph[tmp_eventuality['id']]
                tmp_eventuality['graph'] = tmp_graph
                eventuality_observations.append(tmp_eventuality)
        else:
            eventuality_observations = list()
        if tmp_k in merged_edge_match:
            edge_observations = list()
            for tmp_edge in merged_edge_match[tmp_k]:
                tmp_graph = edge_id_to_graph[tmp_edge['id']]
                tmp_edge['graph'] = tmp_graph
                edge_observations.append(tmp_edge)
        else:
            edge_observations = list()
        tmp_example['knowledge'] = tmp_k
        tmp_example['eventuality_observations'] = eventuality_observations
        tmp_example['edge_observations'] = edge_observations
        tmp_example['plausibility'] = 0
        if len(eventuality_observations) == 0 and len(edge_observations) == 0:
            missing_count += 1
            continue
        else:
            matched_count += 1
            tmp_dataset.append(tmp_example)
    with open('extracted_knowledge/'+r+'.json', 'w') as f:
        json.dump(tmp_dataset, f)

print('end')
