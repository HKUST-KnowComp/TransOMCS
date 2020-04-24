import ujson as json
import spacy
from tqdm import tqdm
import math


def path_to_pattern(head_edges, between_edges, tail_edges):
    # we only focus on the between_edges first
    tmp_edges = list()
    pattern = ''
    current_word = 'HEAD'
    next_word = ''
    seen_positions = list()

    # we are working on the between_edges first
    while next_word != 'TAIL':
        new_word = ''
        for edge in between_edges:
            if edge[0][1] in seen_positions or edge[2][1] in seen_positions:
                continue
            if edge[0][0] == current_word:
                if edge[2][0] == 'TAIL':
                    pattern += '->'
                    pattern += edge[1]
                    pattern += '->'
                    next_word = 'TAIL'
                else:
                    pattern += '->'
                    pattern += edge[1]
                    pattern += '->'
                    pattern += edge[2][0]
                    # pattern += '->'
                    new_word = edge[2][0]
                    seen_positions.append(edge[0][1])
                break
            elif edge[2][0] == current_word:
                if edge[0][0] == 'TAIL':
                    pattern += '<-'
                    pattern += edge[1]
                    pattern += '<-'
                    next_word = 'TAIL'
                else:
                    pattern += '<-'
                    pattern += edge[1]
                    pattern += '<-'
                    pattern += edge[0][0]
                    # pattern += '<-'
                    new_word = edge[0][0]
                    seen_positions.append(edge[2][1])
                break
        current_word = new_word

    # we are working on the head_edges

    if len(head_edges) == 0:
        head_pattern = '()'
    else:
        head_pattern = '(-'
        for edge in head_edges:
            head_pattern += edge[1]
            head_pattern += '-'
        head_pattern += ')'

    # we are working on the tail edges
    if len(tail_edges) == 0:
        tail_pattern = '()'
    else:
        tail_pattern = '(-'
        for edge in tail_edges:
            tail_pattern += edge[1]
            tail_pattern += '-'
        tail_pattern += ')'

    overall_pattern = head_pattern + pattern + tail_pattern

    return overall_pattern


def find_shortest_path(all_edges, start, end, used_edges):
    potential_pathes = list()
    for edge in all_edges:
        if edge in used_edges:
            continue
        if edge[0][1] == start:
            if edge[2][1] == end:
                return [edge], 1
            else:
                potential_pathes.append({'edges': [edge], 'new_start': edge[2][1]})
                continue
        if edge[2][1] == start:
            if edge[0][1] == end:
                return [edge], 1
            else:
                potential_pathes.append({'edges': [edge], 'new_start': edge[0][1]})
    if len(potential_pathes) == 0:
        return [], 0
    shortest_path = list()
    shortest_length = 100
    for potential_path in potential_pathes:
        all_used_edges = used_edges + potential_path['edges']
        tmp_new_edges, tmp_new_length = find_shortest_path(all_edges, potential_path['new_start'], end, all_used_edges)
        if len(tmp_new_edges) > 0 and tmp_new_length < shortest_length:
            shortest_length = tmp_new_length
            shortest_path = tmp_new_edges + potential_path['edges']
    return shortest_path, shortest_length + 1


def extract_pattern(OMCS_pair, tmp_graph):
    head_words = OMCS_pair.split('$$')[0].split()
    tail_words = OMCS_pair.split('$$')[1].split()
    eventuality_words = tmp_graph['words'].split()
    # check repeat words
    for w in head_words:
        if w in tail_words:
            return None

    # locate position
    head_positions = list()
    tail_positions = list()
    for w in head_words:
        found_location = False
        for i, tmp_word in enumerate(eventuality_words):
            if found_location:
                if w == tmp_word:
                    return None
            else:
                if w == tmp_word:
                    head_positions.append(i)
                    found_location = True
    for w in tail_words:
        found_location = False
        for i, tmp_word in enumerate(eventuality_words):
            if found_location:
                if w == tmp_word:
                    return None
            else:
                if w == tmp_word:
                    tail_positions.append(i)
                    found_location = True

    doc = nlp(tmp_graph['words'])
    all_dependency_edges = list()
    for word in doc:
        all_dependency_edges.append(((word.head.norm_, word.head.i), word.dep_, (word.norm_, word.i)))

    head_dependency_edges = list()
    tail_dependency_edges = list()

    # find head internal edges:
    if len(head_positions) > 1:
        for position_1 in head_positions:
            for position_2 in head_positions:
                if position_1 < position_2:
                    paths, length = find_shortest_path(all_dependency_edges, position_1, position_2, list())
                    head_dependency_edges += paths
    head_dependency_edges = list(set(head_dependency_edges))

    # find tail internal edges
    if len(tail_positions) > 1:
        for position_1 in tail_positions:
            for position_2 in tail_positions:
                if position_1 < position_2:
                    paths, length = find_shortest_path(all_dependency_edges, position_1, position_2, list())
                    tail_dependency_edges += paths
    tail_dependency_edges = list(set(tail_dependency_edges))

    head_contained_positions = list()
    tail_contained_positions = list()
    if len(head_dependency_edges) == 0:
        head_contained_positions.append(head_positions[0])
    else:
        for d_edge in head_dependency_edges:
            head_contained_positions.append(d_edge[0][1])
            head_contained_positions.append(d_edge[2][1])
    if len(tail_dependency_edges) == 0:
        tail_contained_positions.append(tail_positions[0])
    else:
        for d_edge in tail_dependency_edges:
            tail_contained_positions.append(d_edge[0][1])
            tail_contained_positions.append(d_edge[2][1])

    # We need to check if there is overlap
    for position in head_contained_positions:
        if position in tail_contained_positions:
            return None

    new_edges = list()
    for d_edge in all_dependency_edges:
        if d_edge[0][1] in head_contained_positions:
            if d_edge[2][1] in head_contained_positions:
                continue
            elif d_edge[2][1] in tail_contained_positions:
                new_edges.append((('HEAD', 'HEAD'), d_edge[1], ('TAIL', 'TAIL')))
            else:
                new_edges.append((('HEAD', 'HEAD'), d_edge[1], d_edge[2]))
        elif d_edge[0][1] in tail_contained_positions:
            if d_edge[2][1] in head_contained_positions:
                new_edges.append((('TAIL', 'TAIL'), d_edge[1], ('HEAD', 'HEAD')))
            elif d_edge[2][1] in tail_contained_positions:
                continue
            else:
                new_edges.append((('TAIL', 'TAIL'), d_edge[1], d_edge[2]))
        else:
            if d_edge[2][1] in head_contained_positions:
                new_edges.append((d_edge[0], d_edge[1], ('HEAD', 'HEAD')))
            elif d_edge[2][1] in tail_contained_positions:
                new_edges.append((d_edge[0], d_edge[1], ('TAIL', 'TAIL')))
            else:
                new_edges.append((d_edge[0], d_edge[1], d_edge[2]))
    between_edges, _ = find_shortest_path(new_edges, 'HEAD', 'TAIL', list())

    # find shortest path between head and tail
    if len(between_edges) == 0:
        return None

    pattern = path_to_pattern(head_dependency_edges, between_edges, tail_dependency_edges)

    return pattern


def extract_pattern_from_edge(OMCS_pair, tmp_edge):
    head_words = OMCS_pair.split('$$')[0].split()
    tail_words = OMCS_pair.split('$$')[1].split()

    eventuality1_words = tmp_edge['event_1_words'].split()
    eventuality2_words = tmp_edge['event_2_words'].split()

    # check repeat words
    for w in head_words:
        if w in tail_words:
            return None

    head_in_event1 = True
    head_in_event2 = True
    tail_in_event1 = True
    tail_in_event2 = True
    for w in head_words:
        if w not in eventuality1_words:
            head_in_event1 = False
        if w not in eventuality2_words:
            head_in_event2 = False
    for w in tail_words:
        if w not in eventuality1_words:
            tail_in_event1 = False
        if w not in eventuality2_words:
            tail_in_event2 = False

    if (head_in_event1 and tail_in_event2 and not head_in_event2 and not tail_in_event1) or (
            head_in_event2 and tail_in_event1 and not head_in_event1 and not tail_in_event2):
        pass
    else:
        return None

    all_words = list()
    parsed_eventuality1_words = list()
    doc = nlp(tmp_edge['event_1_words'])
    event1_dependency_edges = list()
    event1_verb = []
    for word in doc:
        event1_dependency_edges.append(((word.head.norm_, word.head.i), word.dep_, (word.norm_, word.i)))
        all_words.append(word.text)
        parsed_eventuality1_words.append(word.text)
        if word.dep_ == 'ROOT':
            event1_verb = (word.norm_, word.i)

    doc = nlp(tmp_edge['event_2_words'])
    event2_dependency_edges = list()
    event2_verb = []
    for word in doc:
        event2_dependency_edges.append(((word.head.norm_, word.head.i + len(parsed_eventuality1_words)), word.dep_,
                                        (word.norm_, word.i + len(parsed_eventuality1_words))))
        all_words.append(word.text)
        if word.dep_ == 'ROOT':
            event2_verb = (word.norm_, word.i + len(parsed_eventuality1_words))

    head_dependency_edges = list()
    tail_dependency_edges = list()
    all_dependency_edges = event1_dependency_edges + event2_dependency_edges

    all_dependency_edges.append((event1_verb, tmp_edge['connective'], event2_verb))

    # locate position
    head_positions = list()
    tail_positions = list()
    for w in head_words:
        found_location = False
        for i, tmp_word in enumerate(all_words):
            if found_location:
                if w == tmp_word:
                    return None
            else:
                if w == tmp_word:
                    head_positions.append(i)
                    found_location = True
    for w in tail_words:
        found_location = False
        for i, tmp_word in enumerate(all_words):
            if found_location:
                if w == tmp_word:
                    return None
            else:
                if w == tmp_word:
                    tail_positions.append(i)
                    found_location = True

    if head_in_event1:
        # find head internal edges:
        if len(head_positions) > 1:
            for position_1 in head_positions:
                for position_2 in head_positions:
                    if position_1 < position_2:
                        paths, length = find_shortest_path(event1_dependency_edges, position_1, position_2, list())
                        head_dependency_edges += paths
        head_dependency_edges = list(set(head_dependency_edges))

        # find tail internal edges
        if len(tail_positions) > 1:
            for position_1 in tail_positions:
                for position_2 in tail_positions:
                    if position_1 < position_2:
                        paths, length = find_shortest_path(event2_dependency_edges, position_1, position_2, list())
                        tail_dependency_edges += paths
        tail_dependency_edges = list(set(tail_dependency_edges))
    else:
        # find head internal edges:
        if len(head_positions) > 1:
            for position_1 in head_positions:
                for position_2 in head_positions:
                    if position_1 < position_2:
                        paths, length = find_shortest_path(event2_dependency_edges, position_1, position_2, list())
                        head_dependency_edges += paths
        head_dependency_edges = list(set(head_dependency_edges))

        # find tail internal edges
        if len(tail_positions) > 1:
            for position_1 in tail_positions:
                for position_2 in tail_positions:
                    if position_1 < position_2:
                        paths, length = find_shortest_path(event1_dependency_edges, position_1, position_2, list())
                        tail_dependency_edges += paths
        tail_dependency_edges = list(set(tail_dependency_edges))

    head_contained_positions = list()
    tail_contained_positions = list()
    if len(head_dependency_edges) == 0:
        head_contained_positions.append(head_positions[0])
    else:
        for d_edge in head_dependency_edges:
            head_contained_positions.append(d_edge[0][1])
            head_contained_positions.append(d_edge[2][1])
    if len(tail_dependency_edges) == 0:
        tail_contained_positions.append(tail_positions[0])
    else:
        for d_edge in tail_dependency_edges:
            tail_contained_positions.append(d_edge[0][1])
            tail_contained_positions.append(d_edge[2][1])

    # We need to check if there is overlap
    for position in head_contained_positions:
        if position in tail_contained_positions:
            return None

    new_edges = list()
    for d_edge in all_dependency_edges:
        if d_edge[0][1] in head_contained_positions:
            if d_edge[2][1] in head_contained_positions:
                continue
            elif d_edge[2][1] in tail_contained_positions:
                new_edges.append((('HEAD', 'HEAD'), d_edge[1], ('TAIL', 'TAIL')))
            else:
                new_edges.append((('HEAD', 'HEAD'), d_edge[1], d_edge[2]))
        elif d_edge[0][1] in tail_contained_positions:
            if d_edge[2][1] in head_contained_positions:
                new_edges.append((('TAIL', 'TAIL'), d_edge[1], ('HEAD', 'HEAD')))
            elif d_edge[2][1] in tail_contained_positions:
                continue
            else:
                new_edges.append((('TAIL', 'TAIL'), d_edge[1], d_edge[2]))
        else:
            if d_edge[2][1] in head_contained_positions:
                new_edges.append((d_edge[0], d_edge[1], ('HEAD', 'HEAD')))
            elif d_edge[2][1] in tail_contained_positions:
                new_edges.append((d_edge[0], d_edge[1], ('TAIL', 'TAIL')))
            else:
                new_edges.append((d_edge[0], d_edge[1], d_edge[2]))
    between_edges, _ = find_shortest_path(new_edges, 'HEAD', 'TAIL', list())

    # find shortest path between head and tail
    if len(between_edges) == 0:
        return None

    pattern = path_to_pattern(head_dependency_edges, between_edges, tail_dependency_edges)

    return pattern


def get_unique_score(tmp_p, tmp_r, unique_dict):
    tmp_score = None
    for relation_pair in unique_dict[tmp_p]:
        if relation_pair[0] == tmp_r:
            tmp_score = relation_pair[1]
            break
    return tmp_score


def compute_length_score(tmp_pattern):
    head_pattern = tmp_pattern.split(')')[0][1:]
    internal_pattern = tmp_pattern.split(')')[1].split('(')[0]
    tail_pattern = tmp_pattern.split('(')[2][:-1]

    head_count = 0
    for w in head_pattern.split('-'):
        if w not in ['', '<', '>']:
            head_count += 1
    internal_count = 0
    for w in internal_pattern.split('-'):
        if w not in ['', '<', '>']:
            internal_count += 1
    tail_count = 0
    for w in tail_pattern.split('-'):
        if w not in ['', '<', '>']:
            tail_count += 1

    tmp_score = min(3, head_count + internal_count + tail_count)

    return tmp_score


def find_discourse_relation(tmp_pattern):
    tmp_status = False
    for discourse_r in discourse_relations:
        if discourse_r in tmp_pattern:
            tmp_status = True
            break
    return tmp_status


def check_pattern_stop_relations(stop_relations, pattern):
    for tmp_r in stop_relations:
        if tmp_r in pattern:
            return True
    return False


selected_relations = ['AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'CreatedBy', 'DefinedAs', 'Desires', 'HasA',
                      'HasPrerequisite', 'HasProperty', 'HasSubevent', 'HasFirstSubevent', 'HasLastSubevent',
                      'InstanceOf', 'LocatedNear', 'MadeOf', 'MotivatedByGoal', 'PartOf', 'ReceivesAction', 'UsedFor']

discourse_relations = ['Precedence', 'Succession', 'Synchronous', 'Reason', 'Result', 'Condition', 'Contrast',
                       'Concession', 'Conjunction', 'Instantiation', 'Restatement', 'Alternative', 'ChosenAlternative',
                       'Exception']

with open('node_matches.json', 'r') as f:
    sample_data = json.load(f)
nlp = spacy.load('en')

raw_eventuality_patterns = dict()

for tmp_r in sample_data:
    print('We are working on:', tmp_r)
    test_data = sample_data[tmp_r]
    pattern_counting = dict()
    for OMCS_pair in tqdm(test_data):
        if len(test_data[OMCS_pair]) == 0:
            continue
        for tmp_eventuality in test_data[OMCS_pair][:50]:
            pattern = extract_pattern(OMCS_pair, tmp_eventuality)
            if not pattern:
                continue
            if pattern not in pattern_counting:
                pattern_counting[pattern] = 0
            pattern_counting[pattern] += 1

    sorted_patterns = sorted(pattern_counting.items(), key=lambda x: x[1], reverse=True)
    selected_patterns = sorted_patterns
    raw_eventuality_patterns[tmp_r] = selected_patterns


with open('edge_matches.json', 'r') as f:
    sample_edge_data = json.load(f)
nlp = spacy.load('en')

raw_edge_patterns = dict()

for tmp_r in sample_edge_data:
    print('We are working on:', tmp_r)
    test_edge_data = sample_edge_data[tmp_r]
    pattern_counting = dict()
    for OMCS_pair in tqdm(test_edge_data):
        if len(test_edge_data[OMCS_pair]) == 0:
            continue
        selected_match_eventualities = test_edge_data[OMCS_pair][:50]
        for tmp_edge in selected_match_eventualities:
            pattern = extract_pattern_from_edge(OMCS_pair, tmp_edge)
            if not pattern:
                continue
            if pattern not in pattern_counting:
                pattern_counting[pattern] = 0
            pattern_counting[pattern] += 1

    sorted_patterns = sorted(pattern_counting.items(), key=lambda x: x[1], reverse=True)
    selected_patterns = sorted_patterns
    raw_edge_patterns[tmp_r] = selected_patterns

new_eventuality_patterns = dict()
seen_eventuality_patterns = dict()
for r in raw_eventuality_patterns:
    if r not in selected_relations:
        continue
    new_eventuality_patterns[r] = list()
    seen_eventuality_patterns[r] = list()
    for pattern in raw_eventuality_patterns[r]:
        no_direction_pattern = pattern[0].replace('>', '').replace('<', '')
        if no_direction_pattern in seen_eventuality_patterns[r]:
            continue
        seen_eventuality_patterns[r].append(no_direction_pattern)
        new_eventuality_patterns[r].append(pattern)

new_edge_patterns = dict()
seen_edge_patterns = dict()
for r in raw_edge_patterns:
    if r not in selected_relations:
        continue
    new_edge_patterns[r] = list()
    seen_edge_patterns[r] = list()
    for pattern in raw_edge_patterns[r]:
        no_direction_pattern = pattern[0].replace('>', '').replace('<', '')
        if no_direction_pattern in seen_edge_patterns[r]:
            continue
        seen_edge_patterns[r].append(no_direction_pattern)
        new_edge_patterns[r].append(pattern)

eventuality_patterns = new_eventuality_patterns
edge_patterns = new_edge_patterns


with open('lemmatized_commonsense_knowledge.json', 'r') as f:
    lemmatized_commonsense_knowledge = json.load(f)


all_eventualities_patterns_count = dict()
for r in eventuality_patterns:
    for p in eventuality_patterns[r]:
        if p[0] not in all_eventualities_patterns_count:
            all_eventualities_patterns_count[p[0]] = dict()
        all_eventualities_patterns_count[p[0]][r] = p[1] / math.sqrt(len(lemmatized_commonsense_knowledge[r]))

# prepare u_score
new_eventuality_pattern_count = dict()
for p in all_eventualities_patterns_count:
    sum_count = 0
    for r in all_eventualities_patterns_count[p]:
        sum_count += all_eventualities_patterns_count[p][r]
    new_tmp_count = list()
    for r in all_eventualities_patterns_count[p]:
        new_tmp_count.append((r, all_eventualities_patterns_count[p][r] / sum_count))
    sorted_tmp_count = sorted(new_tmp_count, key=lambda x: x[1], reverse=True)
    new_eventuality_pattern_count[p] = sorted_tmp_count

# p[1] is the counting (c_score)
eventuality_patterns_by_score = dict()
for r in eventuality_patterns:
    tmp_patterns = list()
    for p in eventuality_patterns[r]:
        u_score = get_unique_score(p[0], r, new_eventuality_pattern_count)
        l_score = compute_length_score(p[0])
        if u_score:
            tmp_patterns.append((p[0], p[1] * l_score * u_score))
    eventuality_patterns_by_score[r] = sorted(tmp_patterns, key=lambda x: x[1], reverse=True)

all_edge_patterns_count = dict()
for r in edge_patterns:
    for p in edge_patterns[r]:
        if p[0] not in all_edge_patterns_count:
            all_edge_patterns_count[p[0]] = dict()
        all_edge_patterns_count[p[0]][r] = p[1] / math.sqrt(len(lemmatized_commonsense_knowledge[r]))

# prepare u_score
new_edge_pattern_count = dict()
for p in all_edge_patterns_count:
    sum_count = 0
    for r in all_edge_patterns_count[p]:
        sum_count += all_edge_patterns_count[p][r]
    new_tmp_count = list()
    for r in all_edge_patterns_count[p]:
        new_tmp_count.append((r, all_edge_patterns_count[p][r] / sum_count))
    sorted_tmp_count = sorted(new_tmp_count, key=lambda x: x[1], reverse=True)
    new_edge_pattern_count[p] = sorted_tmp_count

# p[1] is the counting (c_score)
edge_patterns_by_score = dict()
for r in edge_patterns:
    tmp_patterns = list()
    for p in edge_patterns[r]:
        u_score = get_unique_score(p[0], r, new_edge_pattern_count)
        l_score = compute_length_score(p[0])
        if u_score:
            tmp_patterns.append((p[0], p[1] * l_score * u_score))
    edge_patterns_by_score[r] = sorted(tmp_patterns, key=lambda x: x[1], reverse=True)

# Merge extracted patterns from eventuality and edge
overall_pattern_by_score = dict()
for r in edge_patterns_by_score:
    tmp_patterns = list()
    overall_score = 0
    for pattern in eventuality_patterns_by_score[r]:
        tmp_patterns.append(pattern)
        overall_score += pattern[1]
    for pattern in edge_patterns_by_score[r]:
        tmp_patterns.append((pattern[0], pattern[1]))
        overall_score += pattern[1]
    tmp_patterns = sorted(tmp_patterns, key=lambda x: x[1], reverse=True)
    overall_pattern_by_score[r] = list()
    for pattern in tmp_patterns:
        overall_pattern_by_score[r].append((pattern[0], pattern[1] / overall_score))

# setup the linguistic relation we do not want in our pattern, which is like the stop words filtering.
pattern_stop_relations = ['det']
threshold = 0.05
selected_patterns = dict()
for r in overall_pattern_by_score:
    tmp_selected_pattern = list()
    for pattern in overall_pattern_by_score[r]:
        if pattern[1] > threshold and not check_pattern_stop_relations(pattern_stop_relations, pattern[0]):
            tmp_selected_pattern.append(pattern)
    selected_patterns[r] = tmp_selected_pattern

with open('selected_patterns.json', 'w') as f:
    json.dump(selected_patterns, f)

print('end')
