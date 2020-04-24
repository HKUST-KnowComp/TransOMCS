from util import *
import torch
from pytorch_transformers import *
import logging
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import math
import numpy

Connective_dict = {'Precedence': 'before', 'Succession': 'after', 'Synchronous': 'simultaneously', 'Reason': 'because',
                   'Result': 'so', 'Condition': 'if', 'Contrast': 'but', 'Concession': 'although',
                   'Conjunction': 'and', 'Instantiation': 'for example', 'Restatement': 'in other words',
                   'Alternative': 'or', 'ChosenAlternative': 'instead', 'Exception': 'except'}

all_relations = ['AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'CreatedBy', 'DefinedAs', 'Desires', 'HasA',
                 'HasPrerequisite', 'HasProperty', 'HasSubevent', 'HasFirstSubevent', 'HasLastSubevent', 'InstanceOf',
                 'MadeOf', 'MotivatedByGoal', 'PartOf', 'ReceivesAction', 'UsedFor']


def get_adj_matrix(tmp_tokenized_sentence, tmp_graph, max_length):
    raw_matrix = numpy.zeros((max_length, max_length))
    for e in tmp_graph:
        head_word_index = tokenizer.encode(e[0][0])[0]
        tail_word_index = tokenizer.encode(e[2][0])[0]
        if head_word_index in tmp_tokenized_sentence and tail_word_index in tmp_tokenized_sentence:
            raw_matrix[tmp_tokenized_sentence.index(head_word_index)][
                tmp_tokenized_sentence.index(tail_word_index)] = 1.0
    for tmp_p in range(len(tmp_tokenized_sentence)):
        raw_matrix[tmp_p][tmp_p] = 1.0
    return torch.tensor(raw_matrix)


class TrainingExample:
    def __init__(self, raw_example):
        self.knowledge = raw_example['knowledge']
        self.eventuality_observation = raw_example['eventuality_observations']
        self.edge_observation = raw_example['edge_observations']
        if raw_example['label'] == 'positive':
            self.label = torch.tensor([1]).to(device)
        else:
            self.label = torch.tensor([0]).to(device)
        self.all_observations, self.all_head_masks, self.all_tail_masks, self.all_frequencies, self.all_types, self.all_adj_matrices = self.tensorize_training_example()

    def tensorize_training_example(self):
        all_observations = list()
        all_head_masks = list()
        all_tail_masks = list()
        all_frequencies = list()  # number of frequency and type of graph
        all_types = list()
        all_adj_matrices = list()
        head_words = tokenizer.encode(self.knowledge.split('$$')[0])
        tail_words = tokenizer.encode(self.knowledge.split('$$')[1])

        max_observation_length = 0
        for tmp_observation in self.eventuality_observation:
            tmp_sentence = tokenizer.encode('[CLS] ' + tmp_observation['words'] + ' . [SEP]')
            if len(tmp_sentence) > max_observation_length:
                max_observation_length = len(tmp_sentence)

        for tmp_observation in self.edge_observation:
            tmp_sentence = tokenizer.encode('[CLS] ' + tmp_observation['event_1_words'] + ' . [SEP] ' + Connective_dict[
                tmp_observation['connective']] + ' ' + tmp_observation['event_2_words'] + ' . [SEP] ')
            if len(tmp_sentence) > max_observation_length:
                max_observation_length = len(tmp_sentence)

        for tmp_observation in self.eventuality_observation:
            tmp_sentence = tokenizer.encode('[CLS] ' + tmp_observation['words'] + ' . [SEP]')
            tmp_head_mask = list()
            for w in tmp_sentence:
                if w in head_words:
                    tmp_head_mask.append(1.0)
                else:
                    tmp_head_mask.append(0.0)
            tmp_tail_mask = list()
            for w in tmp_sentence:
                if w in tail_words:
                    tmp_tail_mask.append(1.0)
                else:
                    tmp_tail_mask.append(0.0)
            if tmp_observation['frequency'] > 64:
                tmp_frequency = [6]
            else:
                tmp_frequency = [int(math.log(tmp_observation['frequency'], 2))]
            all_observations.append(torch.tensor(tmp_sentence))
            all_head_masks.append(torch.tensor(tmp_head_mask))
            all_tail_masks.append(torch.tensor(tmp_tail_mask))
            all_frequencies.append(torch.tensor(tmp_frequency))
            all_types.append(torch.tensor([0]))
            all_adj_matrices.append(get_adj_matrix(tmp_sentence, tmp_observation['graph'], max_observation_length))
        for tmp_observation in self.edge_observation:
            tmp_sentence = tokenizer.encode('[CLS] ' + tmp_observation['event_1_words'] + ' . [SEP] ' + Connective_dict[
                tmp_observation['connective']] + ' ' + tmp_observation['event_2_words'] + ' . [SEP] ')
            tmp_head_mask = list()
            for w in tmp_sentence:
                if w in head_words:
                    tmp_head_mask.append(1.0)
                else:
                    tmp_head_mask.append(0.0)
            tmp_tail_mask = list()
            for w in tmp_sentence:
                if w in tail_words:
                    tmp_tail_mask.append(1.0)
                else:
                    tmp_tail_mask.append(0.0)
            if tmp_observation['frequency'] > 64:
                tmp_frequency = [6]
            else:
                tmp_frequency = [int(math.log(tmp_observation['frequency'], 2))]
            all_observations.append(torch.tensor(tmp_sentence))
            all_head_masks.append(torch.tensor(tmp_head_mask))
            all_tail_masks.append(torch.tensor(tmp_tail_mask))
            all_frequencies.append(torch.tensor(tmp_frequency))
            all_types.append(torch.tensor([1]))
            all_adj_matrices.append(get_adj_matrix(tmp_sentence, tmp_observation['graph'], max_observation_length))

        tensorized_all_observations = pad_sequence(all_observations, batch_first=True).to(device)
        tensorized_head_masks = pad_sequence(all_head_masks, batch_first=True).to(device)
        tensorized_tail_masks = pad_sequence(all_tail_masks, batch_first=True).to(device)
        tensorized_frequencies = pad_sequence(all_frequencies, batch_first=True).to(device)
        tensorized_types = pad_sequence(all_types, batch_first=True).to(device)
        tensorized_adj_matrices = pad_sequence(all_adj_matrices, batch_first=True).to(device)

        return tensorized_all_observations, tensorized_head_masks, tensorized_tail_masks, tensorized_frequencies, tensorized_types, tensorized_adj_matrices


class DataLoader:
    def __init__(self, data_path, relation_name):
        with open(data_path, 'r') as f:
            raw_dataset = json.load(f)
        self.train_set = raw_dataset[relation_name]['train']
        self.test_set = raw_dataset[relation_name]['test']
        random.shuffle(self.train_set)
        random.shuffle(self.test_set)
        print('Start to tensorize the train set.')
        self.tensorized_train = self.tensorize_dataset(self.train_set)
        print('Start to tensorize the test set.')
        self.tensorized_test = self.tensorize_dataset(self.test_set)

    def random_sample_train_set(self):
        print('We are randomly selecting the train example')
        positive_dataset = list()
        negative_dataset = list()
        for tmp_example in self.train_set:
            if tmp_example['label'] == 'positive':
                positive_dataset.append(tmp_example)
            else:
                negative_dataset.append(tmp_example)
        random.shuffle(positive_dataset)
        random.shuffle(negative_dataset)
        if len(positive_dataset) > len(negative_dataset):
            new_dataset = positive_dataset[:len(negative_dataset)] + negative_dataset
        else:
            new_dataset = positive_dataset + negative_dataset[:len(positive_dataset)]
        random.shuffle(new_dataset)
        self.tensorized_train = self.tensorize_dataset(new_dataset)

    def tensorize_dataset(self, input_dataset):
        tmp_tensorized_dataset = list()
        positive_count = 0
        negative_count = 0
        for tmp_example in tqdm(input_dataset):
            tmp_tensorized_dataset.append(TrainingExample(tmp_example))
            if tmp_example['label'] == 'positive':
                positive_count += 1
            else:
                negative_count += 1
        print('Positive count:', positive_count, 'Negative count:', negative_count)
        return tmp_tensorized_dataset


class DataLoaderPredict:
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            raw_dataset = json.load(f)
        self.test_set = list()
        for tmp_example in tqdm(raw_dataset):
            new_example = tmp_example
            new_example['label'] = 'na'
            new_eventuality_observations = list()
            new_edge_observations = list()

            for tmp_eventuality in tmp_example['eventuality_observations']:
                tmp_sentence = '[CLS] ' + tmp_eventuality['words'] + ' . [SEP]'
                if len(tmp_sentence.split(' ')) < 30 and len(tokenizer.encode(tmp_sentence)) < 64:
                    new_eventuality_observations.append(tmp_eventuality)
            for tmp_edge in tmp_example['edge_observations']:
                tmp_sentence = '[CLS] ' + tmp_edge['event_1_words'] + ' . [SEP] ' + Connective_dict[
                    tmp_edge['connective']] + ' ' + tmp_edge['event_2_words'] + ' . [SEP] '
                if len(tmp_sentence.split(' ')) < 30 and len(tokenizer.encode(tmp_sentence)) < 64:
                    new_edge_observations.append(tmp_edge)
            new_example['eventuality_observations'] = new_eventuality_observations
            new_example['edge_observations'] = new_edge_observations
            if len(new_example['eventuality_observations']) + len(new_example['edge_observations']) > 0:
                self.test_set.append(new_example)
        print('number of new examples:', len(self.test_set))
        self.trunked_test_sets = list()

        self.number_of_trunks = int(len(self.test_set) / 10000) + 1
        print('Number of trunks:', self.number_of_trunks)
        for i in range(self.number_of_trunks):
            self.trunked_test_sets.append(self.test_set[i * 10000:(i + 1) * 10000])

    def tensorize_dataset(self, input_dataset):
        print('Start to tensorize the set.')
        tmp_tensorized_dataset = list()
        for tmp_example in tqdm(input_dataset):
            tmp_tensorized_dataset.append(TrainingExample(tmp_example))
        return tmp_tensorized_dataset


class CommonsenseRelationClassifier(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.bert = BertModel(config)
        self.edge_attention_weight = torch.nn.Linear(768 * 2, 1)
        self.last_layer = torch.nn.Linear(768 * 6, 2)
        self.frequency_embedding = torch.nn.Embedding(7, 768)
        self.type_embedding = torch.nn.Embedding(2, 768)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, raw_sentences, first_mask=None, second_mask=None, all_frequencies=None, all_types=None,
                adj_matrices=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        number_of_observation = raw_sentences.size(0)
        number_of_token = raw_sentences.size(1)
        encoding_after_bert = self.bert(raw_sentences)
        bert_last_layer = encoding_after_bert[0]  # [number_of_observation, number_of_token, embedding_size]

        # start to implement the graph attention
        last_layer_pile = bert_last_layer.repeat(1, number_of_token, 1).view(
            [number_of_observation, number_of_token, number_of_token,
             768])  # [number_of_observation, number_of_token, number_of_token, embedding_size]
        last_layer_repeat = bert_last_layer.repeat(1, 1, number_of_token).view(
            [number_of_observation, number_of_token, number_of_token,
             768])  # [number_of_observation, number_of_token, number_of_token, embedding_size]
        matched_last_layer = torch.cat([last_layer_pile, last_layer_repeat],
                                       dim=3)  # [number_of_observation, number_of_token, number_of_token, embedding_size*2]
        attention_weight = self.edge_attention_weight(matched_last_layer).squeeze(
            dim=3)  # [number_of_observation, number_of_token, number_of_token]
        adj_matrices = adj_matrices.float()
        attention_weight = attention_weight * adj_matrices  # [number_of_observation, number_of_token, number_of_token]
        weight_after_softmax = F.softmax(attention_weight, dim=2).unsqueeze(
            3)  # [number_of_observation, number_of_token, number_of_token, 1]
        weight_after_softmax_matrices = weight_after_softmax.repeat(1, 1, 1,
                                                                    768)  # [number_of_observation, number_of_token, number_of_token, embedding_size]
        aggegated_embedding = torch.sum(last_layer_pile * weight_after_softmax_matrices,
                                        dim=2)  # [number_of_observation, number_of_token, embedding_size]

        aggegated_embedding = torch.cat([aggegated_embedding, bert_last_layer], dim=2)

        # Start to implement the head/tail mask
        first_mask = first_mask[:, :, None]
        second_mask = second_mask[:, :, None]
        head_selection_mask = first_mask.expand(
            [-1, -1, 768 * 2])  # [number_of_observation, number_of_token, embedding_size]
        head_representation = torch.mean(aggegated_embedding * head_selection_mask,
                                         dim=1)  # [number_of_observation, embedding_size]
        tail_selection_mask = second_mask.expand(
            [-1, -1, 768 * 2])  # [number_of_observation, number_of_token, embedding_size]
        tail_representation = torch.mean(aggegated_embedding * tail_selection_mask,
                                         dim=1)  # [number_of_observation, embedding_size]

        # Start to add features
        frequency_feature = self.frequency_embedding(all_frequencies).squeeze(
            dim=1)  # [number_of_observation, embedding_size]
        type_feature = self.type_embedding(all_types).squeeze(dim=1)  # [number_of_observation, embedding_size]
        overall_representation = torch.cat(
            [head_representation, tail_representation, frequency_feature, type_feature],
            dim=1)  # [number_of_observation, embedding_size*2]

        overall_representation = self.dropout(overall_representation)
        final_prediction = self.last_layer(overall_representation)  # [batch_size, 2]

        final_prediction = torch.mean(final_prediction, dim=0).unsqueeze(0)  # [1, 2]
        return final_prediction


def train(model, train_data):
    all_loss = 0
    print('training:')
    random.shuffle(train_data)
    model.train()
    for tmp_example in tqdm(train_data):
        final_prediction = model(raw_sentences=tmp_example.all_observations, first_mask=tmp_example.all_head_masks,
                                 second_mask=tmp_example.all_tail_masks, all_frequencies=tmp_example.all_frequencies,
                                 all_types=tmp_example.all_types, adj_matrices=tmp_example.all_adj_matrices)  # 1 * 2
        loss = loss_func(final_prediction, tmp_example.label)
        test_optimizer.zero_grad()
        loss.backward()
        test_optimizer.step()
        all_loss += loss.item()
    print('current loss:', all_loss / len(current_data.tensorized_train))


def test(model, test_data):
    correct_count = 0
    print('Testing')
    model.eval()
    for tmp_example in tqdm(test_data):
        final_prediction = model(raw_sentences=tmp_example.all_observations, first_mask=tmp_example.all_head_masks,
                                 second_mask=tmp_example.all_tail_masks, all_frequencies=tmp_example.all_frequencies,
                                 all_types=tmp_example.all_types, adj_matrices=tmp_example.all_adj_matrices)  # 1 * 2
        if tmp_example.label.data[0] == 1:
            # current example is positive
            if final_prediction.data[0][1] >= final_prediction.data[0][0]:
                correct_count += 1
        else:
            # current example is negative
            if final_prediction.data[0][1] <= final_prediction.data[0][0]:
                correct_count += 1

    print('current accuracy:', correct_count, '/', len(test_data), correct_count / len(test_data))
    return correct_count / len(test_data)


def predict(model, data_for_predict, relation):
    model.eval()
    tmp_prediction_dict = dict()
    print('Start to predict')
    for tmp_example in tqdm(data_for_predict):
        final_prediction = model(raw_sentences=tmp_example.all_observations, first_mask=tmp_example.all_head_masks,
                                 second_mask=tmp_example.all_tail_masks, all_frequencies=tmp_example.all_frequencies,
                                 all_types=tmp_example.all_types, adj_matrices=tmp_example.all_adj_matrices)  # 1 * 2
        scores = F.softmax(final_prediction, dim=1)
        tmp_prediction_dict[tmp_example.knowledge] = scores.data.tolist()[0][1]
    tmp_file_name = 'prediction/' + relation + '.json'
    try:
        with open(tmp_file_name, 'r') as f:
            prediction_dict = json.load(f)
    except FileNotFoundError:
        prediction_dict = dict()
    for tmp_k in tmp_prediction_dict:
        prediction_dict[tmp_k] = tmp_prediction_dict[tmp_k]
    with open(tmp_file_name, 'w') as f:
        json.dump(prediction_dict, f)


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='graph', type=str, required=False,
                    help="choose the model to test")
parser.add_argument("--lr", default=0.001, type=float, required=False,
                    help="initial learning rate")
parser.add_argument("--lrdecay", default=0.8, type=float, required=False,
                    help="learning rate decay every 5 epochs")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device:', device)
n_gpu = torch.cuda.device_count()
print('number of gpu:', n_gpu)
torch.cuda.get_device_name(0)

current_model = CommonsenseRelationClassifier.from_pretrained('bert-base-uncased')
test_optimizer = torch.optim.SGD(current_model.parameters(), lr=args.lr)
loss_func = torch.nn.CrossEntropyLoss()
current_model.to(device)

performance_dict = dict()

selected_relations = all_relations

if not os.path.isdir('models'):
    os.mkdir('models')

for r in selected_relations:
    current_data = DataLoader('ranking_dataset.json', r)
    print('Finish loading data')

    test(current_model, current_data.tensorized_test)

    best_performance = 0
    tmp_lr = args.lr
    current_data.random_sample_train_set()
    for i in range(50):
        if i % 5 == 0:
            test_optimizer = torch.optim.SGD(current_model.parameters(), lr=tmp_lr)
            tmp_lr = tmp_lr * args.lrdecay
        print('Current Iteration:', i + 1, '|', 'Relation:', r, '|',
              'Current best performance:', best_performance)
        train(current_model, current_data.tensorized_train)
        tmp_performance = test(current_model, current_data.tensorized_test)
        if tmp_performance >= best_performance:
            best_performance = tmp_performance
            print('We are saving the new best model')
            torch.save(current_model.state_dict(), 'models/' + r + '.pth')
    performance_dict[r] = best_performance

if not os.path.isdir('prediction'):
    os.mkdir('prediction')
# This process might be slow due to the huge dataset scale.
for r in selected_relations:
    print('Start to load data...')
    data_for_prediction = DataLoaderPredict('extracted_knowledge/' + r + '.json')
    print('Finish loading data...')
    current_model = CommonsenseRelationClassifier.from_pretrained('bert-base-uncased')
    current_model.load_state_dict(torch.load('models/' + r + '.pth'))
    current_model.to(device)
    print('We are working on relation:', r)
    for i in range(data_for_prediction.number_of_trunks):
        print('Working on set:', i + 1, '/', data_for_prediction.number_of_trunks, 'relation:', r)
        tmp_tensorized_data = data_for_prediction.tensorize_dataset(data_for_prediction.trunked_test_sets[i])
        predict(current_model, tmp_tensorized_data, r)
        tmp_tensorized_data = list()

overall_dict = dict()
for r in all_relations:
    print('We are working on:', r)
    with open('prediction/' + r + '.json', 'r') as f:
        tmp_dict = json.load(f)
    for tmp_k in tqdm(tmp_dict):
        tmp_head = tmp_k.split('$$')[0]
        tmp_tail = tmp_k.split('$$')[1]
        new_k = tmp_head + '$$' + r + '$$' + tmp_tail
        overall_dict[new_k] = tmp_dict[tmp_k]

sorted_result = sorted(overall_dict, key=lambda x: overall_dict[x], reverse=True)
with open('prediction/TransOMCS.txt', 'w') as f:
    for tmp_k in sorted_result:
        f.write(tmp_k.split('$$')[0])
        f.write('\t')
        f.write(tmp_k.split('$$')[1])
        f.write('\t')
        f.write(tmp_k.split('$$')[2])
        f.write('\t')
        f.write(str(overall_dict[tmp_k]))
        f.write('\n')

print('end')
