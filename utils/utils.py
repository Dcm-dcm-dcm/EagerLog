import numpy as np
import random
import torch
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os
from .preprocessing import parsing


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def levenshtein_distance(s1, s2, threshold):
    m, n = len(s1), len(s2)

    if m < n:
        return levenshtein_distance(s2, s1, threshold)

    previous_row = list(range(n + 1))

    for i in range(1, m + 1):
        current_row = [i] + [0] * n
        for j in range(1, n + 1):
            insert = previous_row[j] + 1
            delete = current_row[j - 1] + 1
            replace = previous_row[j - 1] + (s1[i - 1] != s2[j - 1])
            current_row[j] = min(insert, delete, replace)

        if min(current_row) > threshold:
            return float('inf')

        previous_row = current_row

    return previous_row[n]


def join_strings_with_tabs_and_newline(strings, n):
    new_string = []
    for str in strings:
        if not str.endswith('\n'):
            str += '\n'
        new_string.append(str)
    strings = new_string
    if not strings or n < 0:
        return ""

    tab = "\t" * n
    result = tab.join(strings) + "\n"
    return result


def get_label(dataset, label):
    if dataset == 'BGL':
        return max(label)
    elif dataset == 'Zookeeper':
        return max(label)
    elif dataset == 'Thunderbird':
        return max(label)
    else:
        print(f'dataset:{dataset} not support!')
        return 0


def parse_label(label_text):
    pattern = '\[([^\]]+)\]'
    match = re.search(pattern, label_text)
    try:
        state = match.group(1)
        label = 0 if state == 'Normal' else 1
        return label, state
    except:
        # Case conversion
        label_text = label_text.lower()
        normal_pos = label_text.find('normal')
        abnormal_pos = label_text.find('abnormal')
        if normal_pos != -1 and abnormal_pos != -1:
            return ((0, 'Normal') if normal_pos < abnormal_pos else (1, 'Abnormal'))
        elif normal_pos == -1 and abnormal_pos == -1:
            return 2, ''
        else:
            return ((0, 'Normal') if normal_pos != -1 else (1, 'Normal'))


def get_uncertainty(label_text):
    pattern = r'Uncertainty:\s*([\d\.]+)'
    match = re.search(pattern, label_text)
    try:
        score = match.group(1)
        return score
    except:
        return 0


def save_prompt(file_path, prompt):
    with open(file_path, 'w') as file:
        file.write(prompt)


def get_dpr_model_and_tokenizer():
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
    import torch
    model_name = "facebook/dpr-ctx_encoder-multiset-base"
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.cuda()
    return model, tokenizer


def get_encode_passage(passages, model, tokenizer, config):
    docs = []
    window_size = config['window_size']
    batch_size = config['batch_size']
    all_embeddings = []

    # Prepare the documents using the sliding window approach
    for i in range(len(passages) - window_size):
        a_doc = ''
        for j in range(window_size):
            a_doc += passages[i + j]
        docs.append(a_doc)

    # Iterate over the docs to create batches and encode them
    for batch_start in tqdm(range(0, len(docs), batch_size)):
        # Create a batch by slicing the docs
        batch_docs = docs[batch_start: batch_start + batch_size]

        # Tokenize the batch
        encoded_passages = tokenizer(batch_docs, return_tensors="pt", padding=True, truncation=True)

        # Move tokenized passages to GPU if available
        if torch.cuda.is_available():
            encoded_passages = {k: v.cuda() for k, v in encoded_passages.items()}

        # Compute embeddings for the batch
        with torch.no_grad():
            passage_embeddings = model(**encoded_passages).pooler_output

        # Collect embeddings from all batches
        all_embeddings.append(passage_embeddings.cpu())

    # Combine embeddings from all batches
    all_embeddings = torch.cat(all_embeddings, dim=0)

    torch.cuda.empty_cache()
    return all_embeddings


def find_similar_index(query, model, tokenizer, passage_embeddings, passage_uncertainty, config):
    encoded_query = tokenizer(query, return_tensors="pt", truncation=True)

    if torch.cuda.is_available():
        encoded_query = {k: v.cuda() for k, v in encoded_query.items()}
    with torch.no_grad():
        query_embedding = model(**encoded_query).pooler_output

    batch_size = config['batch_size']

    # Calculate the similarity between a query and a text collection
    all_similarity_scores = []

    for batch_start in range(0, len(passage_embeddings), batch_size):
        batch_embeddings = passage_embeddings[batch_start: batch_start + batch_size]
        if torch.cuda.is_available():
            similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, batch_embeddings.cuda())
        else:
            similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, batch_embeddings)
        all_similarity_scores.append(similarity_scores.cpu())
    all_similarity_scores = torch.cat(all_similarity_scores, dim=0)

    # Get the most relevant text
    w_all_similarity_scores = all_similarity_scores * torch.tensor(passage_uncertainty)

    most_similar_passage_index = w_all_similarity_scores.argmax().item()

    torch.cuda.empty_cache()
    return most_similar_passage_index, query_embedding.cpu()


def find_most_similar_sequence(a_sequence, model, tokenizer, passage_embeddings, passage_uncertainty, config):
    a_seq = ''
    for seq in a_sequence:
        a_seq += seq

    most_similar_passage_index, encoded_query = find_similar_index(a_seq, model, tokenizer, passage_embeddings,
                                                                   passage_uncertainty, config)
    index = [most_similar_passage_index]

    return index, encoded_query


def print_label(dataset_name, label):
    label = np.array(label)
    print(
        f'dataset:{dataset_name}, normal:{len(label[label == 0])}, abnormal:{len(label[label == 1])}, all:{len(label)}')


def print_metrics(result_file_url):
    result_df = pd.read_csv(result_file_url)
    y_true = np.array(result_df['ground_truth'].to_list())
    y_pred = np.array(result_df['result_label'].to_list())
    acc = accuracy_score(y_true, y_pred)

    print_label('test set seqs', y_true)

    y_true = y_true[y_pred != 2]
    y_pred = y_pred[y_pred != 2]

    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    print(f'acc:{acc * 100:.2f},f1:{f1 * 100:.2f},pre:{pre * 100:.2f},rec:{rec * 100:.2f}')


def print_metrics(result_file_url):
    result_df = pd.read_csv(result_file_url)
    y_true = np.array(result_df['ground_truth'].to_list())
    y_pred = np.array(result_df['result_label'].to_list())
    acc = accuracy_score(y_true, y_pred)

    print_label('test set seqs', y_true)

    y_true = y_true[y_pred != 2]
    y_pred = y_pred[y_pred != 2]

    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    print(f'acc:{acc * 100:.2f},f1:{f1 * 100:.2f},pre:{pre * 100:.2f},rec:{rec * 100:.2f}')


def print_label(dataset_name, label):
    label = np.array(label)
    print(
        f'dataset:{dataset_name}, normal:{len(label[label == 0])}, abnormal:{len(label[label == 1])}, all:{len(label)}')


def preprocess(config, dataset_dir='dataset'):
    seed = config['random_seed']
    dataset_name = config['dataset_name']

    set_seed(seed)
    parsing(dataset_name)

    directory = os.path.join(dataset_dir, dataset_name)

    df = pd.read_csv(os.path.join(directory, f'{dataset_name}.log_structured.csv'))
    return df
