import argparse
import yaml
import json
from utils.utils import get_label, join_strings_with_tabs_and_newline, parse_label, save_prompt, \
    get_dpr_model_and_tokenizer, print_metrics, get_uncertainty, get_encode_passage, \
    find_most_similar_sequence, print_label, preprocess
import pandas as pd
from llm.chat import get_qwen
from tqdm import tqdm
import torch


def main():
    parser = argparse.ArgumentParser(description="EagerLog")
    parser.add_argument("--config", type=str, default="BGL.yaml")
    args = parser.parse_args()
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(json.dumps(config, indent=4))
    log_df = preprocess(config)

    window_size = config['window_size']
    step_size = config['step_size']
    dataset_name = config['dataset_name']

    rag_start = config['rag_start']
    rag_end = config['rag_end']
    test_start = config['test_start']
    test_end = config['test_end']

    rag_doc = log_df[rag_start:rag_end]

    test_logs = log_df[test_start:test_end]

    print_label('doc', rag_doc['Label'].to_list())
    print_label('test set', test_logs['Label'].to_list())

    rag_logs = rag_doc['RawLog'].to_list()
    rag_split_logs = [rag_logs[i:i + window_size] for i in range(len(rag_logs) - window_size)]

    dpr_model, dpr_tokenizer = get_dpr_model_and_tokenizer()
    passage_embeddings = get_encode_passage(rag_logs, dpr_model, dpr_tokenizer, config)
    passage_labels = [get_label(dataset_name, item) for item in
                      [rag_doc['Label'].to_list()[i:i + window_size] for i in range(len(rag_doc) - window_size)]]
    passage_uncertainty = [1.0] * len(passage_labels)

    human_label_efforts = 0
    add_samples = 0
    save_epoch = 100
    epoch = 0

    result_df = {
        'index': [],
        'llm_text': [],
        'result_text': [],
        'result_label': [],
        'rag_labels': [],
        'ground_truth': [],
        'doc_index': [],
        'uncertainty': [],
        'human_label': [],
    }

    for i in tqdm(range(0, len(test_logs) - window_size, step_size)):
        log_lines = test_logs['RawLog'].iloc[i:i + window_size].to_list()
        labels = test_logs['Label'].iloc[i:i + window_size].to_list()
        # find max similar seq
        index, encoded_query = find_most_similar_sequence(log_lines, dpr_model, dpr_tokenizer,
                                                          passage_embeddings, passage_uncertainty, config)
        index = index[:1]
        evidences = []
        rag_labels = []
        # get evidence
        for j, item in enumerate(index):
            explanation_raw_logs = rag_split_logs[item]
            explanation_logs = join_strings_with_tabs_and_newline(explanation_raw_logs, 2)
            label = passage_labels[item]
            rag_labels.append(label)
            uncertainty = passage_uncertainty[item]
            state = '[Normal]' if label == 0 else '[Abnormal]'
            # get explanation
            explanation_template = open(f"prompts/explanation_prompt.txt", "r", encoding='utf-8')
            explanation_prompt = explanation_template.read().format(
                file_system=dataset_name, logs=explanation_logs, state=state
            )
            explanation = get_qwen(explanation_prompt)
            evidence_template = open(f"prompts/evidence.txt", "r", encoding='utf-8')
            evidence = evidence_template.read().format(
                i=j, state=state, explanation=explanation, logs=explanation_logs, uncertainty=uncertainty
            )
            evidences.append(evidence)
        evidences = join_strings_with_tabs_and_newline(evidences, 0)
        logs = join_strings_with_tabs_and_newline(log_lines, 2)
        analysis_template = open(f"prompts/analysis_prompt.txt", "r", encoding='utf-8')
        analysis_prompt = analysis_template.read().format(
            file_system=dataset_name, evidences=evidences, logs=logs
        )
        llm_text = get_qwen(analysis_prompt)
        save_prompt(f'./prompts/.cache/{dataset_name}_{i}.txt', analysis_prompt)
        result_label, result_text = parse_label(llm_text)
        result_df['index'].append(i)
        result_df['llm_text'].append(llm_text)
        result_df['result_text'].append(result_text)
        result_df['ground_truth'].append(get_label(dataset_name, labels))
        result_df['doc_index'].append(index)
        result_df['rag_labels'].append(rag_labels)
        uncertainty = float(get_uncertainty(llm_text))
        result_df['uncertainty'].append(uncertainty)
        human_label = 0

        # If the uncertainty is less than the threshold, manual labeling is performed and then added to the training set.
        if uncertainty < float(config['label_threshold']):
            result_label = get_label(dataset_name, labels)
            human_label_efforts += 1
            uncertainty = 1.0
            human_label = 1
        if uncertainty > float(config['select_threshold']):
            rag_split_logs.append(log_lines)
            add_samples += 1
            passage_embeddings = torch.cat((passage_embeddings, encoded_query), dim=0)
            passage_uncertainty.append(uncertainty)
            passage_labels.append(result_label)
        result_df['result_label'].append(result_label)
        result_df['human_label'].append(human_label)

        epoch += 1
        if epoch % save_epoch == 0:
            temp_result_df = pd.DataFrame(result_df)
            temp_result_df.to_csv(f'./results/{dataset_name}_epoch_{epoch}.csv', index=False)
            print_metrics(f'./results/{dataset_name}_epoch_{epoch}.csv')

    result_df = pd.DataFrame(result_df)
    result_df.to_csv(f'./results/{dataset_name}.csv', index=False)
    print_metrics(f'./results/{dataset_name}.csv')
    print(f'human_label_efforts: {human_label_efforts}, add_samples:{add_samples}')


if __name__ == "__main__":
    main()
