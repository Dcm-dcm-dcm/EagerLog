# Drain: https://github.com/logpai/logparser

import os
import tarfile
import sys
import urllib.request

from . import DrainBGL,DrainThunderbird,DrainZookeeper

def _progress(block_num, block_size, total_size):
    sys.stdout.write('\r>> Downloading %.1f%%' % (
                     float(block_num * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()


def download_dataset(dataset_name, dataset_dir='dataset'):
    """Download and parsing dataset

    Args:
        dataset_name: name of the log dataset
        output_dir: directory name for data storage

    Returns:
        Structured log data in Pandas Dataframe after adopt Drain
    """
    directory = f'./{dataset_dir}/{dataset_name}'
    save_log_file = os.path.join(directory, f'{dataset_name}.log')
    if os.path.exists(save_log_file):
        print('log file exist')
        return
    if not os.path.exists(directory):
        print('Making directory for dataset storage')
        os.makedirs(directory)
    url_list = {
        'BGL': 'https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1',
        'Thunderbird': 'https://zenodo.org/record/3227177/files/Thunderbird.tar.gz?download=1',
        'Zookeeper': 'https://zenodo.org/record/3227177/files/Zookeeper.tar.gz?download=1'
    }
    if not dataset_name in url_list:
        print('Error: dataset type not found')
        exit()

    url = url_list[dataset_name]
    downloaded_filename = directory + f'/{dataset_name}.tar.gz'
    if not os.path.exists(downloaded_filename):
        print(f'{downloaded_filename} not exists')
        urllib.request.urlretrieve(url, downloaded_filename, _progress)
    if not os.path.exists( f'{directory}/{dataset_name}.log'):
        tar = tarfile.open(downloaded_filename, "r|gz")
        tar.extractall(directory)
        tar.close()


def parsing(dataset_name, dataset_dir='dataset'):
    """Download and parsing dataset

    Args:
        dataset_name: name of the log dataset
        output_dir: directory name for data storage

    Returns:
        Structured log data in Pandas Dataframe after adopt Drain
    """
    directory = f'./{dataset_dir}/{dataset_name}'
    input_dir = directory  # The input directory of log file
    output_dir = directory  # The output directory of parsing results
    log_file = f'{dataset_name}.log'  # The input log file name
    download_dataset(dataset_name, dataset_dir)

    parse_file_url = os.path.join(directory, f'{dataset_name}.log_structured.csv')
    if os.path.exists(parse_file_url):
        print(f'{parse_file_url} exists')
        return

    if dataset_name == 'BGL':
        log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # BGL log format
        regex = [
            {"regex_pattern": "core\.\d+", "mask_with": "CORE"},
            {"regex_pattern": "blk_(|-)[0-9]+", "mask_with": "ID"},
            {"regex_pattern": "(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)", "mask_with": "IP"},
            {"regex_pattern": "([0-9a-f]+[:][0-9a-f]+)", "mask_with": "Word"},
            {"regex_pattern": "fpr[0-9]+[=]0x[0-9a-f]+ [0-9a-f]+ [0-9a-f]+ [0-9a-f]+", "mask_with": "FPR"},
            {"regex_pattern": "r[0-9]+[=]0x[0-9a-f]+", "mask_with": "Word"},
            {"regex_pattern": "[l|c|xe|ct]r=0x[0-9a-f]+", "mask_with": "Word"},
            {"regex_pattern": "0x[0-9a-f]+", "mask_with": "Word"},
            {"regex_pattern": "(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$", "mask_with": " NUM"},
        ]
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes

        parser = DrainBGL.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_file)

        try:
            os.remove(log_file)
        except OSError:
            pass

    elif dataset_name == 'Thunderbird':
        log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
        # Regular expression list for optional preprocessing (default: [])
        regex = [
            {"regex_pattern": "(\d+\.){3}\d+", "mask_with": "IP"},
            {"regex_pattern": "[a-d]n[0-9]+", "mask_with": "ID"},
            {"regex_pattern": "\<[0-9a-f]{16}\>\{.+\}", "mask_with": "SEQ"}
        ]
        st = 0.3  # Similarity threshold
        depth = 2  # Depth of all leaf nodes

        parser = DrainThunderbird.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st,
                                            rex=regex)
        parser.parse(log_file)

        try:
            os.remove(log_file)
        except OSError:
            pass

    elif dataset_name == 'Zookeeper':
        log_format = '<Date> <Time> - <Label>  \[<Node>:<Component>@<Id>\] - <Content>'  # Zookeeper log format
        regex = [
            {"regex_pattern": "(/|)(\d+\.){3}\d+(:\d+)?", "mask_with": "IP"}
        ]
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes

        parser = DrainZookeeper.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_file)

        try:
            os.remove(log_file)
        except OSError:
            pass


