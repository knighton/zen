from functools import reduce
import json
import os
from random import shuffle
from tqdm import tqdm
from zipfile import ZipFile

from .util import download, get_dataset_dir


class SNLIConfig(object):
    dataset_name = 'snli'
    url = 'https://www.nyu.edu/projects/bowman/multinli/snli_1.0.zip'
    zip_path_pattern = 'snli_1.0/snli_1.0_%s.jsonl'
    all_splits = ['train', 'dev', 'test']
    processed_subdir = 'processed'


class MultiNLIConfig(object):
    dataset_name = 'multi_nli'
    url = 'https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
    zip_path_pattern = 'multinli_0.9/multinli_0.9_%s.jsonl'
    all_splits = ['train', 'dev_matched', 'dev_mismatched']
    processed_subdir = 'processed'


def _get_processed_filename(processed_dir, input_format, split):
    return os.path.join(processed_dir, 'data_%s_%s.txt' % (input_format, split))


def _get_processed_count_filename(processed_dir, split):
    return os.path.join(processed_dir, 'count_%s.txt' % split)


def _write_line(out, premise, hypothesis, label):
    assert '|' not in premise
    assert '|' not in hypothesis
    label = {
        'entailment': 'E',
        'neutral': 'N',
        'contradiction': 'C',
    }[label]
    line = '%s|%s|%s\n' % (label, premise, hypothesis)
    out.write(line.encode('utf-8'))


def _process_split(zip_file, zip_path_pattern, processed_dir, split, verbose):
    if verbose:
        print('Processing %s split...' % split)
    zip_path = zip_path_pattern % split
    f = _get_processed_filename(processed_dir, 'text', split)
    out_text = open(f, 'wb')
    f = _get_processed_filename(processed_dir, 'parse', split)
    out_parse = open(f, 'wb')
    f = _get_processed_filename(processed_dir, 'binary_parse', split)
    out_binary_parse = open(f, 'wb')
    count = 0
    gen = zip_file.open(zip_path).readlines()
    if verbose == 2:
        gen = tqdm(gen, leave=False)
    for line in gen:
        x = json.loads(line.decode('utf-8'))
        label = x['gold_label']
        if label == '-':
            continue
        _write_line(out_text, x['sentence1'], x['sentence2'], label)
        _write_line(out_parse, x['sentence1_parse'], x['sentence2_parse'],
                    label)
        _write_line(out_binary_parse, x['sentence1_binary_parse'],
                    x['sentence2_binary_parse'], label)
        count += 1
    f = _get_processed_count_filename(processed_dir, split)
    with open(f, 'wb') as out:
        out.write(str(count).encode('utf-8'))


def _process(filename, zip_path_pattern, all_splits, processed_dir, verbose):
    os.mkdir(processed_dir)
    zip_file = ZipFile(filename)
    for split in all_splits:
        _process_split(zip_file, zip_path_pattern, processed_dir, split,
                       verbose)


def _load_line(line):
    label, premise, hypothesis = line.strip().split('|')
    label = {
        'E': 2,
        'N': 1,
        'C': 0,
    }[label]
    return premise, hypothesis, label


def _load_split(processed_dir, input_format, split, verbose):
    assert input_format in {'binary_parse', 'parse', 'text'}
    f = _get_processed_count_filename(processed_dir, split)
    count = int(open(f).read())
    filename = _get_processed_filename(processed_dir, input_format, split)
    premises = []
    hypotheses = []
    labels = []
    lines = open(filename)
    if verbose == 2:
        lines = tqdm(lines, total=count, leave=False)
    for line in lines:
        premise, hypothesis, label = _load_line(line)
        premises.append(premise)
        hypotheses.append(hypothesis)
        labels.append(label)
    return (premises, hypotheses), labels


def _load_nli(input_format, config, splits, verbose):
    assert len(splits) == 2
    dataset_dir = get_dataset_dir(config.dataset_name)
    processed_dir = os.path.join(dataset_dir, config.processed_subdir)
    if not os.path.exists(processed_dir):
        filename = os.path.join(dataset_dir, os.path.basename(config.url))
        if not os.path.exists(filename):
            download(config.url, filename, verbose)
        _process(filename, config.zip_path_pattern, config.all_splits,
                 processed_dir, verbose)
    train = _load_split(processed_dir, input_format, splits[0], verbose)
    val = _load_split(processed_dir, input_format, splits[1], verbose)
    return train, val


def load_snli(input_format, verbose=2):
    return _load_nli(input_format, SNLIConfig, ['train', 'dev'], verbose)


def load_multi_nli_same(input_format, verbose=2):
    return _load_nli(input_format, MultiNLIConfig, ['train', 'dev_matched'],
                     verbose)


def load_multi_nli_diff(input_format, verbose=2):
    return _load_nli(input_format, MultiNLIConfig, ['train', 'dev_mismatched'],
                     verbose)


def _blend(datasets):
    trains, vals = zip(*datasets)
    train = reduce(lambda a, b: a + b, trains)
    val = reduce(lambda a, b: a + b, vals)
    shuffle(train)
    shuffle(val)
    return train, val


def load_combined_nli(input_format, verbose=2):
    snli = load_snli(input_format, verbose)
    multi_nli = load_multi_nli_same(input_format, verbose)
    return _blend([snli, multi_nli])


_DATASET2LOAD_NLI = {
    'snli': load_snli,
    'multi_nli_same': load_multi_nli_same,
    'multi_nli_diff': load_multi_nli_diff,
    'combined_nli': load_combined_nli,
}


def load_nli(dataset, input_format, verbose=2):
    return _DATASET2LOAD_NLI[dataset](input_format, verbose)
