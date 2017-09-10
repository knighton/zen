import numpy as np
import os

from .util import download, get_dataset_dir


_DATASET_NAME = 'quora_question_pairs'
_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'


def _load(filename, verbose):
    question_1s = []
    question_2s = []
    is_dupes = []
    lines = open(filename).read().strip().split('\n')[1:]
    for i, line in enumerate(lines):
        try:
            id_ = int(line.split()[0])
        except:
            lines[i - 1] += lines[i]
            lines[i] = None
    lines = list(filter(bool, lines))
    for line in lines:
        ss = line.split('\t')
        q1 = ss[3]
        question_1s.append(q1)
        q2 = ss[4]
        question_2s.append(q2)
        is_dupe = int(ss[5])
        assert is_dupe in {0, 1}
        is_dupes.append(is_dupe)
    return question_1s, question_2s, np.array(is_dupes, dtype='int64')


def load_quora_duplicate_questions(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    local = os.path.join(dataset_dir, os.path.basename(_URL))
    download(_URL, local, verbose)
    return _load(local, verbose)
