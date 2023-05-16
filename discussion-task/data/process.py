import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


script_folder = Path(__file__).parent


# Read data
dataset = []

with open(script_folder / 'threads.jsonl') as f:
    for line in tqdm(f, total=65169, desc='Read Threads'):
        thread = json.loads(line)

        original_poster = thread['author']
        delta_thread = thread['delta']

        def _go_through_comments(comments: list) -> bool:
            global sum_num_comments
            # check delta
            has_delta_comment = False
            for comment in comments:
                if comment['author'] == original_poster and ('∆' in comment['body'] or '\u2206' in comment['body']):
                    has_delta_comment = True
                    comments.remove(comment)  # delete delta award comment
                    break
            # process
            for comment in comments:
                # filter delete comments by moderator
                if comment['distinguished'] == 'moderator' and 'removed' in comment['body'].lower():
                    continue
                # filter deleted comments
                if comment['body'] == '[deleted]':
                    continue
                dataset.append({
                    'id': comment['id'],
                    'author': comment['author'],
                    'parent_id': comment['parent_id'],
                    'level': comment['level'],
                    'distinguished': comment['distinguished'],
                    'score': comment['score'],
                    'title': None,
                    'body': comment['body'],
                    'delta_awarded': _go_through_comments(comment['children']),  # TODO: delta
                    'delta_thread': delta_thread,
                })
            return has_delta_comment

        # filter empty discussions
        if thread['num_comments'] <= 1:
            continue
        # filter modpost
        if '[Mod Post]' in thread['title']:
            continue

        # start recursion for the thread
        _go_through_comments(thread['comments'])

        dataset.append({
            'id': thread['id'],
            'author': original_poster,
            'parent_id': None,
            'level': None,
            'distinguished': thread['distinguished'],
            'score': thread['score'],
            'title': thread['title'],
            'body': thread['selftext'],
            'delta_awarded': False,
            'delta_thread': delta_thread,
        })


dataset = dataset[::-1]
dataset_lvl0 = [sample for sample in dataset if sample['level'] is None or sample['level'] == 0]
dataset_delta = [sample for sample in dataset if sample['delta_thread']]
dataset_delta_lvl0 = [sample for sample in dataset_lvl0 if sample['delta_thread']]


# EDA
print('Dataset (full):', len(dataset))
print('Dataset (lvl0):', len(dataset_lvl0))
print('Dataset (delta):', len(dataset_delta))
print('Dataset (delta lvl0):', len(dataset_delta_lvl0))

num_threads = sum(1 for sample in dataset if sample['level'] is None)
num_threads_delta = sum(1 for sample in dataset_delta if sample['level'] is None)
print('Num threads:', num_threads)
print('Num delta threads:', num_threads_delta)

print('Average comments total:', sum(1 for sample in dataset if sample['level'] is not None) / num_threads)
print('Average comments lvl0:', sum(1 for sample in dataset_lvl0 if sample['level'] == 0) / num_threads)
print('Average comments total delta:', sum(1 for sample in dataset_delta if sample['level'] is not None) / num_threads_delta)
print('Average comments lvl0 delta:', sum(1 for sample in dataset_delta_lvl0 if sample['level'] == 0) / num_threads_delta)


authors_threads_dict = defaultdict(int)
for sample in dataset:
    if sample['level'] is None:
        authors_threads_dict[sample['author']] += 1
print('Num authors:', len(authors_threads_dict.keys()))
print('Average num author threads:', np.mean(list(authors_threads_dict.values())))

delta_authors_threads_dict = defaultdict(int)
for sample in dataset_delta:
    if sample['level'] is None:
        delta_authors_threads_dict[sample['author']] += 1
print('Num authors delta:', len(delta_authors_threads_dict.keys()))
print('Average num author threads delta:', np.mean(list(delta_authors_threads_dict.values())))


# Save
with open(script_folder / 'dataset_full.jsonl', 'w') as f:
    for sample in tqdm(dataset, desc='Save dataset (full)'):
        json.dump(sample, f)
        f.write('\n')

with open(script_folder / 'dataset_lvl0.jsonl', 'w') as f:
    for sample in tqdm(dataset_lvl0, desc='Save dataset (lvl0)'):
        json.dump(sample, f)
        f.write('\n')

with open(script_folder / 'dataset_full_delta.jsonl', 'w') as f:
    for sample in tqdm(dataset_delta, desc='Save dataset (full \w delta)'):
        json.dump(sample, f)
        f.write('\n')

with open(script_folder / 'dataset_lvl0_delta.jsonl', 'w') as f:
    for sample in tqdm(dataset_delta_lvl0, desc='Save dataset (lvl0 \w delta)'):
        json.dump(sample, f)
        f.write('\n')
