import copy
import glob
import json
import re
from collections import Counter
import pandas as pd
from ast import literal_eval
USED_LABELS = ['Subsumtion',
               'Vorherige Rechtsprechung des EGMR',
               'Verhältnismäßigkeitsprüfung – Angemessenheit', #Proportionality
               'Entscheidung des EGMR',
               'Verhältnismäßigkeitsprüfung – Rechtsgrundlage',  #Proportionality
               'Verhältnismäßigkeitsprüfung – Legitimer Zweck',  #Proportionality
               'Konsens der prozessualen Parteien']

ECTHR_ARG_TYPES = ['Application',
                   'Precedent',
                   'Proportionality',  #Proportionality
                   'Decision',
                   'Legal Basis',  #Proportionality
                   'Legitimate Purpose',  #Proportionality
                   'Non Contestation']

all_labels = []
article_paragraphs = []
sample_ids = []
article_paragraph_labels = []


def fix_labels(labels):
    labels = list(set([re.sub('[BI]-', '', label) for label in labels]))
    if len(labels) >= 2 and 'O' in labels:
        labels.remove('O')
    labels = [ECTHR_ARG_TYPES[USED_LABELS.index(label)] for label in labels if label in USED_LABELS]
    if len(labels) == 0:
        labels = ['O']
    return labels

short_chunks = []

for subset in ['train', 'val', 'test']:
    for filename in glob.glob(f'../mining-legal-arguments/data/{subset}/argType/*.csv'):
        with open(filename) as file:
            df = pd.read_csv(file, sep='\t', encoding='utf-8')
            df['labels'] = df['labels'].map(lambda x: literal_eval(x))
            df['tokens'] = df['tokens'].map(lambda x: literal_eval(x))
            paragraphs = []
            paragraph_labels = []
            temp_paragraph = ''
            temp_labels = []
            for tokens, token_labels in zip(df['tokens'], df['labels']):
                paragraph = re.sub(r'([\(\-\[]) ', r'\1', re.sub(r' ([\.\)\,\:\;\'\-\]]|\'s)', r'\1', ' '.join(tokens)))
                labels = fix_labels(token_labels)
                if len(labels) > 1:
                    print()
                if re.match('(FOR THESE REASONS, THE COURT|for these reasons, the court unanimously)', paragraph):
                    break
                if not re.match('\d{2,}\.', paragraph) and not re.match('[IVX]+\. [A-Z]{2,}', paragraph):
                    if len(paragraph.split()) <= 5 and re.match('(([A-Z]|\d)\.|\([a-zα-ω]+\))', paragraph):
                        short_chunks.append(paragraph)
                        continue
                    temp_paragraph = temp_paragraph + '\n' + paragraph
                    temp_labels.extend(labels)
                    continue
                elif len(paragraphs) and len(temp_paragraph):
                    paragraphs[-1] = paragraphs[-1] + '' + temp_paragraph
                    paragraph_labels[-1] = list(set(copy.deepcopy(paragraph_labels[-1]) + copy.deepcopy(temp_labels)))
                    if len(paragraph_labels[-1]) > 1 and 'O' in paragraph_labels[-1]:
                        paragraph_labels[-1].remove('O')
                    temp_paragraph = ''
                    temp_labels = []
                if len(paragraph.split()) <= 10 and not re.match('\d{2,}\.', paragraph) and not re.match('[IVX]+\. [A-Z]{2,}', paragraph):
                    continue
                if re.match('[IVX]+\. [A-Z]{2,}', paragraph) and len(paragraphs):
                    article_paragraphs.append(copy.deepcopy(paragraphs))
                    article_paragraph_labels.append(copy.deepcopy(paragraph_labels))
                    sample_ids.append(filename.split('argType/')[1].replace('.csv', ''))
                    paragraphs = []
                    paragraph_labels = []
                paragraphs.append(paragraph)
                paragraph_labels.append(labels)
            article_paragraphs.append(copy.deepcopy(paragraphs))
            article_paragraph_labels.append(copy.deepcopy(paragraph_labels))
            sample_ids.append(filename.split('argType/')[1].replace('.csv', ''))

article_paragraphs_clean = []
article_paragraph_labels_clean = []
for paragraphs, paragraph_labels in zip(article_paragraphs, article_paragraph_labels):
    if len(paragraphs) == len(paragraph_labels):
        article_paragraphs_clean.append(copy.deepcopy(paragraphs[:32]))
        article_paragraph_labels_clean.append(copy.deepcopy(paragraph_labels[:32]))
        all_labels.extend([label for label_group in copy.deepcopy(paragraph_labels[:32]) for label in label_group])

label_counts = Counter(all_labels)
n_paragraphs = []
for paragraphs in article_paragraphs_clean:
    if len(paragraphs) <= 32:
        n_paragraphs.append(32)
    elif len(paragraphs) <= 64:
        n_paragraphs.append(64)
    elif len(paragraphs) <= 128:
        n_paragraphs.append(128)
    else:
        n_paragraphs.append('Long')

par_counts = Counter(n_paragraphs)
print(par_counts.most_common())

count = 0
with open('ecthr_arguments.jsonl', 'w') as file:
    for paragraphs, labels, sample_id in zip(article_paragraphs_clean, article_paragraph_labels_clean, sample_ids):
        count += 1
        if count <= 900:
            data_type = 'train'
        elif count <= 1000:
            data_type = 'dev'
        elif count <= 1100:
            data_type = 'test'
        else:
            break
        if labels is None:
            print()
        else:
            for paragraph_labels in labels:
                if paragraph_labels is None:
                    print()
        file.write(json.dumps({'case_id': sample_id, 'paragraphs': paragraphs, 'labels': labels, 'data_type': data_type}) + '\n')


label_counts = {'train': [], 'dev': [], 'test': []}
with open('ecthr_arguments.jsonl', ) as file:
    for line in file:
        data = json.loads(line)
        label_counts[data['data_type']].extend([label for par_labels in data['labels'] for label in par_labels])

for key in label_counts:
    print(Counter(label_counts[key]).most_common())