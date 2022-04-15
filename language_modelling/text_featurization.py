import pickle
import random

import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def train_text_featurizer(documents, tokenizer, hidden_units=768):

    def tokenize(document: str):
        return tokenizer.tokenize(document)

    # init tfidf vectorizer
    vocab = [(key, value) for (key, value) in tokenizer.vocab.items()]
    vocab = sorted(vocab, key=lambda tup: tup[1])
    vocab = [key for (key, value) in vocab]
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, preprocessor=None, tokenizer=tokenize,
                                       vocabulary=vocab)
    pca_solver = PCA(n_components=hidden_units)

    tfidf_scores = tfidf_vectorizer.fit_transform(documents)
    print('TFIDF-VECTORIZER DONE!')

    pca_solver.fit(tfidf_scores.toarray())
    print('PCA SOLVER DONE!')

    with open('./data/wikipedia-dataset/tifidf_vectorizer.pkl', 'wb') as fin:
        pickle.dump(tfidf_vectorizer, fin)
    print('TFIDF-VECTORIZER SAVED!')

    with open('./data/wikipedia-dataset/pca_solver.pkl', 'wb') as fin:
        pickle.dump(pca_solver, fin)
    print('PCA SOLVER SAVED!')


def learn_idfs(documents, tokenizer):

    def tokenize(document: str):
        return tokenizer.tokenize(document)

    # init tfidf vectorizer
    vocab = [(key, value) for (key, value) in tokenizer.vocab.items()]
    vocab = sorted(vocab, key=lambda tup: tup[1])
    vocab = [key for (key, value) in vocab]
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, preprocessor=None, tokenizer=tokenize,
                                       vocabulary=vocab)

    tfidf_vectorizer.fit(documents)

    with open('./data/wikipedia-dataset/idf_scores.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer.idf_, file)
    print('IDFs SAVED!')

    # with open('./data/wikipedia-dataset/idf_terms.pkl', 'wb') as file:
    #     idf_terms = {term: idx for term, idx in enumerate(tfidf_vectorizer.get_feature_names_out())}
    #     pickle.dump(idf_terms, file)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # load dataset
    dataset = load_dataset("./data/wikipedia-dataset", split='train')
    dataset = dataset['text']
    subset = random.sample(range(len(dataset)), k=500000)
    CUSTOM_TOK_FOLDER = 'google/bert_uncased_L-6_H-256_A-4'
    dataset_small = []
    for i in tqdm.tqdm(subset):
        dataset_small.append(dataset[i])

    # re-load tokenizer and test
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOK_FOLDER,  model_max_length=1024)
    train_text_featurizer(documents=dataset_small, tokenizer=tokenizer, hidden_units=256)

