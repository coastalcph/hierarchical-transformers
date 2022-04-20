import pickle
import random

import nltk
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def train_text_featurizer(documents, tokenizer_path='google/bert_uncased_L-6_H-256_A-4', hidden_units=768):

    def tokenize(document: str):
        return tokenizer.tokenize(document,
                                  padding=False,
                                  truncation=True,
                                  max_length=1024)

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=1024)

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


def learn_idfs(documents, tokenizer_path='google/bert_uncased_L-6_H-256_A-4'):

    def tokenize(document: str):
        return tokenizer.tokenize(document)

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=1024)

    # init tfidf vectorizer
    vocab = [(key, value) for (key, value) in tokenizer.vocab.items()]
    vocab = sorted(vocab, key=lambda tup: tup[1])
    vocab = [key for (key, value) in vocab]
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, preprocessor=None, tokenizer=tokenize,
                                       vocabulary=vocab)

    tfidf_vectorizer.fit(documents)

    with open('./data/wikipedia-dataset/idf_scores.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer.idf_, file)


def embed_sentences(documents, model_path='all-MiniLM-L6-v2'):
    from sentence_transformers import SentenceTransformer

    # Define the model
    model = SentenceTransformer(model_path)

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    # Sub-sample sentences
    grouped_sentences = []
    for document in documents:
        doc_sentences = nltk.sent_tokenize(' '.join(document.split()[:1024]))
        # Build grouped sentences up to 100 words
        temp_sentence = ''
        for doc_sentence in doc_sentences:
            if len(temp_sentence.split()) + len(doc_sentence.split()) <= 100:
                temp_sentence += ' ' + doc_sentence
            else:
                if len(temp_sentence):
                    grouped_sentences.append(temp_sentence)
                temp_sentence = doc_sentence
        if len(temp_sentence):
            grouped_sentences.append(temp_sentence)
    del documents

    # Compute the embeddings using the multi-process pool
    sentence_embeddings = model.encode_multi_process(grouped_sentences, pool)
    print("Embeddings computed. Shape:", sentence_embeddings.shape)

    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

    with open('../data/wikipedia-dataset/sentence_embeddings.pkl', 'wb') as file:
        pickle.dump(sentence_embeddings, file)

    print('SENTENCE EMBEDDINGS DONE!')


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # load dataset
    dataset = load_dataset("lex_glue", "eurlex", split='train')
    dataset = dataset['text']
    subset = random.sample(range(len(dataset)), k=100)
    dataset_small = []
    for i in tqdm.tqdm(subset):
        dataset_small.append(dataset[i])

    # re-load tokenizer and test
    embed_sentences(documents=dataset_small)

