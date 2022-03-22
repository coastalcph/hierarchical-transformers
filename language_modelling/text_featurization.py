from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def train_text_featurizer(documents, tokenizer, hidden_units=768):

    def tokenize(document: str):
        return tokenizer.tokenize(document)

    # init tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, preprocessor=None, tokenizer=tokenize,
                                       vocabulary=list(tokenizer.vocab.keys()))
    pca_solver = PCA(n_components=hidden_units)

    tfidf_scores = tfidf_vectorizer.fit_transform(documents)
    pca_solver.fit(tfidf_scores.toarray())

    return tfidf_vectorizer, pca_solver


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # load dataset
    documents = load_dataset("wikitext", "wikitext-103-raw-v1", split='test')['text']

    CUSTOM_TOK_FOLDER = '../data/custom-tokenizer'

    # re-load tokenizer and test
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOK_FOLDER)
    tfidf_vect, pca_solver = train_text_featurizer(documents=documents, tokenizer=tokenizer)
