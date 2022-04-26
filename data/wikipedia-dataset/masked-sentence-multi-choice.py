import json, numpy as np, os, random


def load_all_wiki_articles(dir):
    all_articles = {}
    for dir, _, filenames in os.walk(dir):
        for f in filenames:
            sentences = json.load(open(os.path.join(dir, f)))
            article_idx = len(all_articles)
            all_articles[article_idx] = {"sentences": sentences}
    return all_articles


def load_all_ptb_articles(dir):
    all_articles = {}
    for dir, _, filenames in os.walk(dir):
        for f in filenames:
            sentences = [l.strip() for l in open(os.path.join(dir, f), encoding="latin-1")]
            sentences = [l for l in sentences if len(l) > 0 and not l.startswith(".START")]
            if len(sentences) == 0: continue
            article_idx = len(all_articles)
            all_articles[article_idx] = {"sentences": sentences}
    return all_articles


def create_examples(all_articles, max_examples_per_article=20, num_choices=5):
    def get_random_sent_indices(article_idx, size=1):
        sent_idx = np.random.randint(len(all_articles[article_idx]["sentences"]), size=size)
        return [int(i) for i in set(sent_idx)]

    examples, unique_example_indices = [], set()
    for article_idx in range(len(all_articles)):
        for masked_sent_idx in get_random_sent_indices(article_idx, max_examples_per_article):
            if (article_idx, masked_sent_idx) in unique_example_indices:
                continue
            else:
                unique_example_indices.add((article_idx, masked_sent_idx))

            masked_sentence = all_articles[article_idx]["sentences"][masked_sent_idx]
            chocies = set([masked_sentence])
            for other_article_idx in np.random.randint(len(all_articles), size=num_choices * 2):
                if len(chocies) >= num_choices:
                    break
                if other_article_idx == article_idx: continue
                for negative_sent_idx in get_random_sent_indices(other_article_idx):
                    chocies.add(all_articles[other_article_idx]["sentences"][negative_sent_idx])
            chocies = list(chocies)
            random.shuffle(chocies)
            examples.append({"original_sentences": all_articles[article_idx]["sentences"], "masked_sent_idx": masked_sent_idx, "article_idx": article_idx, "chocies": chocies, "label": chocies.index(masked_sentence)})
    return examples


if __name__ == "__main__":
    all_articles = load_all_wiki_articles("/data/dai031/TEMP/hi-transformers/articles")
    examples = create_examples(all_articles)
    with open("/data/dai031/TEMP/hi-transformers/wiki.examples", "w") as f:
        for e in examples:
            f.write(f"{json.dumps(e)}\n")

    all_articles = load_all_ptb_articles("/data/dai031/Corpora/PTB/LDC99T42/treebank_3_LDC99T42/treebank_3/raw/wsj")
    examples = create_examples(all_articles)
    with open("/data/dai031/TEMP/hi-transformers/ptb.examples", "w") as f:
        for e in examples:
            f.write(f"{json.dumps(e)}\n")