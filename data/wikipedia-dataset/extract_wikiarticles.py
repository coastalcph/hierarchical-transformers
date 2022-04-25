from datasets import load_dataset
import json, tqdm


'''The full wikipedia does not look good to our task, as we do not want articles having just a list of short items;
Try wikitext-103 to find page titles'''
def extract_titles_from_wiki103():
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    wikitext_103_titles = set()
    for line in dataset["train"]["text"] + dataset["validation"]["text"] + dataset["test"]["text"]:
        line = line.strip()
        if line.startswith("= ") and line.endswith(" ="):
            line = line[2:-2]
            if not line.startswith("="):
                wikitext_103_titles.add(line)
    # Extract 28918 titles from wikitext-103-v1
    print(f"Extract {len(wikitext_103_titles)} titles from wikitext-103-v1")
    return wikitext_103_titles


def extract_articles_from_wiki(wikitext_103_titles):
    dataset = load_dataset("wikipedia", "20200501.en")["train"]
    examples = []
    for title, text in tqdm.tqdm(zip(dataset["title"], dataset["text"]), total=len(dataset["title"])):
        if title in wikitext_103_titles:
            examples.append({"title": title, "text": text.strip()})
    return examples


def main():
    wikitext_103_titles = extract_titles_from_wiki103()
    articles = extract_articles_from_wiki(wikitext_103_titles)
    with open("/data/dai031/TEMP/hi-transformers/articles.json", "w") as f:
        for i in articles:
            f.write(f"{json.dumps(i)}\n")
    # Extract 19135 articles from wikipedia
    print(f"Extract {len(articles)} articles from wikipedia")


if __name__ == "__main__":
    main()
