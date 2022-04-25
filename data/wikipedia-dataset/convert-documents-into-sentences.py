import json, os, spacy, sys, time
print(spacy.__version__)
nlp = spacy.load("en_core_web_lg")


def split_text_into_sentences(text):
    sentences = []
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para.split()) <= 5:
            sentences.append(para)
        else:
            sent_boundaries = []
            for t in nlp(para, disable=["ner"]):
                if t.is_sent_start:
                    sent_boundaries.append(t.idx)

            if len(sent_boundaries) > 1:
                for s, e in zip(sent_boundaries[:-1], sent_boundaries[1:]):
                    sentences.append(para[s:e].strip())
            else:
                sentences.append(para)
    return [s for s in sentences if len(s) > 0]


def main():
    num_articles, num_sentences = 0, 0
    start_time = time.time()
    with open("/data/dai031/TEMP/hi-transformers/articles.json") as f:
        for line in f:
            num_articles += 1
            if len(sys.argv) > 1 and num_articles >= int(sys.argv[1]): break
            text = json.loads(line)["text"]
            output_filepath = f"/data/dai031/TEMP/hi-transformers/articles/{num_articles}.json"
            if os.path.exists(output_filepath): continue
            sentences = split_text_into_sentences(text)
            num_sentences += len(sentences)
            json.dump(sentences, open(output_filepath, "w"))
    print(f"Extract {num_articles} articles from wikipedia, containing {num_sentences} sentences, "
          f"cost {(time.time() - start_time) / 60:.1f}m")


if __name__ == "__main__":
    main()
