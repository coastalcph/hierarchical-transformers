# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""English Wikipedia dataset containing cleaned articles."""

import json
import os

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@ONLINE {wikidump,
    author = {Wikimedia Foundation},
    title  = {Wikimedia Downloads},
    url    = {https://dumps.wikimedia.org}
}
"""

_DESCRIPTION = """\
Wikipedia dataset containing cleaned articles.
The datasets are built from the Wikipedia dump
(https://dumps.wikimedia.org/). Each example
contains the content of one full Wikipedia article with cleaning to strip
markdown and unwanted sections (references, etc.).
"""

_LICENSE = (
    "This work is licensed under the Creative Commons Attribution-ShareAlike "
    "3.0 Unported License. To view a copy of this license, visit "
    "http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to "
    "Creative Commons, PO Box 1866, Mountain View, CA 94042, USA."
)

_VERSION = datasets.Version("1.0.0", "")


class WikipediaConfig(datasets.BuilderConfig):
    """BuilderConfig for Wikipedia."""

    def __init__(self, dump=None, version=_VERSION, **kwargs):
        """BuilderConfig for Wikipedia.

        Args:
          language: string, the language code for the Wikipedia dump to use.
          date: string, date of the Wikipedia dump in YYYYMMDD format. A list of
            available dates can be found at https://dumps.wikimedia.org/enwiki/.
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            name=f"{dump}",
            description=f"Wikipedia dataset for {dump} dump.",
            version=version,
            **kwargs,
        )
        self.dump = dump


_DATE = "20220301"


class Wikipedia(datasets.GeneratorBasedBuilder):
    """Wikipedia dataset."""

    # Use mirror (your.org) to avoid download caps.
    BUILDER_CONFIG_CLASS = WikipediaConfig
    BUILDER_CONFIGS = [
        WikipediaConfig(
            dump="20200501.en",
        )  # pylint:disable=g-complex-comprehension
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
            # No default supervised_keys.
            supervised_keys=None,
            homepage="https://dumps.wikimedia.org",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download(os.path.join('./', f'wikipedia.{self.config.dump}.tar.gz'))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": 'train.jsonl',
                            "split": "train",
                            "files": dl_manager.iter_archive(data_dir)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": 'test.jsonl',
                            "split": "test",
                            "files": dl_manager.iter_archive(data_dir)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": 'dev.jsonl',
                            "split": "validation",
                            "files": dl_manager.iter_archive(data_dir)},
            ),
        ]

    def _generate_examples(self, filepath, split, files):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, "r", encoding="utf-8") as f:
            for path, f in files:
                if path == filepath:
                    for id_, row in enumerate(f):
                        data = json.loads(row)
                        yield id_, {
                            "title": data['title'],
                            "text": data['text'],
                        }
