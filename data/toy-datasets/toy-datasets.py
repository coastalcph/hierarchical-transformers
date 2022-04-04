# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Datasets"""

import json
import os
import datasets


_DESCRIPTION = """Datasets"""


class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for LexGLUE."""

    def __init__(
        self,
        **kwargs,
    ):
        """BuilderConfig for Datasets.
        """
        super(DatasetConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.data_file = f'{self.name}.jsonl'


class ToyDataset(datasets.GeneratorBasedBuilder):
    """LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. Version 1.0"""

    BUILDER_CONFIGS = [
        DatasetConfig(
            name="wikipedia",
        ),
        DatasetConfig(
            name="eurlex",
        ),
    ]

    def _info(self):
        features = {"text": datasets.Value("string")}
        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(features),
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract('.')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.data_file), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.data_file), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.data_file), "split": "validation"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if data["data_type"] == split:
                    yield id_, {
                        "text": data['text']
                    }
