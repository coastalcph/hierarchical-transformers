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
"""QuALITY: Question Answering with Long Input Texts, Yes!"""


import json
import os

import datasets


_CITATION = """@article{pang2021quality,
  title={{QuALITY}: Question Answering with Long Input Texts, Yes!},
  author={Pang, Richard Yuanzhe and Parrish, Alicia and Joshi, Nitish and Nangia, Nikita and Phang, Jason and Chen, 
  Angelica and Padmakumar, Vishakh and Ma, Johnny and Thompson, Jana and He, He and Bowman, Samuel R.},
  journal={arXiv preprint arXiv:2112.08608},
  year={2021}
}"""

_DESCRIPTION = """QuALITY: Question Answering with Long Input Texts, Yes!"""

_LICENSE = "CC BY-SA (Creative Commons / Attribution-ShareAlike)"

_LABEL_NAMES = [f'choice_{i}' for i in range(5)]


class QuALITY(datasets.GeneratorBasedBuilder):
    """QuALITY: Question Answering with Long Input Texts, Yes!"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="quality", version=VERSION, description="QuALITY: Question Answering with Long Input Texts"
        ),
    ]

    DEFAULT_CONFIG_NAME = "quality"

    def _info(self):
        features = datasets.Features(
            {
                "article": datasets.Value("string"),
                "question": datasets.Value("string"),
                "options": datasets.Sequence(datasets.Value("string")),
                "label": datasets.ClassLabel(names=_LABEL_NAMES),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage='https://github.com/nyu-mll/quality',
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract('QuALITY.v1.0.zip')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'QuALITY.v1.0.htmlstripped.train'),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, 'QuALITY.v1.0.htmlstripped.test'),
                            "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'QuALITY.v1.0.htmlstripped.dev'),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(
        self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""

        with open(filepath, encoding="utf-8") as f:
            count = 0
            for id_, row in enumerate(f):
                data = json.loads(row)
                for question in data['questions']:
                    count += 1
                    yield count, {
                        'article': data['article'],
                        'question': question['question'],
                        'options': question['options'],
                        'label': f"choice_{question['gold_label']}"
                    }
