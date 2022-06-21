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
"""MIMIC-III."""


import json

import datasets


_CITATION = """@article{johnson-mit-2016-mimic-iii, 
  author={Johnson, Alistair E W and Pollard, Tom J and Shen, Lu and Li-Wei, H Lehman and Feng, Mengling and Ghassemi, 
  Mohammad and Moody, Benjamin and Szolovits, Peter and Celi, Leo Anthony and Mark, Roger G}, 
  year={2016}, 
  title={{MIMIC-III, a freely accessible critical care database}}, 
  journal={Sci. Data}, 
  volume={3}, 
  url={https://www.nature.com/articles/sdata201635.pdf}
}"""

_DESCRIPTION = """MIMIC"""

_HOMEPAGE = "https://physionet.org/content/mimiciii/1.4/"

_LICENSE = "CC BY-SA (Creative Commons / Attribution-ShareAlike)"

_LABEL_NAMES = ['001-139', '140-239', '240-279', '280-289', '290-319', '320-389', '390-459', '460-519',
                '520-579', '580-629', '630-679', '680-709', '710-739', '740-759', '760-779', '780-799',
                '800-999', 'V01-V91', 'E000-E999']

class MIMIC(datasets.GeneratorBasedBuilder):
    """MIMIC"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="mimic", version=VERSION, description="MIMIC"
        ),
    ]

    DEFAULT_CONFIG_NAME = "mimic"

    def _info(self):
        features = datasets.Features(
            {
                "summary_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "labels": datasets.features.Sequence(datasets.ClassLabel(names=_LABEL_NAMES)),
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
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract('mimic.jsonl')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dir,
                            "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(
        self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if data['data_type'] == split:
                    yield id_, {
                        "summary_id": data["summary_id"],
                        "text": data['text'],
                        "labels": data["level_1"],
                    }