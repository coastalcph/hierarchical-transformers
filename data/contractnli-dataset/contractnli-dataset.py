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
"""ContractNLI"""

import json
import os
import textwrap

import datasets

MAIN_CITATION = """\
@inproceedings{koreeda-manning-2021-contractnli-dataset,
        title = "{C}ontract{NLI}: A Dataset for Document-level Natural Language Inference for Contracts",
        author = "Koreeda, Yuta  and
          Manning, Christopher",
        booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
        month = nov,
        year = "2021",
        address = "Punta Cana, Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.findings-emnlp.164",
        doi = "10.18653/v1/2021.findings-emnlp.164",
        pages = "1907--1919",
}"""

_DESCRIPTION = """\
The ContractNLI dataset consists of Non-Disclosure Agreements (NDAs). All NDAs have been labeled based 
on several hypothesis templates as entailment, neutral or contradiction. In this version of the task
(Task B), the input consists of the full document.
"""

LABELS = ["contradiction", "entailment", "neutral"]


class ContractNLIConfig(datasets.BuilderConfig):
    """BuilderConfig for ContractNLI."""

    def __init__(
        self,
        text_column,
        label_column,
        url,
        data_url,
        data_file,
        citation,
        label_classes=None,
        multi_label=None,
        dev_column="dev",
        **kwargs,
    ):
        """BuilderConfig for ContractNLI.

        Args:
          text_column: ``string`, name of the column in the jsonl file corresponding
            to the text
          label_column: `string`, name of the column in the jsonl file corresponding
            to the label
          url: `string`, url for the original project
          data_url: `string`, url to download the zip file from
          data_file: `string`, filename for data set
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          multi_label: `boolean`, True if the task is multi-label
          dev_column: `string`, name for the development subset
          **kwargs: keyword arguments forwarded to super.
        """
        super(ContractNLIConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_column = text_column
        self.label_column = label_column
        self.label_classes = label_classes
        self.multi_label = multi_label
        self.dev_column = dev_column
        self.url = url
        self.data_url = data_url
        self.data_file = data_file
        self.citation = citation


class LexGLUE(datasets.GeneratorBasedBuilder):
    """LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. Version 1.0"""

    BUILDER_CONFIGS = [
        ContractNLIConfig(
            name="contractnli",
            description=textwrap.dedent(
                """\
            The ContractNLI dataset consists of Non-Disclosure Agreements (NDAs). All NDAs have been labeled based 
            on several hypothesis templates as entailment, neutral or contradiction. In this version of the task
            (Task B), the input consists of the full document.
            """
            ),
            text_column="premise",
            label_column="label",
            label_classes=LABELS,
            multi_label=False,
            dev_column="dev",
            data_url="contract_nli.zip",
            data_file="contract_nli_long.jsonl",
            url="https://stanfordnlp.github.io/contract-nli/",
            citation=textwrap.dedent(
                """\
            @inproceedings{koreeda-manning-2021-contractnli-dataset,
                    title = "{C}ontract{NLI}: A Dataset for Document-level Natural Language Inference for Contracts",
                    author = "Koreeda, Yuta  and
                      Manning, Christopher",
                    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
                    month = nov,
                    year = "2021",
                    address = "Punta Cana, Dominican Republic",
                    publisher = "Association for Computational Linguistics",
                    url = "https://aclanthology.org/2021.findings-emnlp.164",
                    doi = "10.18653/v1/2021.findings-emnlp.164",
                    pages = "1907--1919",
                }
            }"""
            ),
        )
    ]

    def _info(self):
        features = {"premise": datasets.Value("string"), "hypothesis": datasets.Value("string"),
                    'label': datasets.ClassLabel(names=LABELS)}
        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + MAIN_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(self.config.data_url)
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
                gen_kwargs={
                    "filepath": os.path.join(data_dir, self.config.data_file),
                    "split": self.config.dev_column,
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, "r", encoding="utf-8") as f:
            sid = -1
            for id_, row in enumerate(f):
                data = json.loads(row)
                if data["subset"] == split:
                    for sample in data['hypothesises/labels']:
                        sid += 1
                        yield sid, {
                            "premise": data["premise"],
                            "hypothesis": sample['hypothesis'],
                            "label": sample['label'],
                        }