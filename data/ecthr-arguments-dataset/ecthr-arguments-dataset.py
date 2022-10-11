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
"""ECtHRArguments"""

import json
import os
import textwrap

import datasets

MAIN_CITATION = """\
@article{Habernal.et.al.2022.arg,
      author = {Habernal, Ivan and Faber, Daniel and Recchia, Nicola and
                Bretthauer, Sebastian and Gurevych, Iryna and
                Spiecker genannt Döhmann, Indra and Burchard, Christoph}, 
      title = {{Mining Legal Arguments in Court Decisions}},
      journal = {arXiv preprint},
      year = {2022},
      doi = {10.48550/arXiv.2208.06178},
}"""

_DESCRIPTION = """\
The dataset contains approx. 300 cases from the European Court of Human Rights (ECtHR). For each case, the dataset provides 
a list of argumentative paragraphs from the case analysis. Spans in each paragraph has been labeled with one or more out 
of 13 argument types. We re-formulate this task, as a sequential paragraph classification task, where each paragraph is 
labelled with one or more labels. The input of the model is the list of paragraphs of a case, and the output is the set 
of relevant argument types per paragraph.
"""

ECTHR_ARG_TYPES = ['Application', 'Precedent', 'Proportionality', 'Decision',
                   'Legal Basis', 'Legitimate Purpose', 'Non Contestation']


class ECtHRArgumentsConfig(datasets.BuilderConfig):
    """BuilderConfig for ECtHRArguments."""

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
        """BuilderConfig for ECtHRArguments.

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
        super(ECtHRArgumentsConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
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
        ECtHRArgumentsConfig(
            name="ecthr-arguments-dataset",
            description=textwrap.dedent(
                """\
            The UKLEX dataset consists of UK laws that have been labeled with concepts.
            Given a document, the task is to predict its labels (concepts).
            """
            ),
            text_column="paragraphs",
            label_column="labels",
            label_classes=ECTHR_ARG_TYPES,
            multi_label=True,
            dev_column="dev",
            data_url=f"ecthr_arguments.tar.gz",
            data_file="ecthr_arguments.jsonl",
            url="https://github.com/trusthlt/mining-legal-arguments",
            citation=textwrap.dedent(
                """@article{Habernal.et.al.2022.arg,
                      author = {Habernal, Ivan and Faber, Daniel and Recchia, Nicola and
                                Bretthauer, Sebastian and Gurevych, Iryna and
                                Spiecker genannt Döhmann, Indra and Burchard, Christoph}, 
                      title = {{Mining Legal Arguments in Court Decisions}},
                      journal = {arXiv preprint},
                      year = {2022},
                      doi = {10.48550/arXiv.2208.06178},
                }"""
            ),
        )
    ]

    def _info(self):
        features = {"text": datasets.features.Sequence(datasets.Value("string")),
                    "labels": datasets.features.Sequence(datasets.features.Sequence(datasets.ClassLabel(names=self.config.label_classes)))}
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
            for id_, row in enumerate(f):
                data = json.loads(row)
                labels = [list(set(par_labels)).remove('O') if 'O' in par_labels else list(set(par_labels)) for
                          par_labels in data[self.config.label_column]]
                if data["data_type"] == split:
                    yield id_, {
                        "text": data[self.config.text_column],
                        "labels": labels,
                    }