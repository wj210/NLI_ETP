import os
import json
import datasets
from pathlib import Path


_DESCRIPTION = "nli rationale dataset"
_DOCUMENT = "text"
_ID = "id"
_LABEL = 'label'


class NliDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _DOCUMENT: datasets.Value("string"),
                    _ID: datasets.Value("string"),
                    _LABEL: datasets.Value("int32"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager._data_dir
        dataset = data_dir.split("/")[-1]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_dir, "nli_train.jsonl"), "name": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_dir, "nli_val.jsonl"), "name": "validation"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(data_dir, "nli_test.jsonl"), "name": "test"}
            )
        ]

    def _generate_examples(self, path=None, name=None):
        """Yields examples."""
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                id = str(x["id"])
                item = {
                    _ID: id,
                    _DOCUMENT: x["text"],
                    _LABEL: x["label"],
                }
                yield id, item
