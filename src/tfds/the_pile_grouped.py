import dataclasses
import typing as tp
import io
import os
import simdjson as json
import tensorflow as tf
import zstandard
import jsonlines
import tensorflow_datasets as tfds
from transformers import AutoTokenizer
from etils import epath

from .conversion_utils import MultiThreadedDatasetBuilder


@dataclasses.dataclass
class ThePileGroupConfig(tfds.core.BuilderConfig):
  paths: tp.Dict[str, str] = dataclasses.field(default_factory=dict)


TRAIN_ROOT = ...  # path where the grouped pile train set is
VAL_ROOT = ...  # path where the grouped pile val set is
TEST_ROOT = ...  # path where the grouped pile test set is
TASKS = [
    "ArXiv",  # 0
    "BookCorpus2",  # 1
    "Books3",  # 2
    "DM Mathematics",  # 3
    "Enron Emails",  # 4
    "EuroParl",  # 5
    "FreeLaw",  # 6
    "Github",  # 7
    "Gutenberg (PG-19)",  # 8
    "HackerNews",  # 9
    "NIH ExPorter",  # 10
    "OpenSubtitles",  # 11
    "OpenWebText2",  # 12
    "PhilPapers",  # 13
    "Pile-CC",  # 14
    "PubMed Abstracts",  # 15
    "PubMed Central",  # 16
    "StackExchange",  # 17
    "USPTO Backgrounds",  #18
    "Ubuntu IRC",  # 19
    "Wikipedia (en)",  # 20
    "YoutubeSubtitles",  # 21
]

# [s.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower() for s in TASKS]
VALID_TASK_NAMES = [
    'arxiv',
    'bookcorpus2',
    'books3',
    'dm_mathematics',
    'enron_emails',
    'europarl',
    'freelaw',
    'github',
    'gutenberg_pg_19',
    'hackernews',
    'nih_exporter',
    'opensubtitles',
    'openwebtext2',
    'philpapers',
    'pile_cc',
    'pubmed_abstracts',
    'pubmed_central',
    'stackexchange',
    'uspto_backgrounds',
    'ubuntu_irc',
    'wikipedia_en',
    'youtubesubtitles'
]

TASK_TO_VALID_NAME = {k: v for k, v in zip(TASKS, VALID_TASK_NAMES)}
TASK_TO_AVG_LENGTH = {
    "ArXiv": 13527.87,
    "BookCorpus2": 90571.28,
    "Books3": 136710.2,
    "DM Mathematics": 3610.32,
    "Enron Emails": 391.11,
    "EuroParl": 22242.88,
    "FreeLaw": 3217.57,
    "Github": 2467.51,
    "Gutenberg (PG-19)": 105884.62,
    "HackerNews": 1092.34,
    "NIH ExPorter": 381.44,
    "OpenSubtitles": 9086.99,
    "OpenWebText2": 1230.31,
    "PhilPapers": 15188.79,
    "Pile-CC": 1263.04,
    "PubMed Abstracts": 273.83,
    "PubMed Central": 7762.79,
    "StackExchange": 603.58,
    "USPTO Backgrounds": 853.75,
    "Ubuntu IRC": 87724.2,
    "Wikipedia (en)": 860.59,
    "YoutubeSubtitles": 4456.02,
}

LINE_PER_TASK = {
    'FreeLaw': 5069088,
    'Enron Emails': 926132,
    'GitHub': 18044218,
    'OpenSubtitles': 632485,
    'PubMed Central': 5679903,
    'OpenWebText2': 32333654,
    'StackExchange': 29529008,
    'Pile-CC': 52441354,
    'ArXiv': 2377741,
    'USPTO Backgrounds': 11123325,
    'Books3': 277655,
    'Wikipedia (en)': 16939503,
    'PubMed Abstracts': 29329202,
    'NIH ExPorter': 1777926,
    'BookCorpus2': 25355,
    'EuroParl': 131723,
    'HackerNews': 1571968,
    'DM Mathematics': 1918535,
    'YoutubeSubtitles': 328030,
    'PhilPapers': 63875,
    'Ubuntu IRC': 20067,
    'Gutenberg (PG-19)': 66981
}

_CONFIGS = []
for task, valid_name in TASK_TO_VALID_NAME.items():
    task_paths = {
        "train": list(epath.Path(os.path.join(TRAIN_ROOT, task)).glob("*.jsonl.zst")),
        "val": list(epath.Path(os.path.join(VAL_ROOT, task)).glob("*.jsonl.zst")),
        "test": list(epath.Path(os.path.join(TEST_ROOT, task)).glob("*.jsonl.zst")),
    }
    _CONFIGS.append(ThePileGroupConfig(
        paths=task_paths,
        name=valid_name,
        description=task,
    ))


def _generate_examples(paths: tp.List[epath.Path]):
    parser = json.Parser()
    def json_parser(x):
        return parser.parse(x).as_dict()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    for path in paths:
        with tf.io.gfile.GFile(path, 'rb+') as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream, loads=json_parser)
            for i, item in enumerate(reader):
                key = str(path) + f"-{i}"
                text = item['text']
                yield key, {'tokens': tokenizer.encode(text)}


class ThePileGrouped(MultiThreadedDatasetBuilder):
    VERSION = tfds.core.Version('2.0.0')
    PARSE_FCN = _generate_examples
    BUILDER_CONFIGS = _CONFIGS

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'tokens': tfds.features.Tensor(shape=(None,), dtype=tf.int32),
            }),
            supervised_keys=("tokens", "tokens"),
        )

    def _split_paths(self):
        return self.builder_config.paths
