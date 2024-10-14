import argparse
import io
import tensorflow as tf
import zstandard
import jsonlines
import simdjson as json

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="", required=True)
args = parser.parse_args()

parser = json.Parser()
def json_parser(x):
    return parser.parse(x).as_dict()

unique_sets = set()
# .jsonl.zst file
with tf.io.gfile.GFile(args.path, 'rb+') as f:
    cctx = zstandard.ZstdDecompressor()
    reader_stream = io.BufferedReader(cctx.stream_reader(f))
    reader = jsonlines.Reader(reader_stream, loads=json_parser)
    for line in reader:
        print(line)
        # wait for input to continue
        input("Press Enter to continue...")