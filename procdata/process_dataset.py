"""Groups the data in The Pile into its different domains (by set name)."""
import argparse
import concurrent.futures
import os
import threading
import io
import tensorflow as tf
import zstandard
import simdjson as json
from tqdm import tqdm


PILE_SET_NAMES = [
    'FreeLaw', 'Enron Emails', 'Github', 'OpenSubtitles', 'PubMed Central', 'OpenWebText2', 'StackExchange',
    'Pile-CC', 'ArXiv', 'USPTO Backgrounds', 'Books3', 'Wikipedia (en)', 'PubMed Abstracts', 'NIH ExPorter',
    'BookCorpus2', 'EuroParl', 'HackerNews', 'DM Mathematics', 'YoutubeSubtitles', 'PhilPapers',
    'Ubuntu IRC', 'Gutenberg (PG-19)'
]


def write_to_file(items, fname):
    cctx = zstandard.ZstdCompressor()
    with tf.io.gfile.GFile(fname, 'wb') as f:
        writer_stream = cctx.stream_writer(f)
        for item in items:
            writer_stream.write(item)
        writer_stream.close()


class StoreAndFlush:
    def __init__(self, limit: int, name: str, worker_id: int):
        self.items = []
        self.size = 0
        self.limit = limit
        self.iter = 0
        self.name = name
        self.worker_id = worker_id
        self.lines_pushed = 0
        self.threads = []

    def push(self, line, size: int):
        self.items.append(line)
        self.size += size
        if self.size >= self.limit:
            self.flush()

    def flush(self):
        fname = os.path.join(self.name, f'worker{self.worker_id}_shard{self.iter}.jsonl.zst')
        thread = threading.Thread(target=write_to_file, args=(self.items.copy(), fname))
        thread.start()
        self.lines_pushed += len(self.items)
        self.size = 0
        self.items = []
        self.iter += 1
        self.threads.append(thread)
        self.threads = [t for t in self.threads if thread.is_alive()]

    def close(self):
        if len(self.items) > 0:
            thread = self.flush()
        for thread in self.threads:
            thread.join()


def read_files(paths, output_path, worker_id, size_per_file):
    trackers = {}
    for set_name in PILE_SET_NAMES:
        set_path = os.path.join(output_path, set_name)
        trackers[set_name] = StoreAndFlush(2 ** size_per_file, set_path, worker_id)

    parser = json.Parser()
    def json_parser(x):
        return parser.parse(x).as_dict()

    total_lines = 0
    for path in tqdm(paths):
        with tf.io.gfile.GFile(path, 'rb+') as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            for line in reader_stream:
                parsed_line = json_parser(line)
                set_name = parsed_line['meta']['pile_set_name']
                trackers[set_name].push(line, size=len(line))
                total_lines += 1
    lines_pushed = {}
    for name, tracker in trackers.items():
        tracker.close()
        if name not in lines_pushed:
            lines_pushed[name] = 0
        lines_pushed[name] += tracker.lines_pushed
    return total_lines, lines_pushed


def wrapper(args):
    return read_files(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="", required=True)
    parser.add_argument("--output_path", type=str, default="", required=True)
    parser.add_argument("--size_per_file", type=int, default=28)  # 2^N bytes per file (before compression)
    parser.add_argument("--num_workers", type=int, default=20)
    args = parser.parse_args()

    # make output directory
    if not tf.io.gfile.exists(args.output_path):
        tf.io.gfile.makedirs(args.output_path)

    for set_name in PILE_SET_NAMES:
        set_path = os.path.join(args.output_path, set_name)
        if not tf.io.gfile.exists(set_path):
            tf.io.gfile.makedirs(set_path)

    num_workers = args.num_workers
    paths = tf.io.gfile.glob(args.path)
    print(f"Processings {len(paths)} items with {num_workers} workers.")
    worker_args = []
    for i in range(num_workers):
        worker_paths = paths[i::num_workers]
        worker_args.append((worker_paths, args.output_path, i, args.size_per_file))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(wrapper, worker_args)
    global_lines_pushed = {}
    for i, (lines_read, lines_pushed) in enumerate(results):
        tot_lines_pushed = sum(lines_pushed.values())
        print(f"Worker {i} lines read: {lines_read}. lines pushed: {tot_lines_pushed}")
        for name, val in lines_pushed.items():
            if name not in global_lines_pushed:
                global_lines_pushed[name] = 0
            global_lines_pushed[name] += val
    print("Lines per set.")
    print(global_lines_pushed)


if __name__ == "__main__":
    main()
