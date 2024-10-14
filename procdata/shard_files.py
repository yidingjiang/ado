import argparse
import os
import concurrent.futures
import threading
import glob
import io
import tensorflow as tf
import zstandard


def write_to_file(items, fname):
    cctx = zstandard.ZstdCompressor()
    with tf.io.gfile.GFile(fname, 'wb') as f:
        writer_stream = cctx.stream_writer(f)
        for item in items:
            writer_stream.write(item)
        writer_stream.close()


def shard(args):
    path, out_path, worker_id = args
    shards_so_far = 0
    size_so_far = 0
    items = []
    threads = []
    with tf.io.gfile.GFile(path, 'rb+') as f:
        cctx = zstandard.ZstdDecompressor()
        reader_stream = io.BufferedReader(cctx.stream_reader(f))
        for line in reader_stream:
            items.append(line)
            size_so_far += len(line)
            if size_so_far >= 2 ** 28:
                fname = os.path.join(out_path, f'worker{worker_id}_shard{shards_so_far}.jsonl.zst')
                thread = threading.Thread(target=write_to_file, args=(items.copy(), fname))
                thread.start()
                threads.append(thread)
                items, size_so_far = [], 0
                shards_so_far += 1
    if len(items) > 0:
        fname = os.path.join(out_path, f'worker{worker_id}_shard{shards_so_far}.jsonl.zst')
        thread = threading.Thread(target=write_to_file, args=(items.copy(), fname))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="", required=True)
    parser.add_argument("--out_path", type=str, default="", required=True)
    args = parser.parse_args()

    tf.io.gfile.makedirs(args.out_path)
    paths = glob.glob(args.path)
    num_workers = min(100, len(paths))
    print(f"Processings {len(paths)} items with {num_workers} workers.")
    args = [(path, args.out_path, i) for i, path in enumerate(paths)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(shard, args)
    print([result for result in results]) 


if __name__ == "__main__":
    main()
