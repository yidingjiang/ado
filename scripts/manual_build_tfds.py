import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import argparse
import tensorflow_datasets as tfds
import src.tfds.the_pile_grouped as the_pile_grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_name', type=str, required=True)
    args = parser.parse_args()

    tfds.load(f"the_pile_grouped/{args.valid_name}", split="all")

    return


if __name__ == "__main__":
    main()