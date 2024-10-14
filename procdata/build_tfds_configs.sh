#!/bin/bash

set -e

local_base_path= ...  # local strage where you want the config to be
remote_base_path= ...  # where the actual dataset is

datasets=(
    'arxiv'
    'bookcorpus2'
    'books3'
    'dm_mathematics'
    'enron_emails'
    'europarl'
    'freelaw'
    'github'
    'gutenberg_pg_19'
    'hackernews'
    'nih_exporter'
    'opensubtitles'
    'openwebtext2'
    'philpapers'
    'pile_cc'
    'pubmed_abstracts'
    'pubmed_central'
    'stackexchange'
    'uspto_backgrounds'
    'ubuntu_irc'
    'wikipedia_en'
    'youtubesubtitles'
)

# Loop over each dataset
for name in "${datasets[@]}"; do
    # Check if remote dataset exists
    if ! gsutil ls "${remote_base_path}/${name}" > /dev/null 2>&1; then
        echo "Building ${name}"
        # Build the dataset if it does not exist
        python scripts/manual_build_tfds.py --valid_name "$name"

        # Copy the built dataset to the remote storage
        gsutil -m cp -r "${local_base_path}/${name}" "${remote_base_path}"

        # Remove the local directory
        rm -rf "${local_base_path}/${name}"
    else
        echo "Dataset ${name} already exists in remote storage. Skipping..."
    fi
done
