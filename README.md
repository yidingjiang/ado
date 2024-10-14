# Adaptive Data Optimization

This repository contains the code for the paper "Adaptive Data Optimization: Dynamic Sample Selection with Scaling Laws".

The main algorithmic changes of the paper is contained in `src/data_selectors/ado.py`. The rest is standard language model training codebase and can be replaced by your favorite training framework as long as it supports sampling from different domains with different weights and has access to the loss of each domain.

## Launching experiments
This codebase is designed to run on GCP TPUs, with data stored in GCS buckets.

### Processing data (The Pile)
Note: we recommend you run the commands in this section on a GCP VM with at least 2TB of disk space.

By default, The Pile comes as .jsonl.zst files which contain examples from all domains mixed together.
We use `procdata/process_dataset.py` to separate examples out by domain:
```bash
python procdata/process_dataset.py --path /path/to/the/pile --output_path /outputs/go/here
```
This will produce a folder with 22 subdirectories, one per domain in The Pile. Each subdirectory should contain `.jsonl.zst` files
with examples for the corresponding domain.


Once done, we will materialize the data as a tensorflow dataset in a GCS bucket.
In `procdata/build_tfds_configs.sh`, set `local_base_path` to point to the outputs from your previous command and `remote_base_path`
to point to the desired location in your bucket, e.g., `gs://your-bucket/tfds/the_pile_grouped`. Run:
```bash
./procdata/build_tfds_configs.sh
```
This should materialize the TF dataset at `gs://your-bucket/tfds/the_pile_grouped`.

### Running experiments on TPUs
We assume you already have a TPU named `node1` running in the same region as your GCS bucket--see below for instructions on launching TPU VMs.

First, modify `scripts/tpu_commands.sh` to update `ado_project` [here](https://github.com/yidingjiang/ado/blob/main/scripts/tpu_commands.sh#L189) with your GCP project information:
overwrite the existing definitions of `tpu_project`, `tpu_zone`, and `tpu_gen`. Next, you need to specify the path to the dataset in `configs/experiments/*.yaml`.
Set `data_dir=gs://your-bucket/tfds/the_pile_grouped`.


Then, from your local machine, you can launch a run with:
```bash
echo '[your wandb key]' >> .wandbkey  # store your API key, which you can find at [wandb.ai/authorize](wandb.ai/authorize)
source scripts/tpu_commands.sh
tpu ado_project reset_key node1  # maybe necessary if you get weird SSH key issues
tpu ado_project setup node1  # copy code over, install deps, etc.

tpu ado_project cl "python ado/launch.py [OPTIONS]"  # copy and launch
```

Under hydra, here is how we configure the run:
```bash
python ado/launch.py +experiment=124_ado debug=true
```
See possible experiment configurations in `configs/experiment`. Set `debug=false` to enable checkpointing and larger sample size during evaluation.

## Creating/deleting TPUs
To create a new node called `node1`:

```bash
# Make sure you're in the right project
gcloud projects list
gcloud config set project [PROJECT]

ZONE=europe-west4-a  # change if your TPUs are in a different zone.
# Create
gcloud compute tpus tpu-vm create node1 \
--zone=$ZONE \
--accelerator-type=v3-128 \
--version=tpu-ubuntu2204-base \
--preemptible  # only if you want it to be preemptible
```
To delete the instance:
```bash
# Delete
gcloud compute tpus tpu-vm delete node1 --zone=$ZONE
```
### Adding your own dataset
1. Implement a new dataloader. See `src/dataloader/pile_loader.py` for an example of new dataloader. 
2. Add a new task configuration to `configs/loader/tasks_sampler_cfg` (see `configs/loader/tasks_sampler_cfg/pile_tasks.yaml` for an example).
3. Add a loader configuration to `configs/loader` (see `configs/loader/pile_natural.yaml` for an example).