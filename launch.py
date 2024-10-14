import hydra


@hydra.main(config_name="main.yaml", config_path="./configs", version_base=None)
def main(cfg=None):
    from datetime import datetime
    import json
    import os
    from omegaconf import OmegaConf
    from src.train import train
    import gcsfs
    import jax
    import wandb
    from jax.experimental.multihost_utils import sync_global_devices

    print(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.multihost:
        jax.distributed.initialize()
    if cfg.rundir is None and not cfg.debug:
        assert not cfg.multihost, "Multihost must prespecify rundir."
        # TODO: use jax broadcast_one_to_all to sync this?
        cfg.rundir = os.path.join(
            "outputs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
    if jax.process_index() == 0:  # Wandb and cfg setup
        # read key from .env
        with open(".wandbkey") as f:
            wandb_key = f.read().strip()
        wandb.login(key=wandb_key)
        wandb_id = None
        cfg_dict = OmegaConf.to_object(cfg)
        if not cfg.debug:
            print(f"Writing to {cfg.rundir}")
            if cfg.rundir.startswith("gs://"):
                print("Using GCS filesystem")
                fs = gcsfs.GCSFileSystem()
                fopen, exists = fs.open, fs.exists
            else:
                print("Using local filesystem")
                cfg.rundir = os.path.abspath(cfg.rundir)
                fs, fopen, exists = os, open, os.path.exists

            # make sure the directory exists
            fs.makedirs(cfg.rundir, exist_ok=True)

            # write cfg as json
            with fopen(os.path.join(cfg.rundir, "cfg.json"), "w") as f:
                f.write(json.dumps(cfg_dict))

            # Load wandb id or write it, for proper wandb resuming.
            wandb_id_path = os.path.join(cfg.rundir, "wandb_id.txt")
            if exists(wandb_id_path):
                with fopen(wandb_id_path, "r") as f:
                    wandb_id = f.read()
            else:
                wandb_id = wandb.util.generate_id()
                with fopen(wandb_id_path, "w") as f:
                    f.write(wandb_id)
        wandb.init(
            project="ado", id=wandb_id, resume="allow", config=cfg_dict)
    if cfg.multihost:
        sync_global_devices("end_wandb_init")
    train(cfg)


if __name__ == "__main__":
    main()
