import sys
import argparse
from omegaconf import OmegaConf

from exp_saliency_general_hybrid import ColaxPipline
from txai.synth_data.synth_data_base import SynthTrainDataset


BETA_ROOT = "./"


def build_config(dataset_name: str, num_classes: int = 4, seq_len: int = 500, proto_len: int = 30,
                 num_prototypes_local: int = None, num_prototypes_global: int = None):
    """
    Construct a config object compatible with exp_saliency_general.get_args.
    The dataset-specific defaults can be overridden by CLI flags that
    exp_saliency_general.get_args will parse from sys.argv.
    """
    # Each dataset must be an attribute on the config with (num_classes, seq_len, proto_len)
    cfg = {
        "base": {"root_dir": BETA_ROOT},
        "dataset": {"name": dataset_name, "meta_dataset": "default"},
        dataset_name: {
            "num_classes": num_classes,
            "seq_len": seq_len,
            "proto_len": proto_len,
        },
    }
    # Add prototype counts if provided
    if num_prototypes_local is not None:
        cfg[dataset_name]["num_prototypes"] = num_prototypes_local  # For backward compatibility
        cfg[dataset_name]["num_prototypes_local"] = num_prototypes_local
    if num_prototypes_global is not None:
        cfg[dataset_name]["num_prototypes_global"] = num_prototypes_global
    return OmegaConf.create(cfg)


def main():
    # Parse command line arguments for prototype counts
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--num_prototypes_local', type=int, default=None,
                        help='Number of local prototypes')
    parser.add_argument('--num_prototypes_global', type=int, default=None,
                        help='Number of global prototypes')
    args_cli, unknown = parser.parse_known_args()  # Use parse_known_args to allow other args to pass through

    # Load run params from YAML (only three keys)
    run_cfg = OmegaConf.load(f"{BETA_ROOT}/configs/run_configs.yaml").defaults
    datasets = str(run_cfg.datasets)
    do_train = bool(run_cfg.train)
    do_test = bool(run_cfg.test)

    ptototype_len = str(run_cfg.prototype_len)


    dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
    prototype_len_list = [pl.strip() for pl in ptototype_len.split(",") if pl.strip()]



    for ds in dataset_list:
        print(f"===== Running dataset: {ds} =====")
        config = build_config(ds,
                              num_prototypes_local=args_cli.num_prototypes_local,
                              num_prototypes_global=args_cli.num_prototypes_global)
        # prevent downstream arg parse from reading CLI
        saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        pipeline = ColaxPipline(config)
        sys.argv = saved_argv

        # Train
        if do_train:
            # get_args() inside pipeline will read is_training from CLI; ensure it is set
            # If not provided by user, training flag here implies is_training=1
            # Users can still override via CLI (e.g., --is_training 0)
            pipeline.args.is_training = 1
            pipeline.train_colax()

        # Evaluate
        if do_test:
            pipeline.args.saliency = False
            pipeline.eval_colax()


if __name__ == "__main__":
    main()


