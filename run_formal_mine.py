"""

Usage:
  # Default fusion mode (hybrid, alpha=0.5)
  python run_formal_mine.py

  # Specify fusion parameters
  python run_formal_mine.py --fuse_method multiply --fuse_alpha 0.3

  # No fusion (equivalent to the original run_formal.py)
  python run_formal_mine.py --fuse_method none

  # Ablation experiments (--ablation overrides the ablation config in YAML)
  python run_formal_mine.py --ablation static_global_weights
  python run_formal_mine.py --ablation no_matching_loss
  python run_formal_mine.py --ablation no_variances_loss
  python run_formal_mine.py --ablation no_prototype_layer

"""

import sys
import os
import argparse
import numpy as np
from omegaconf import OmegaConf
from itertools import product
from exp_mine import ColaxPipline

from txai.synth_data.synth_data_base import SynthTrainDataset

COLAX_BETA_ROOT = "./"


def parse_param_string(param_value):
    if isinstance(param_value, int):
        return [param_value]
    if not isinstance(param_value, str):
        try:
            return [int(param_value)]
        except (ValueError, TypeError):
            return [param_value]
    param_str = str(param_value).strip()
    result = []
    parts = param_str.split(',')
    for part in parts:
        part = part.strip()
        if ':' in part:
            range_parts = part.split(':')
            if len(range_parts) == 2:
                start, end = int(range_parts[0]), int(range_parts[1])
                result.extend(range(start, end + 1))
            elif len(range_parts) == 3:
                start, end, step = int(range_parts[0]), int(range_parts[1]), int(range_parts[2])
                result.extend(range(start, end + 1, step))
        else:
            result.append(int(part))
    return sorted(list(set(result)))


def load_config(config_path=None, fuse_method='hybrid', fuse_alpha=0.5):

    if config_path is None:
        config_path = f"{COLAX_BETA_ROOT}/configs/run_configs.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    run_cfg = OmegaConf.load(config_path)
    defaults = run_cfg.get('defaults', {})
    datasets_str = str(defaults.get('datasets', ''))
    datasets = [d.strip() for d in datasets_str.split(",") if d.strip()]
    if not datasets:
        raise ValueError("No datasets specified in config file.")

    configs = []
    for dataset_name in datasets:
        ds_cfg_path = f"{COLAX_BETA_ROOT}/configs/{dataset_name}.yaml"
        cfg = {
            "base": {"root_dir": COLAX_BETA_ROOT},
            "dataset": {"name": dataset_name, "meta_dataset": "default"},
        }
        if os.path.exists(ds_cfg_path):
            ds_cfg = OmegaConf.load(ds_cfg_path)
            if dataset_name in ds_cfg:
                cfg[dataset_name] = dict(ds_cfg[dataset_name])
            elif 'dataset' in ds_cfg:
                cfg["dataset"] = dict(ds_cfg["dataset"])
                if dataset_name in ds_cfg:
                    cfg[dataset_name] = dict(ds_cfg[dataset_name])
        else:
            print(f"Warning: Dataset config {ds_cfg_path} not found. Using defaults.")
            cfg[dataset_name] = {
                "num_classes": 2, "seq_len": 500, "proto_len": 30,
                "num_prototypes": 1, "num_prototypes_local": 1, "num_prototypes_global": 1,
            }

        if dataset_name in cfg:
            ds_block = cfg[dataset_name]
            if 'num_prototypes' in ds_block and 'num_prototypes_local' not in ds_block:
                ds_block['num_prototypes_local'] = ds_block['num_prototypes']
            if 'num_prototypes' in ds_block and 'num_prototypes_global' not in ds_block:
                ds_block['num_prototypes_global'] = ds_block['num_prototypes']
            if 'num_prototypes_local' not in ds_block:
                ds_block['num_prototypes_local'] = ds_block.get('num_prototypes', 1)
            if 'num_prototypes_global' not in ds_block:
                ds_block['num_prototypes_global'] = ds_block.get('num_prototypes', 1)

        ds_block = cfg[dataset_name]
        yaml_ablation = ds_block.get('ablation', 'none') if hasattr(ds_block, 'get') else 'none'
        proto_len_values = parse_param_string(ds_block.get('proto_len', 30))
        num_prototypes_local_values = parse_param_string(ds_block.get('num_prototypes_local', 1))
        num_prototypes_global_values = parse_param_string(ds_block.get('num_prototypes_global', 1))

        combinations = list(product(
            proto_len_values, num_prototypes_local_values, num_prototypes_global_values
        ))

        print(f"  Generated {len(combinations)} combinations for {dataset_name}:")
        print(f"    proto_len: {proto_len_values}")
        print(f"    num_prototypes_local: {num_prototypes_local_values}")
        print(f"    num_prototypes_global: {num_prototypes_global_values}")
        print(f"    fuse_method: {fuse_method}, fuse_alpha: {fuse_alpha}")
        print(f"    ablation (YAML): {yaml_ablation}")

        for proto_len, n_local, n_global in combinations:
            combo_cfg = {
                "base": {"root_dir": COLAX_BETA_ROOT},
                "dataset": {"name": dataset_name, "meta_dataset": "default"},
                dataset_name: dict(ds_block)
            }
            combo_cfg[dataset_name]['proto_len'] = proto_len
            combo_cfg[dataset_name]['num_prototypes_local'] = n_local
            combo_cfg[dataset_name]['num_prototypes_global'] = n_global
            combo_cfg[dataset_name]['num_prototypes'] = n_local

            configs.append({
                'config': OmegaConf.create(combo_cfg),
                'dataset_name': dataset_name,
                'do_train': bool(defaults.get('train', True)),
                'do_test': bool(defaults.get('test', True)),
                'ablation': yaml_ablation,  
                'combo': {
                    'proto_len': proto_len,
                    'num_prototypes_local': n_local,
                    'num_prototypes_global': n_global,
                }
            })
    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Colax Experiment Runner — with Global-Local Fusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to main configuration YAML file')
    parser.add_argument('--fuse_method', type=str, default='hybrid',
                        choices=['multiply', 'add', 'gate', 'hybrid', 'none'],
                        help='Fusion method for global-local knowledge (default: hybrid)')
    parser.add_argument('--fuse_alpha', type=float, default=0.5,
                        help='Global weight coefficient (default: 0.5)')
    parser.add_argument('--diagnose', action='store_true',
                        help='Run diagnosis on the first sample before full eval')
    parser.add_argument('--ablation', type=str, default=None,
                        help='ablation setting'
                             'none, static_global_weights, no_matching_loss, '
                             'no_variances_loss, no_prototype_layer '
                             )

    args = parser.parse_args()

    try:
        configs = load_config(args.config, args.fuse_method, args.fuse_alpha)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    print("=" * 80)
    print("Colax Experiment Runner — Global-Local Fusion")
    print(f"  Fuse method: {args.fuse_method}, Alpha: {args.fuse_alpha}")
    if args.ablation is not None:
        print(f"  Ablation (CLI override): {args.ablation}")
    print("=" * 80)
    print(f"Total number of experiment combinations: {len(configs)}")
    print()

    for idx, cfg_info in enumerate(configs, 1):
        dataset_name = cfg_info['dataset_name']
        config = cfg_info['config']
        do_train = cfg_info['do_train']
        do_test = cfg_info['do_test']
        combo = cfg_info.get('combo', {})
        yaml_ablation = cfg_info.get('ablation', 'none')

        if args.ablation is not None:
            final_ablation = args.ablation
        else:
            final_ablation = yaml_ablation if yaml_ablation else 'none'

        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(configs)}] Processing: {dataset_name}")
        if combo:
            print(f"  Combination: proto_len={combo['proto_len']}, "
                  f"n_local={combo['num_prototypes_local']}, "
                  f"n_global={combo['num_prototypes_global']}")
        print(f"{'=' * 80}")
        print(f"Train: {do_train}, Test: {do_test}")
        print(f"Ablation: {final_ablation} (CLI: {args.ablation}, YAML: {yaml_ablation})")

        if dataset_name in config:
            ds_cfg = config[dataset_name]
            print(f"Configuration:")
            print(f"  num_classes: {ds_cfg.get('num_classes', 'N/A')}")
            print(f"  seq_len: {ds_cfg.get('seq_len', 'N/A')}")
            print(f"  proto_len: {ds_cfg.get('proto_len', 'N/A')}")
        print()

        try:
            saved_argv = sys.argv
            sys.argv = [sys.argv[0]]

            pipeline = ColaxPipline(config)
            sys.argv = saved_argv


            pipeline.args.ablation = final_ablation

            if do_train:
                pipeline.args.is_training = 1
                pipeline.train_colax()

            if do_test:

                from colax import ScoreSubsequencesMine
                from get_data import get_saliency_data
                from txai.utils.evaluation import ground_truth_xai_eval

                pipeline.args.saliency = True

                pipeline.args.fuse_method = args.fuse_method
                pipeline.args.fuse_alpha = args.fuse_alpha

                seg_model = pipeline.load_seg_model()
                X, y, times, gt_exps = get_saliency_data(pipeline.args, return_dict=False)
                X, y, times, gt_exps = X.cpu(), y.cpu(), times.cpu(), gt_exps.cpu()
                class_model = pipeline.get_class_model(X)

                Scorer = ScoreSubsequencesMine(pipeline.args, class_model, seg_model.cpu())
                score = Scorer.get_score_as_GTEXP(X)

                results_dict = ground_truth_xai_eval(score, gt_exps)
                for k, v in results_dict.items():
                    print('\t{} = {:.4f} +- {:.4f}'.format(
                        k, np.mean(v), np.std(v) / np.sqrt(len(v))))

                with open("results.txt", "a") as f:
                    npro_local = getattr(pipeline.args, 'num_prototypes_local',
                                         getattr(pipeline.args, 'num_prototypes', 1))
                    npro_global = getattr(pipeline.args, 'num_prototypes_global',
                                          getattr(pipeline.args, 'num_prototypes', 1))
                    ablation_str = ""
                    if hasattr(pipeline.args, 'ablation') and pipeline.args.ablation and pipeline.args.ablation != 'none':
                        ablation_str = f"_ablation{pipeline.args.ablation}"
                    f.write(
                        f"{pipeline.args.data}_slen{pipeline.args.seq_len}"
                        f"_npro_local{npro_local}_npro_global{npro_global}"
                        f"_prototype_len{pipeline.args.prototype_len}"
                        f"_fuse_{args.fuse_method}_alpha{args.fuse_alpha}{ablation_str}"
                    )
                    for k, v in results_dict.items():
                        f.write(f' {k}:{np.mean(v):.4f}-{(np.std(v) / np.sqrt(len(v))):.4f}')
                    f.write("\n")

            combo_str = ""
            if combo:
                combo_str = (f" (proto_len={combo['proto_len']}, "
                             f"n_local={combo['num_prototypes_local']}, "
                             f"n_global={combo['num_prototypes_global']})")
            print(f"\n✓ Successfully completed {dataset_name}{combo_str}")

        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print("All experiments completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
