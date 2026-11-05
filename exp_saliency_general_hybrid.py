"""Colax experiment pipeline: training classification explainer and saliency evaluation.

This module defines the pipeline to train the ProtoPTST explainer and evaluate
saliency explanations against ground truth for supported datasets.
"""

import os
import time
import numpy as np
import torch
from torch import nn, optim
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import argparse
from torch.utils.data import DataLoader
import sys
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.encoders.simple import CNN, LSTM
from txai.utils.evaluation import ground_truth_xai_eval
from utils_training import EarlyStopping
from get_data import get_saliency_data
# from shapelet_encoder.models import ProtoPTST
from shapelet_encoder.models import colax_fo as ProtoPTST
# Add parent directory to sys.path for local imports
from colax import ScoreSubsequences
from tensorboardX import SummaryWriter


class Exp_Basic(object):
    """Base experiment wrapper providing device setup and model interface."""

    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "ProtoPTST": ProtoPTST,  # ProtoPTST

        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """Acquire compute device (GPU/CPU) based on configuration."""
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class Exp_Classification(Exp_Basic):
    """Classification experiment for training ProtoPTST explainer."""

    def __init__(self, args, data_dict):
        self.data_dict = data_dict
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa
        # TensorBoard writer
        self.writer = None
        if SummaryWriter is not None:
            print("Initialize TensorBoard SummaryWriter")
            runs_root = os.path.join(self.args.root_path or '.', 'runs')
            os.makedirs(runs_root, exist_ok=True)
            run_name = f"{self.args.dataset_name}_{time.strftime('%Y%m%d-%H%M%S')}"
            self.writer = SummaryWriter(logdir=os.path.join(runs_root, run_name))

    def _build_model(self):
        """Build explainer model and set basic args from data."""
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(self.data_dict, flag="TEST")
        self.args.seq_len = self.args.seq_len  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = self.args.enc_in  # redefine enc_in
        self.args.num_class = self.args.num_classes
        if self.args.data == "mitecg":
            self.args.num_class = 2

        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, data_dict, flag):
        """Return dataset and dataloader for the requested split flag."""

        if self.args.meta_dataset == 'ucr':
            data_set = data_dict[2]
            data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True)
            return data_set, data_loader
        else:
            random.seed(self.args.seed)
            data_set = data_dict[flag]
            data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True)
            return data_set, data_loader

    def _select_optimizer(self):
        """Create optimizer for training."""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Classification loss function."""
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """Validate on a data split and compute metrics dictionary."""
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    model_outputs = self.swa_model(batch_x, padding_mask, None, None)
                    # SWA model might return 3, 4, or 6 values, extract outputs
                    outputs = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
                else:
                    # Handle old model (3 returns), hybrid model (4 returns), and hybrid_new (6 returns)
                    model_outputs = self.model(batch_x, padding_mask, None, None)
                    # if len(model_outputs) == 7:
                    #     outputs, place_holder, _, _, _, _,_ = model_outputs
                    if len(model_outputs) == 6:
                        outputs, place_holder, _, _, _, _ = model_outputs
                    elif len(model_outputs) == 4:
                        outputs, place_holder, _, _ = model_outputs
                    else:
                        outputs, place_holder, _ = model_outputs

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes)
        # Guard against NaN/Inf from unstable logits
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        # Re-normalize rows to sum 1 after nan_to_num
        row_sums = probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs = probs / row_sums

        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        probs_np = probs.cpu().numpy()
        trues_np = trues.flatten().cpu().numpy()

        metrics_dict = {
            "Accuracy": accuracy_score(trues_np, predictions),
            "Precision": precision_score(trues_np, predictions, average="macro", zero_division=0),
            "Recall": recall_score(trues_np, predictions, average="macro", zero_division=0),
            "F1": f1_score(trues_np, predictions, average="macro", zero_division=0),
        }

        unique_labels = np.unique(trues_np)
        if unique_labels.size < 2:
            print("[vali] Only one class present in y_true; set AUROC/AUPRC to 0 for this split.")
            metrics_dict["AUROC"] = 0.0
            metrics_dict["AUPRC"] = 0.0
        else:
            trues_onehot = (
                torch.nn.functional.one_hot(
                    torch.tensor(trues_np, dtype=torch.long),
                    num_classes=self.args.num_class,
                )
                .float()
                .cpu()
                .numpy()
            )
            # If probs still contains invalid numbers, skip AUROC/AUPRC to avoid crash
            if not np.isfinite(probs_np).all():
                print("[vali] probs contain non-finite values; setting AUROC/AUPRC to 0.0")
                metrics_dict["AUROC"] = 0.0
                metrics_dict["AUPRC"] = 0.0
            else:
                try:
                    metrics_dict["AUROC"] = roc_auc_score(trues_onehot, probs_np, multi_class="ovr")
                except Exception as e:
                    print(f"[vali] AUROC failed: {e}; set to 0.0")
                    metrics_dict["AUROC"] = 0.0
                try:
                    metrics_dict["AUPRC"] = average_precision_score(trues_onehot, probs_np, average="macro")
                except Exception as e:
                    print(f"[vali] AUPRC failed: {e}; set to 0.0")
                    metrics_dict["AUPRC"] = 0.0

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self):
        """Train explainer model with early stopping and optional SWA updates."""
        train_data, train_loader = self._get_data(self.data_dict, flag="TRAIN")
        vali_data, vali_loader = self._get_data(self.data_dict, flag="VAL")
        test_data, test_loader = self._get_data(self.data_dict, flag="TEST")
        path = self.args.saving_path
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        global_step = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # Debug: Check first batch of first epoch
                if epoch == 0 and i == 0:
                    print(f"\n[DEBUG] First batch info:")
                    print(f"  batch_x shape: {batch_x.shape}, range: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
                    print(
                        f"  label shape: {label.shape}, unique: {label.unique()}, counts: {[(l.item(), (label == l).sum().item()) for l in label.unique()]}")

                # Handle both old model (3 returns), hybrid model (4 returns), and hybrid_new (6 returns)
                model_outputs = self.model(batch_x, padding_mask, None, None)
                # if len(model_outputs) == 7:
                #     # ProtoPTST_hybrid_new: output, activations_local, prototype_local, memory_attn, z_global, global_prototypes
                #     outputs, place_holder_1, prototype, memory_attn, z_global, global_prototypes, gate_weights = model_outputs
                #     has_alignment_loss = True
                if len(model_outputs) == 6:
                    # ProtoPTST_hybrid_new: output, activations_local, prototype_local, memory_attn, z_global, global_prototypes
                    outputs, place_holder_1, prototype, memory_attn, z_global, global_prototypes = model_outputs
                    has_alignment_loss = True
                elif len(model_outputs) == 4:
                    # ProtoPTST_hybrid: output, activations_local, prototype_local, memory_attn
                    outputs, place_holder_1, prototype, memory_attn = model_outputs
                    has_alignment_loss = False
                else:
                    # Original ProtoPTST: output, activations, prototype
                    outputs, place_holder_1, prototype = model_outputs
                    has_alignment_loss = False

                # place_holder_1——>activation (local prototypes)
                prototype_loss = self.model.seg_prototype_loss(place_holder_1, batch_x, prototype, outputs)

                global_seg_loss = self.model.seg_global_prototype_loss(memory_attn, batch_x, global_prototypes, outputs)
                # Compute global prototype alignment loss if available
                alignment_loss = 0.0
                if has_alignment_loss and hasattr(self.model, 'global_prototype_alignment_loss'):
                    # Use InfoNCE-style alignment loss
                    alignment_loss = self.model.global_prototype_alignment_loss(
                        z_global, global_prototypes, memory_attn, temperature=getattr(self.args, 'alignment_temperature', 0.07)
                    )

                    # Get alignment loss weight from args, default to 0.1
                    alignment_weight = getattr(self.args, 'alignment_loss_weight', 0.1)

                # global_diversity_loss = self.model.global_prototype_diversity_loss_margin(global_prototypes)
                global_diversity_loss = self.model.global_prototype_diversity_loss(global_prototypes, method='cosine')
                #local_diversity_loss = self.model.local_prototype_diversity_loss(prototype, place_holder_1, method='activation_cosine')

                # Debug: Check model outputs
                if epoch == 0 and i == 0:
                    print(f"  outputs shape: {outputs.shape}, range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                    probs = torch.softmax(outputs, dim=1)
                    preds = probs.argmax(dim=1)
                    print(f"  predictions: {preds}")
                    print(f"  true labels: {label}")
                    print(f"  initial accuracy: {(preds == label).float().mean().item():.4f}")
                    print(f"  prototype_loss: {prototype_loss.item() if hasattr(prototype_loss, 'item') else prototype_loss}")
                    print(f"  global_diversity_loss: {global_diversity_loss.item() if hasattr(global_diversity_loss, 'item') else global_diversity_loss}")
                    if has_alignment_loss:
                        print(f"  z_global shape: {z_global.shape}, global_prototypes shape: {global_prototypes.shape}")
                        print(f"  alignment_loss: {alignment_loss.item() if hasattr(alignment_loss, 'item') else alignment_loss}")

                global place_holder
                place_holder = place_holder_1
                ce_loss = criterion(outputs, label)  # cls
                
                # Combine losses
                loss = ce_loss + 0.5 * prototype_loss

                if has_alignment_loss and alignment_loss != 0.0:
                    alignment_weight = getattr(self.args, 'alignment_loss_weight', 0.1)
                    loss = loss + alignment_weight * alignment_loss

                loss = loss + 1e-3 * global_diversity_loss + 2e-3 * global_seg_loss


                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("prototype_loss:",
                          prototype_loss.item() if hasattr(prototype_loss, 'item') else prototype_loss, "ce_loss:",
                          ce_loss.item(), end="")
                    print(
                        f"  global_diversity_loss: {global_diversity_loss.item() if hasattr(global_diversity_loss, 'item') else global_diversity_loss}")
                    # if has_alignment_loss and alignment_loss != 0.0:
                    if has_alignment_loss:
                        print(f", alignment_loss: {alignment_loss.item() if hasattr(alignment_loss, 'item') else alignment_loss}")
                    else:
                        print()
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # Check gradient norms before clipping
                if epoch == 0 and i == 0:
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"  gradient norm before clip: {total_norm:.4f}")

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                if self.writer is not None:
                    self.writer.add_scalar('train/total_loss', float(loss.item()), global_step)
                    self.writer.add_scalar('train/ce_loss', float(ce_loss.item()), global_step)
                    self.writer.add_scalar('train/prototype_loss', float(prototype_loss.item()), global_step)
                    if has_alignment_loss and alignment_loss != 0.0:
                        self.writer.add_scalar('train/alignment_loss', float(alignment_loss.item() if hasattr(alignment_loss, 'item') else alignment_loss), global_step)
                global_step += 1

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # 计算训练集的ACC和AUPRC（用于判断过拟合）
            train_loss_metrics, train_metrics_dict = self.vali(train_data, train_loader, criterion)

            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            if self.writer is not None:
                # epoch-level scalars
                self.writer.add_scalar('epoch/train_loss', float(train_loss), epoch)
                self.writer.add_scalar('epoch/val_loss', float(vali_loss), epoch)
                self.writer.add_scalar('epoch/test_loss', float(test_loss), epoch)

                # Training metrics (for overfitting detection)
                self.writer.add_scalar('epoch/train_acc', float(train_metrics_dict['Accuracy']), epoch)
                self.writer.add_scalar('epoch/train_auprc', float(train_metrics_dict['AUPRC']), epoch)
                self.writer.add_scalar('epoch/train_auroc', float(train_metrics_dict['AUROC']), epoch)
                self.writer.add_scalar('epoch/train_f1', float(train_metrics_dict['F1']), epoch)

                self.writer.add_scalar('epoch/val_acc', float(val_metrics_dict['Accuracy']), epoch)
                self.writer.add_scalar('epoch/val_precision', float(val_metrics_dict['Precision']), epoch)
                self.writer.add_scalar('epoch/val_recall', float(val_metrics_dict['Recall']), epoch)
                self.writer.add_scalar('epoch/val_f1', float(val_metrics_dict['F1']), epoch)
                self.writer.add_scalar('epoch/val_auroc', float(val_metrics_dict['AUROC']), epoch)
                self.writer.add_scalar('epoch/val_auprc', float(val_metrics_dict['AUPRC']), epoch)

                self.writer.add_scalar('epoch/test_acc', float(test_metrics_dict['Accuracy']), epoch)
                self.writer.add_scalar('epoch/test_precision', float(test_metrics_dict['Precision']), epoch)
                self.writer.add_scalar('epoch/test_recall', float(test_metrics_dict['Recall']), epoch)
                self.writer.add_scalar('epoch/test_f1', float(test_metrics_dict['F1']), epoch)
                self.writer.add_scalar('epoch/test_auroc', float(test_metrics_dict['AUROC']), epoch)
                self.writer.add_scalar('epoch/test_auprc', float(test_metrics_dict['AUPRC']), epoch)

                # Gap metrics (for overfitting detection) - 训练-验证差距
                train_val_gap_acc = train_metrics_dict['Accuracy'] - val_metrics_dict['Accuracy']
                train_val_gap_auprc = train_metrics_dict['AUPRC'] - val_metrics_dict['AUPRC']
                train_val_gap_loss = train_loss - vali_loss
                self.writer.add_scalar('gap/train_val_acc', float(train_val_gap_acc), epoch)
                self.writer.add_scalar('gap/train_val_auprc', float(train_val_gap_auprc), epoch)
                self.writer.add_scalar('gap/train_val_loss', float(train_val_gap_loss), epoch)

                # 训练-测试差距
                train_test_gap_acc = train_metrics_dict['Accuracy'] - test_metrics_dict['Accuracy']
                train_test_gap_auprc = train_metrics_dict['AUPRC'] - test_metrics_dict['AUPRC']
                self.writer.add_scalar('gap/train_test_acc', float(train_test_gap_acc), epoch)
                self.writer.add_scalar('gap/train_test_auprc', float(train_test_gap_auprc), epoch)

            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        best_model_path = path
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path), map_location=self.device)
        else:
            # self.model.load_state_dict(torch.load(best_model_path),map_location=self.device)
            state_dict = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        if self.writer is not None:
            self.writer.close()
        return self.model

    def test(self, test=0):
        """Evaluate model on validation and test splits and log metrics to file."""
        vali_data, vali_loader = self._get_data(self.data_dict, flag="VAL")
        test_data, test_loader = self._get_data(self.data_dict, flag="TEST")
        if test:
            print("loading model")

            path = self.saving_path

            model_path = path
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
                "./results/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(f"{self.args.dataset}  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return


class ColaxPipline:
    """High-level pipeline orchestrator for Colax experiments."""

    def __init__(self, config):
        self.args = get_args(config)

        # Root is provided by runner/config; allow None
        self.root_dir = getattr(self.args, 'root_path', None)

        # Defer data loading until train/test to respect flags set by runner
        self.data_dict = None

        # Explainer checkpoint path with prototype attributes
        base_root = self.root_dir if self.root_dir else "."
        # Normalize path to avoid issues like "././"
        base_root = os.path.normpath(base_root)
        # Get prototype counts and proto_len
        npro_local = getattr(self.args, 'num_prototypes_local', getattr(self.args, 'num_prototypes', 1))
        npro_global = getattr(self.args, 'num_prototypes_global', getattr(self.args, 'num_prototypes', 1))
        proto_len = getattr(self.args, 'proto_len', getattr(self.args, 'prototype_len', 30))
        print(f"[Path] Generating checkpoint path: proto_len={proto_len}, n_local={npro_local}, n_global={npro_global}")
        print(f"[Path] args.num_prototypes_local={getattr(self.args, 'num_prototypes_local', 'NOT_SET')}, args.num_prototypes_global={getattr(self.args, 'num_prototypes_global', 'NOT_SET')}")
        ablation_str = ""
        if hasattr(self.args, 'ablation') and self.args.ablation and self.args.ablation != 'none':
            ablation_str = f"_{self.args.ablation}"
        # Use os.path.join for proper path handling
        # Format: pl_{proto_len}_local_{npro_local}_global_{npro_global}_colax.pt
        self.saving_path = os.path.join(base_root, "checkpoints", "explainer", 
                                       f"hybrid_{self.args.dataset_name}_pl_{proto_len}_local_{npro_local}_global_{npro_global}{ablation_str}_colax.pt")
        print(f"[Path] Final checkpoint path: {self.saving_path}")
        self.args.saving_path = self.saving_path

    def get_class_model(self, X):
        """Instantiate or load the classification backbone used for scoring."""
        print(self.args.class_model_type)
        if self.args.class_model_type == "cnn":
            class_model = CNN(
                d_inp=X[0].shape[-1],
                n_classes=self.args.num_classes, )

        elif self.args.class_model_type == "lstm":
            class_model = LSTM(
                d_inp=X[0].shape[-1],
                n_classes=self.args.num_classes,
            )

        else:
            class_model = TransformerMVTS(

                d_inp=X.shape[-1],
                max_len=X.shape[1],
                nlayers=1,
                n_classes=self.args.num_classes,
                trans_dim_feedforward=64,
                trans_dropout=0.1,
                d_pe=16,
                stronger_clf_head=False,
                norm_embedding=True,
            )

        if self.args.data == "mitecg":
            # Use fixed transformer classifier checkpoint for mitecg
            class_model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[1],
                n_classes=self.args.num_classes,
                trans_dim_feedforward=64,
                trans_dropout=0.1,
                d_pe=16,
                stronger_clf_head=False,
                norm_embedding=True,
            )
            model_path = os.path.join(self.args.root_path or '.', 'checkpoints', 'classification_models',
                                      'mitecg_transformer.pt')
            class_model.load_state_dict(torch.load(model_path, map_location=self.args.device))
            class_model = class_model.to(self.args.device)
            class_model.eval()  # 预训练的分类模型

        # New datasets (mcce/mcch/mtce/mtch) use local classification models
        if self.args.data in ["mcce", "mcch", "mtce", "mtch"]:
            class_model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[1],
                n_classes=4,
                trans_dim_feedforward=16,
                trans_dropout=0.1,
                d_pe=16, )
            model_path = os.path.join(self.args.root_path or '.', 'checkpoints', 'classification_models',
                                      f"{self.args.data}_transformer.pt")
            class_model.load_state_dict(torch.load(model_path, map_location=self.args.device))
            class_model = class_model.to(self.args.device)
            class_model.eval()

        elif self.args.meta_dataset == "ucr":
            # UCR classifiers stored under classification_models
            # For UCR datasets, always use Transformer model
            class_model = CNN(
                d_inp=X.shape[-1],
                n_classes=self.args.num_classes,
            )
            model_path = os.path.join(self.args.root_path or '.', 'checkpoints', 'classification_models',
                                      f"{self.args.dataset}.pt")
            class_model.load_state_dict(torch.load(model_path, map_location=self.args.device))
            class_model = class_model.to(self.args.device)
            class_model.eval()

        # Ensure model is on the correct device for all cases (fallback for cases not handled above)
        class_model = class_model.to(self.args.device)
        class_model.eval()

        return class_model

    import numpy as np
    import torch

    def _linear_interp_replacement_1d(self, x_1d: torch.Tensor, mask_keep_1d: torch.Tensor) -> torch.Tensor:
        """
        x_1d: (T,) torch tensor on device
        mask_keep_1d: (T,) torch tensor, 1=keep anchor, 0=replace (to be interpolated)
        return: x_rep (T,) torch tensor on same device
        """
        device = x_1d.device
        x_np = x_1d.detach().float().cpu().numpy()
        m_np = mask_keep_1d.detach().cpu().numpy().astype(bool)

        idx = np.arange(x_np.shape[0])

        anchor_idx = idx[m_np]
        if anchor_idx.size == 0:
            # 全被 mask：回退策略（你可改成 0 或 全局均值）
            fill_val = float(np.mean(x_np))
            rep_np = np.full_like(x_np, fill_val)
        elif anchor_idx.size == 1:
            # 只有一个锚点：全序列填这个值
            rep_np = np.full_like(x_np, x_np[anchor_idx[0]])
        else:
            # 线性插值；边界自动用最近锚点外推（np.interp 默认）
            rep_np = np.interp(idx, anchor_idx, x_np[anchor_idx]).astype(np.float32)

        return torch.from_numpy(rep_np).to(device)

    def _evaluate_with_perturbation(self, X, y, times, saliency_scores, class_model, num_steps=20, 
                                   positive_class_threshold=0.75):
        """
        使用基于扰动的方法评估 saliency 质量
        
        实现 Deletion 和 Insertion 测试：
        - Deletion: 逐步删除最重要的区域，观察准确率下降
        - Insertion: 逐步添加最重要的区域，观察准确率提升
        
        Args:
            X: 输入数据 (N, T, 1)
            y: 真实标签 (N,)
            times: 时间戳 (N, T)
            saliency_scores: Saliency 分数 (N, T, 1)
            class_model: 分类模型
            num_steps: 扰动步数（默认 20，步数越多评估越精细但耗时越长）
            positive_class_threshold: 用于计算 AUPRC 的正类阈值（默认 0.75，表示将重要性最高的 25% 作为正类）
                                     范围 [0, 1]，值越大，正类比例越小
        
        Returns:
            dict: 包含评估指标的字典
        """
        class_model.eval()
        
        # 确保所有数据都在模型所在的设备上
        device = next(class_model.parameters()).device
        X = X.to(device)
        y = y.to(device)
        times = times.to(device)
        saliency_scores = saliency_scores.to(device)
        
        N, T, _ = X.shape
        
        # 归一化 saliency 分数
        saliency_norm = saliency_scores.clone()
        for i in range(N):
            sal_i = saliency_scores[i, :, 0]
            if sal_i.max() > sal_i.min():
                saliency_norm[i, :, 0] = (sal_i - sal_i.min()) / (sal_i.max() - sal_i.min())
        
        # 计算原始准确率
        # 参照 train_classifier_ucr.py 的方式：创建时间张量并转置数据
        with torch.no_grad():
            # 创建时间张量（参照 train_classifier_ucr.py 第128-130行）
            times_new = torch.arange(T, dtype=torch.float32, device=device)
            times_new = times_new.unsqueeze(0).repeat(N, 1)  # (N, T)
            
            # 转置数据格式（参照 train_classifier_ucr.py 第144-146行）
            X_t = X.transpose(0, 1)  # (N, T, 1) -> (T, N, 1)
            times_t = times_new.transpose(0, 1)  # (N, T) -> (T, N)
            
            # 模型调用（所有模型类型使用相同方式，参照 train_classifier_ucr.py 第146行）
            output = class_model(X_t, times_t)
            if isinstance(output, tuple):
                output = output[0]
            
            original_pred = output.argmax(dim=1)
            original_acc = (original_pred == y).float().mean().item()
            
            # 调试信息：检查预测分布
            unique_preds, counts_preds = torch.unique(original_pred, return_counts=True)
            unique_y, counts_y = torch.unique(y, return_counts=True)
            print(f"预测类别分布: {dict(zip(unique_preds.cpu().tolist(), counts_preds.cpu().tolist()))}")
            print(f"真实类别分布: {dict(zip(unique_y.cpu().tolist(), counts_y.cpu().tolist()))}")
            print(f"预测概率范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        print(f"原始准确率: {original_acc:.4f}")
        
        # 如果准确率太低，给出警告
        if original_acc < 0.2:
            print("\n" + "=" * 80)
            print("警告: 分类模型准确率过低！")
            print(f"当前准确率: {original_acc:.4f} (随机猜测约为 {1.0/self.args.num_classes:.4f})")
            print("可能的原因:")
            print("  1. 分类模型未正确训练或加载")
            print("  2. 模型架构参数不匹配")
            print("  3. 数据预处理有问题")
            print("\n建议:")
            print(f"  运行以下命令重新训练分类模型:")
            print(f"  python train_classifier_ucr.py --dataset {self.args.dataset}")
            print("=" * 80 + "\n")
        print(f"进行 {num_steps} 步扰动测试...")
        
        deletion_accs = []
        insertion_accs = []
        
        # 对每个扰动步数进行评估
        for step in range(num_steps + 1):
            ratio = step / num_steps  # 扰动比例
            
            X_deletion = X.clone()
            X_insertion = X.clone()
            
            for i in range(N):
                sal_i = saliency_norm[i, :, 0]  # (T,) tensor
                X_i = X[i, :, 0]  # (T,) tensor
                
                # 生成替换值：使用样本均值（参照 txai 中的策略）
                X_replacement_val = X_i.mean()  # 标量
                X_replacement = X_replacement_val.expand_as(X_i)  # (T,) 所有位置使用相同的替换值

                
                # 根据 saliency 分数排序（从高到低）
                sorted_indices = torch.argsort(-sal_i)  # 从高到低排序
                
                # Deletion: 删除最重要的区域（用替换值替换）
                # 重要性高的区域mask=0（完全替换），重要性低的区域mask=1（保留原值）
                deletion_mask = torch.ones_like(sal_i)  # (T,) 初始化为1（保留所有）
                num_to_mask = int(T * ratio)
                if num_to_mask > 0:
                    # 将重要性最高的num_to_mask个位置的mask设为0（完全替换）
                    mask_indices = sorted_indices[:num_to_mask]
                    deletion_mask[mask_indices] = 0.0

                # X_replacement = self._linear_interp_replacement_1d(X_i, deletion_mask)
                # 应用mask公式：X_perturbed = mask * X_original + (1 - mask) * X_replacement
                X_deletion[i, :, 0] = deletion_mask * X_i + (1 - deletion_mask) * X_replacement
                
                # Insertion: 只保留最重要的区域（其余用替换值替换）
                # 重要性高的区域mask=1（保留原值），重要性低的区域mask=0（完全替换）
                insertion_mask = torch.zeros_like(sal_i)  # (T,) 初始化为0（替换所有）
                # num_to_keep = int(T * (1 - ratio))
                num_to_keep = int(T * ratio)

                if num_to_keep > 0:
                    # 将重要性最高的num_to_keep个位置的mask设为1（保留原值）
                    keep_indices = sorted_indices[:num_to_keep]
                    insertion_mask[keep_indices] = 1.0
                
                # 应用mask公式：X_perturbed = mask * X_original + (1 - mask) * X_replacement
                X_insertion[i, :, 0] = insertion_mask * X_i + (1 - insertion_mask) * X_replacement
            
            # 评估 Deletion（参照 train_classifier_ucr.py 的方式）
            with torch.no_grad():
                # 创建时间张量
                times_step = torch.arange(T, dtype=torch.float32, device=device)
                times_step = times_step.unsqueeze(0).repeat(N, 1)  # (N, T)
                
                # 转置数据格式
                X_del_t = X_deletion.transpose(0, 1)  # (N, T, 1) -> (T, N, 1)
                times_del_t = times_step.transpose(0, 1)  # (N, T) -> (T, N)
                
                # 模型调用（参照 train_classifier_ucr.py 第146行）
                output_del = class_model(X_del_t, times_del_t)
                if isinstance(output_del, tuple):
                    output_del = output_del[0]
                
                pred_del = output_del.argmax(dim=1)
                acc_del = (pred_del == y).float().mean().item()
                deletion_accs.append(acc_del)
            
            # 评估 Insertion（参照 train_classifier_ucr.py 的方式）
            with torch.no_grad():
                # 创建时间张量
                times_step = torch.arange(T, dtype=torch.float32, device=device)
                times_step = times_step.unsqueeze(0).repeat(N, 1)  # (N, T)
                
                # 转置数据格式
                X_ins_t = X_insertion.transpose(0, 1)  # (N, T, 1) -> (T, N, 1)
                times_ins_t = times_step.transpose(0, 1)  # (N, T) -> (T, N)
                
                # 模型调用（参照 train_classifier_ucr.py 第146行）
                output_ins = class_model(X_ins_t, times_ins_t)
                if isinstance(output_ins, tuple):
                    output_ins = output_ins[0]
                
                pred_ins = output_ins.argmax(dim=1)
                acc_ins = (pred_ins == y).float().mean().item()
                insertion_accs.append(acc_ins)
        
        # 计算 AOPC (Area Over the Perturbation Curve)
        # Deletion AOPC: 准确率下降的面积（越大越好，说明删除重要区域后准确率快速下降）
        deletion_aopc_val = np.trapz([original_acc - acc for acc in deletion_accs], dx=1.0/num_steps)
        
        # Insertion AOPC: 准确率提升的面积（越大越好，说明添加重要区域后准确率快速提升）
        # Insertion测试的逻辑：
        # - ratio=0: 保留所有区域（准确率最高，接近original_acc）
        # - ratio增大: 保留的区域减少（准确率下降）
        # - ratio=1: 全部替换（准确率最低，baseline）
        # insertion_accs[0] = 最高准确率（保留所有）
        # insertion_accs[-1] = 最低准确率（全部替换，baseline）
        # 
        # 标准的Insertion AOPC应该从baseline开始，逐步添加重要区域，准确率逐渐提升
        # 需要反转数组，使其从低到高排列（从baseline到最佳）
        # insertion_accs_reversed = insertion_accs[::-1]  # 反转：从baseline（低）到最佳（高）
        # baseline_acc = insertion_accs_reversed[0]  # baseline（全部替换时的准确率，最低）
        # insertion_aopc_val = np.trapz([acc - baseline_acc for acc in insertion_accs_reversed], dx=1.0/num_steps)


        # Deletion AOPC: 面积越大越好（删除重要区域后性能下降越快、越明显）
        deletion_aopc_val = np.trapz([original_acc - acc for acc in deletion_accs], dx=1.0 / num_steps)

        # Insertion AOPC: 面积越大越好（从baseline开始插入重要区域后性能提升越快、越明显）
        # 你当前 insertion 的定义是：ratio 从 0->1，保留 top-(ratio*T)，即 baseline->best
        baseline_acc = insertion_accs[0]
        insertion_aopc_val = np.trapz([acc - baseline_acc for acc in insertion_accs], dx=1.0 / num_steps)

        ratios = [step / num_steps for step in range(num_steps + 1)]

        results_dict = {
            # 'deletion_aopc': deletion_aopc_val,  # 越小越好
            # 'insertion_aopc': insertion_aopc_val,  # 越大越好
            # 'deletion_acc_final': deletion_accs[-1],  # 删除所有重要区域后的准确率
            # 'insertion_acc_final': insertion_accs[-1],  # 全部替换后的准确率（baseline）
            # 'insertion_acc_initial': insertion_accs[0],  # 保留所有重要区域后的准确率
            # 'deletion_acc_drop': original_acc - deletion_accs[-1],  # 准确率下降幅度
            # 'insertion_acc_gain': insertion_accs[-1] - insertion_accs[0],  # 准确率提升幅度（从baseline到最佳）
            'original_acc': original_acc,
            'deletion_aopc': float(deletion_aopc_val),  # 越大越好
            'insertion_aopc': float(insertion_aopc_val),  # 越大越好
            'deletion_acc_final': float(deletion_accs[-1]),  # ratio=1 (all masked)
            'insertion_acc_initial': float(insertion_accs[0]),  # ratio=0 baseline
            'insertion_acc_final': float(insertion_accs[-1]),  # ratio=1 fully inserted (≈ original)
            'deletion_acc_drop': float(original_acc - deletion_accs[-1]),
            'insertion_acc_gain': float(insertion_accs[-1] - insertion_accs[0]),
            # plotting
            'ratios': ratios,
            'deletion_curve_acc': deletion_accs,
            'insertion_curve_acc': insertion_accs,
        }

        return results_dict
    
    def _evaluate_with_perturbation_auroc(self, X, y, times, saliency_scores, class_model, num_steps=20):
        """
        使用基于扰动的方法评估 saliency 质量，计算 AUROC 随遮挡比例的变化。

        同时执行 Deletion 和 Insertion 测试：
        - Deletion (MoRF): 逐步删除最重要的区域，观察 AUROC 下降
        - Insertion (LeRF): 逐步删除最不重要的区域，观察 AUROC 保持

        Args:
            X: 输入数据 (N, T, 1)
            y: 真实标签 (N,)
            times: 时间戳 (N, T)
            saliency_scores: Saliency 分数 (N, T, 1)
            class_model: 分类模型
            num_steps: 扰动步数（默认 20，步数越多评估越精细但耗时越长）

        Returns:
            dict: 包含评估指标的字典，包括 Deletion 和 Insertion AUROC 曲线
        """
        from sklearn.metrics import roc_auc_score
        
        class_model.eval()
        
        # 确保所有数据都在模型所在的设备上
        device = next(class_model.parameters()).device
        X = X.to(device)
        y = y.to(device)
        times = times.to(device)
        saliency_scores = saliency_scores.to(device)
        
        N, T, _ = X.shape
        
        # 归一化 saliency 分数
        saliency_norm = saliency_scores.clone()
        for i in range(N):
            sal_i = saliency_scores[i, :, 0]
            if sal_i.max() > sal_i.min():
                saliency_norm[i, :, 0] = (sal_i - sal_i.min()) / (sal_i.max() - sal_i.min())
        
        # 计算原始 AUROC
        with torch.no_grad():
            # 创建时间张量
            times_new = torch.arange(T, dtype=torch.float32, device=device)
            times_new = times_new.unsqueeze(0).repeat(N, 1)  # (N, T)
            
            # 转置数据格式
            X_t = X.transpose(0, 1)  # (N, T, 1) -> (T, N, 1)
            times_t = times_new.transpose(0, 1)  # (N, T) -> (T, N)
            
            # 模型调用
            output = class_model(X_t, times_t)
            if isinstance(output, tuple):
                output = output[0]
            
            # 计算原始 AUROC
            probs = torch.nn.functional.softmax(output, dim=1)  # (N, num_classes)
            probs_np = probs.cpu().numpy()
            y_np = y.cpu().numpy()
            
            # 转换为 one-hot 编码
            y_onehot = torch.nn.functional.one_hot(
                torch.tensor(y_np, dtype=torch.long),
                num_classes=self.args.num_classes
            ).float().cpu().numpy()
            
            # 检查是否有足够的类别
            unique_labels = np.unique(y_np)
            if unique_labels.size < 2:
                print("[AUROC] Only one class present; set AUROC to 0.0")
                original_auroc = 0.0
            else:
                try:
                    original_auroc = roc_auc_score(y_onehot, probs_np, multi_class="ovr")
                except Exception as e:
                    print(f"[AUROC] Failed to calculate original AUROC: {e}; set to 0.0")
                    original_auroc = 0.0
        
        print(f"原始 AUROC: {original_auroc:.4f}")
        print(f"进行 {num_steps} 步扰动测试（计算 AUROC）...")
        
        deletion_aurocs = []
        ratios = []
        
        # 对每个扰动步数进行评估
        for step in range(num_steps + 1):
            ratio = step / num_steps  # 扰动比例
            ratios.append(ratio)
            
            X_deletion = X.clone()
            
            for i in range(N):
                sal_i = saliency_norm[i, :, 0]  # (T,) tensor
                X_i = X[i, :, 0]  # (T,) tensor
                
                # 生成替换值：使用样本均值
                X_replacement_val = X_i.mean()  # 标量
                X_replacement = X_replacement_val.expand_as(X_i)  # (T,)
                
                # 根据 saliency 分数排序（从高到低）
                sorted_indices = torch.argsort(-sal_i)  # 从高到低排序
                
                # Deletion: 删除最重要的区域（用替换值替换）
                deletion_mask = torch.ones_like(sal_i)  # (T,) 初始化为1（保留所有）
                num_to_mask = int(T * ratio)
                if num_to_mask > 0:
                    # 将重要性最高的num_to_mask个位置的mask设为0（完全替换）
                    mask_indices = sorted_indices[:num_to_mask]
                    deletion_mask[mask_indices] = 0.0
                
                # 应用mask公式：X_perturbed = mask * X_original + (1 - mask) * X_replacement
                X_deletion[i, :, 0] = deletion_mask * X_i + (1 - deletion_mask) * X_replacement
            
            # 评估 Deletion
            with torch.no_grad():
                # 创建时间张量
                times_step = torch.arange(T, dtype=torch.float32, device=device)
                times_step = times_step.unsqueeze(0).repeat(N, 1)  # (N, T)
                
                # 转置数据格式
                X_del_t = X_deletion.transpose(0, 1)  # (N, T, 1) -> (T, N, 1)
                times_del_t = times_step.transpose(0, 1)  # (N, T) -> (T, N)
                
                # 模型调用
                output_del = class_model(X_del_t, times_del_t)
                if isinstance(output_del, tuple):
                    output_del = output_del[0]
                
                # 计算 AUROC
                probs_del = torch.nn.functional.softmax(output_del, dim=1)  # (N, num_classes)
                probs_del_np = probs_del.cpu().numpy()
                
                # 检查是否有足够的类别
                unique_labels = np.unique(y_np)
                if unique_labels.size < 2:
                    auroc_del = 0.0
                else:
                    try:
                        auroc_del = roc_auc_score(y_onehot, probs_del_np, multi_class="ovr")
                    except Exception as e:
                        print(f"[AUROC] Step {step} failed: {e}; set to 0.0")
                        auroc_del = 0.0
                
                deletion_aurocs.append(auroc_del)
        
        # ===== Insertion (LeRF): 逐步删除最不重要的区域 =====
        print(f"进行 {num_steps} 步 Insertion 测试（计算 AUROC）...")

        insertion_aurocs = []

        for step in range(num_steps + 1):
            ratio = step / num_steps
            X_insertion = X.clone()

            for i in range(N):
                sal_i = saliency_norm[i, :, 0]
                X_i = X[i, :, 0]
                X_replacement_val = X_i.mean()
                X_replacement = X_replacement_val.expand_as(X_i)

                # LeRF: 从低到高排序，先删除最不重要的区域
                sorted_indices = torch.argsort(sal_i)  # 从低到高排序

                insertion_mask = torch.ones_like(sal_i)
                num_to_mask = int(T * ratio)
                if num_to_mask > 0:
                    mask_indices = sorted_indices[:num_to_mask]
                    insertion_mask[mask_indices] = 0.0

                X_insertion[i, :, 0] = insertion_mask * X_i + (1 - insertion_mask) * X_replacement

            with torch.no_grad():
                times_step = torch.arange(T, dtype=torch.float32, device=device)
                times_step = times_step.unsqueeze(0).repeat(N, 1)
                X_ins_t = X_insertion.transpose(0, 1)
                times_ins_t = times_step.transpose(0, 1)

                output_ins = class_model(X_ins_t, times_ins_t)
                if isinstance(output_ins, tuple):
                    output_ins = output_ins[0]

                probs_ins = torch.nn.functional.softmax(output_ins, dim=1)
                probs_ins_np = probs_ins.cpu().numpy()

                if np.unique(y_np).size < 2:
                    auroc_ins = 0.0
                else:
                    try:
                        auroc_ins = roc_auc_score(y_onehot, probs_ins_np, multi_class="ovr")
                    except Exception:
                        auroc_ins = 0.0

                insertion_aurocs.append(auroc_ins)

        # ===== 计算 DAUC 和 PAUC =====
        dauc = float(np.mean(deletion_aurocs))
        pauc = float(np.mean(insertion_aurocs))
        gap = pauc - dauc
        print(f"DAUC (Deletion)={dauc:.4f}  PAUC (Insertion)={pauc:.4f}  Gap={gap:.4f}")

        results_dict = {
            'original_auroc': float(original_auroc),
            'ratios': ratios,
            'deletion_curve_auroc': deletion_aurocs,
            'insertion_curve_auroc': insertion_aurocs,
            'dauc': dauc,
            'pauc': pauc,
            'gap': gap,
        }

        return results_dict

    def train_colax(self):
        """Train the explainer model on the configured dataset."""

        args = self.args
        print(args)
        trained_model = None

        if args.is_training:
            # Load data now with current flags
            self.data_dict = get_saliency_data(self.args, return_dict=True)
            for ii in range(args.itr):
                seed = 41 + ii
                random.seed(seed)
                os.environ["PYTHONHASHSEED"] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                # setting record of experiments
                args.seed = seed

                ############ Train segmentation model
                exp = Exp_Classification(args, self.data_dict)

                exp.train()

                model = exp.model

    def load_seg_model(self):
        """Load a saved segmentation/explainer checkpoint from disk."""

        # seg_model = ProtoPTST.Model(args).float()
        seg_model = ProtoPTST.Model(self.args).float()

        # Load checkpoint with error handling for state_dict mismatches
        checkpoint = torch.load(self.saving_path, map_location=self.args.device)
        
        # Try to load with strict=False first to handle mismatches
        try:
            seg_model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            # If that fails, try to filter out unexpected keys
            model_state = seg_model.state_dict()
            filtered_checkpoint = {}
            missing_keys = []
            unexpected_keys = []
            
            for key, value in checkpoint.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_checkpoint[key] = value
                    else:
                        unexpected_keys.append(f"{key} (shape mismatch: {model_state[key].shape} vs {value.shape})")
                else:
                    unexpected_keys.append(key)
            
            for key in model_state:
                if key not in filtered_checkpoint:
                    missing_keys.append(key)
            
            if unexpected_keys:
                print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint (will be ignored):")
                for key in unexpected_keys[:10]:  # Print first 10
                    print(f"  - {key}")
                if len(unexpected_keys) > 10:
                    print(f"  ... and {len(unexpected_keys) - 10} more")
            
            if missing_keys:
                print(f"Warning: {len(missing_keys)} missing keys in checkpoint (will use random initialization):")
                for key in missing_keys[:10]:  # Print first 10
                    print(f"  - {key}")
                if len(missing_keys) > 10:
                    print(f"  ... and {len(missing_keys) - 10} more")
            
            seg_model.load_state_dict(filtered_checkpoint, strict=False)
        
        seg_model = seg_model.to(self.args.device)

        return seg_model

    def eval_colax(self):
        """Evaluate saliency explanations against ground-truth annotations."""
        # only use when there are ground truth explanations

        seg_model = self.load_seg_model()

        ############ Load test data and classification model

        self.args.saliency = True

        X, y, times, gt_exps = get_saliency_data(self.args, return_dict=False)
        # 不要移动到 CPU，保持在原始设备上，让 _evaluate_with_perturbation 处理设备转换
        class_model = self.get_class_model(X)

        ############ compute saliency scores
        Scorer = ScoreSubsequences(self.args, class_model, seg_model)
        score = Scorer.get_score_as_GTEXP(X)

        # shapes are (N, T, 1) and (N, T, 1)
        
        # Check if ground truth explanations exist (not all zeros)
        has_gt = gt_exps is not None and gt_exps.sum() > 0

        if not has_gt:
            print("\n" + "=" * 80)
            print("警告: 该数据集没有 ground truth saliency 标注")
            print("使用基于扰动的评估方法 (Perturbation-based Evaluation)")
            print("=" * 80)
            
            # 使用基于扰动的评估方法
            # 获取配置参数（如果存在）
            num_steps = getattr(self.args, 'perturbation_num_steps', 20)
            results_dict = self._evaluate_with_perturbation(
                X, y, times, score, class_model, 
                num_steps=num_steps
            )
            
            print("\n基于扰动的评估结果:")
            for k, v in results_dict.items():
                if isinstance(v, (list, np.ndarray)):
                    print(f'\t{k} = {np.mean(v):.4f} +- {np.std(v) / np.sqrt(len(v)):.4f}')
                else:
                    print(f'\t{k} = {v:.4f}')
            
            # 保存结果
            with open("results.txt", "a") as f:
                npro_local = getattr(self.args, 'num_prototypes_local', getattr(self.args, 'num_prototypes', 1))
                npro_global = getattr(self.args, 'num_prototypes_global', getattr(self.args, 'num_prototypes', 1))
                ablation_str = ""
                if hasattr(self.args, 'ablation') and self.args.ablation and self.args.ablation != 'none':
                    ablation_str = f"_ablation{self.args.ablation}"
                f.write(
                    f"{self.args.data}_slen{self.args.seq_len}_npro_local{npro_local}_npro_global{npro_global}_prototype_len{self.args.prototype_len}{ablation_str}_perturbation")
                for k, v in results_dict.items():
                    if isinstance(v, (list, np.ndarray)):
                        f.write(f' {k}:{np.mean(v):.4f}-{(np.std(v) / np.sqrt(len(v))):.4f}')
                    else:
                        f.write(f' {k}:{v:.4f}')
                f.write('\n')
            
            print("=" * 80)
            # ===== Save data before plotting =====
            out_dir = os.path.join(self.args.root_path or '.', 'results', 'perturbation_curves', self.args.dataset_name)
            os.makedirs(out_dir, exist_ok=True)
            
            # filename tag
            tag = f"mine_{self.args.dataset_name}_steps{num_steps}_model{self.args.class_model_type}"
            
            # Save data to files
            try:
                import json
                
                # Save scalar metrics to JSON
                scalar_metrics = {}
                for k, v in results_dict.items():
                    if not isinstance(v, (list, np.ndarray)) and k not in ['ratios', 'deletion_curve_acc', 'insertion_curve_acc']:
                        # Convert numpy types to Python native types for JSON serialization
                        if isinstance(v, (np.integer, np.floating)):
                            scalar_metrics[k] = float(v)
                        else:
                            scalar_metrics[k] = v
                json_path = os.path.join(out_dir, f"{tag}_metrics.json")
                with open(json_path, 'w') as f:
                    json.dump(scalar_metrics, f, indent=2)
                print(f"[Saved] Metrics data: {json_path}")
                
                # Save curve data to NPY
                curve_data = {
                    'ratios': np.array(results_dict['ratios']),
                    'deletion_curve_acc': np.array(results_dict['deletion_curve_acc']),
                    'insertion_curve_acc': np.array(results_dict['insertion_curve_acc']),
                    'original_acc': results_dict['original_acc'],
                }
                npy_path = os.path.join(out_dir, f"{tag}_curves.npy")
                np.save(npy_path, curve_data, allow_pickle=True)
                print(f"[Saved] Curve data: {npy_path}")
                
            except Exception as e:
                print(f"[Save Data] Failed to save data: {e}")
            
            # ===== Plot Deletion / Insertion curves (Optimized) =====
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend

                ratios = np.array(results_dict['ratios'])
                del_curve = np.array(results_dict['deletion_curve_acc'])
                ins_curve = np.array(results_dict['insertion_curve_acc'])
                original_acc = results_dict['original_acc']

                fig_path_png = os.path.join(out_dir, f"{tag}_del_ins_curve.png")
                fig_path_pdf = os.path.join(out_dir, f"{tag}_del_ins_curve.pdf")

                # Create figure with better styling
                available_styles = plt.style.available
                if 'seaborn-v0_8-darkgrid' in available_styles:
                    plt.style.use('seaborn-v0_8-darkgrid')
                elif 'seaborn-darkgrid' in available_styles:
                    plt.style.use('seaborn-darkgrid')
                else:
                    plt.style.use('default')
                fig, ax = plt.subplots(figsize=(8, 5.5))
                
                # Plot curves with better styling
                line1 = ax.plot(ratios, del_curve, label='Deletion', 
                               color='#e74c3c', linewidth=2.5, marker='o', markersize=4, 
                               markerfacecolor='white', markeredgewidth=1.5, alpha=0.9)
                line2 = ax.plot(ratios, ins_curve, label='Insertion', 
                               color='#3498db', linewidth=2.5, marker='s', markersize=4, 
                               markerfacecolor='white', markeredgewidth=1.5, alpha=0.9)
                
                # Add horizontal line for original accuracy
                ax.axhline(y=original_acc, color='gray', linestyle='--', linewidth=1.5, 
                          alpha=0.7, label=f'Original Acc ({original_acc:.3f})')
                
                # Add vertical line at ratio=0.5 for reference
                ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                
                # Labels and title
                ax.set_xlabel('Perturbation Ratio', fontsize=12, fontweight='bold')
                ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
                ax.set_title(f'Deletion/Insertion Curves - {self.args.dataset_name}', 
                           fontsize=13, fontweight='bold', pad=15)
                
                # Grid
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_axisbelow(True)
                
                # Legend with better positioning
                legend = ax.legend(loc='best', fontsize=10, framealpha=0.95, 
                                  edgecolor='gray', fancybox=True, shadow=True)
                legend.get_frame().set_linewidth(0.5)
                
                # Set axis limits for better visualization
                ax.set_xlim(-0.02, 1.02)
                y_min = min(del_curve.min(), ins_curve.min()) * 0.95
                y_max = max(del_curve.max(), ins_curve.max()) * 1.05
                ax.set_ylim(max(0, y_min), min(1, y_max))
                
                # Add text annotation with key metrics
                metrics_text = f"Deletion AOPC: {results_dict['deletion_aopc']:.4f}\n"
                metrics_text += f"Insertion AOPC: {results_dict['insertion_aopc']:.4f}"
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       family='monospace')
                
                # Improve ticks
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='both', which='minor', labelsize=8)
                
                plt.tight_layout()
                plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
                plt.savefig(fig_path_pdf, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f"[Saved] Curve plot (PNG): {fig_path_png}")
                print(f"[Saved] Curve plot (PDF): {fig_path_pdf}")

            except Exception as e:
                print(f"[Plot] Failed to plot curves: {e}")
                import traceback
                traceback.print_exc()

            return

        results_dict = ground_truth_xai_eval(score, gt_exps)
    
    def eval_colax_auroc(self):
        """Evaluate saliency explanations using AUROC with perturbation."""
        seg_model = self.load_seg_model()

        ############ Load test data and classification model
        self.args.saliency = True

        X, y, times, gt_exps = get_saliency_data(self.args, return_dict=False)
        class_model = self.get_class_model(X)

        ############ compute saliency scores
        Scorer = ScoreSubsequences(self.args, class_model, seg_model)
        score = Scorer.get_score_as_GTEXP(X)

        # 使用基于扰动的 AUROC 评估方法
        num_steps = getattr(self.args, 'perturbation_num_steps', 20)
        results_dict = self._evaluate_with_perturbation_auroc(
            X, y, times, score, class_model, 
            num_steps=num_steps
        )
        
        print("\n基于扰动的 AUROC 评估结果:")
        print(f'\t原始 AUROC: {results_dict["original_auroc"]:.4f}')
        print(f'\t遮挡比例范围: {results_dict["ratios"][0]:.2f} - {results_dict["ratios"][-1]:.2f}')
        print(f'\tDeletion 最终 AUROC (ratio={results_dict["ratios"][-1]:.2f}): {results_dict["deletion_curve_auroc"][-1]:.4f}')
        print(f'\tInsertion 最终 AUROC (ratio={results_dict["ratios"][-1]:.2f}): {results_dict["insertion_curve_auroc"][-1]:.4f}')
        print(f'\tDAUC={results_dict["dauc"]:.4f}  PAUC={results_dict["pauc"]:.4f}  Gap={results_dict["gap"]:.4f}')
        
        # ===== Save data before plotting =====
        out_dir = os.path.join(self.args.root_path or '.', 'results', 'perturbation_curves',self.args.dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # filename tag
        tag = f"mine_{self.args.dataset_name}_auroc_steps{num_steps}_model{self.args.class_model_type}"
        
        # Save data to files
        try:
            import json
            
            # Save scalar metrics to JSON
            scalar_metrics = {
                'original_auroc': results_dict['original_auroc'],
                'dauc': results_dict.get('dauc', 0),
                'pauc': results_dict.get('pauc', 0),
                'gap': results_dict.get('gap', 0),
            }
            json_path = os.path.join(out_dir, f"{tag}_metrics.json")
            with open(json_path, 'w') as f:
                json.dump(scalar_metrics, f, indent=2)
            print(f"[Saved] Metrics data: {json_path}")
            
            # Save curve data to NPY
            curve_data = {
                'ratios': np.array(results_dict['ratios']),
                'deletion_curve_auroc': np.array(results_dict['deletion_curve_auroc']),
                'insertion_curve_auroc': np.array(results_dict.get('insertion_curve_auroc', [])),
                'original_auroc': results_dict['original_auroc'],
                'dauc': results_dict.get('dauc', 0),
                'pauc': results_dict.get('pauc', 0),
            }
            npy_path = os.path.join(out_dir, f"{tag}_curves.npy")
            np.save(npy_path, curve_data, allow_pickle=True)
            print(f"[Saved] Curve data: {npy_path}")
            
        except Exception as e:
            print(f"[Save Data] Failed to save data: {e}")
        
        # ===== Plot AUROC curve =====
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend

            ratios = np.array(results_dict['ratios'])
            del_curve = np.array(results_dict['deletion_curve_auroc'])
            ins_curve = np.array(results_dict.get('insertion_curve_auroc', []))
            original_auroc = results_dict['original_auroc']
            dauc = results_dict.get('dauc', float('nan'))
            pauc = results_dict.get('pauc', float('nan'))

            fig_path_png = os.path.join(out_dir, f"{tag}_auroc_curve.png")
            fig_path_pdf = os.path.join(out_dir, f"{tag}_auroc_curve.pdf")

            available_styles = plt.style.available
            if 'seaborn-v0_8-darkgrid' in available_styles:
                plt.style.use('seaborn-v0_8-darkgrid')
            elif 'seaborn-darkgrid' in available_styles:
                plt.style.use('seaborn-darkgrid')
            else:
                plt.style.use('default')
            fig, ax = plt.subplots(figsize=(10, 5.5))

            # Deletion curve (MoRF)
            ax.plot(ratios, del_curve,
                    color='#e74c3c', linewidth=2.5, marker='o', markersize=4,
                    markerfacecolor='white', markeredgewidth=1.5, alpha=0.9,
                    label=f'Deletion (MoRF)  DAUC={dauc:.3f}')

            # Insertion curve (LeRF)
            if len(ins_curve) > 0:
                ax.plot(ratios, ins_curve,
                        color='#2ecc71', linewidth=2.5, marker='s', markersize=4,
                        markerfacecolor='white', markeredgewidth=1.5, alpha=0.9,
                        label=f'Insertion (LeRF)  PAUC={pauc:.3f}')

            ax.axhline(y=original_auroc, color='gray', linestyle='--', linewidth=1.5,
                      alpha=0.7, label=f'Original AUROC ({original_auroc:.3f})')

            ax.set_xlabel('Proportion Perturbed', fontsize=12, fontweight='bold')
            ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
            ax.set_title(f'AUROC vs Perturbation — {self.args.dataset_name}',
                       fontsize=13, fontweight='bold', pad=15)

            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

            legend = ax.legend(loc='best', fontsize=9, framealpha=0.95,
                            edgecolor='gray', fancybox=True, shadow=True)
            legend.get_frame().set_linewidth(0.5)

            ax.set_xlim(-0.02, max(1.0, ratios.max()) + 0.02)
            
            # Improve ticks
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            
            plt.tight_layout()
            plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(fig_path_pdf, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"[Saved] AUROC curve plot (PNG): {fig_path_png}")
            print(f"[Saved] AUROC curve plot (PDF): {fig_path_pdf}")

        except Exception as e:
            print(f"[Plot] Failed to plot AUROC curve: {e}")
            import traceback
            traceback.print_exc()
        
        return results_dict
        for k, v in results_dict.items():
            print('\t{} = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))

        with open("results.txt", "a") as f:  # "a" means append mode
            # Include both local and global prototype counts in results
            npro_local = getattr(self.args, 'num_prototypes_local', getattr(self.args, 'num_prototypes', 1))
            npro_global = getattr(self.args, 'num_prototypes_global', getattr(self.args, 'num_prototypes', 1))
            ablation_str = ""
            if hasattr(self.args, 'ablation') and self.args.ablation and self.args.ablation != 'none':
                ablation_str = f"_ablation{self.args.ablation}"
            f.write(
                f"{self.args.data}_slen{self.args.seq_len}_npro_local{npro_local}_npro_global{npro_global}_prototype_len{self.args.prototype_len}{ablation_str}")

            for k, v in results_dict.items():
                f.write(f' {k}:{np.mean(v):.4f}-{(np.std(v) / np.sqrt(len(v))):.4f}')
            f.write("\n")

    def get_saliency_score(self, X=None):
        """Compute saliency scores for provided data or test split (N, T, 1)."""

        seg_model = self.load_seg_model()

        ############ Load test data and classification model

        self.args.saliency = False

        train, val, test, _ = get_saliency_data(self.args, return_dict=False)

        if X is None:
            X, y, times = test[0], test[2], test[1]

        X = X.to("cpu")
        self.args.device = "cpu"

        class_model = self.get_class_model(X)
        class_model.to("cpu")

        ############ compute saliency scores
        Scorer = ScoreSubsequences(self.args, class_model, seg_model)
        score = Scorer.get_score_as_GTEXP(X)  # (N,T,1)

        return score


# 这里将config读取到args里面
def get_args(config):
    """Construct argument namespace from the provided config.

    Dataset-specific YAML (if present) overrides sensible defaults for
    num_classes, seq_len, prototype length, and number of prototypes.
    """
    # dataset-sensitive defaults: prioritize config object, then YAML file
    dataset_name = config.dataset.name
    
    # First, try to get values from the config object (passed from build_config)
    # OmegaConf supports both attribute and dict access
    dataset_block = None
    if hasattr(config, dataset_name):
        dataset_block = getattr(config, dataset_name)
    elif dataset_name in config:
        dataset_block = config[dataset_name]
    
    # sensible defaults; will be overridden by config object or dataset YAML if present
    NUM_CLASSES = 2
    SEQ_LEN = 500
    PROTO_LEN = 30
    N_PROTOS = 1
    N_PROTOS_LOCAL = None  # None means use N_PROTOS as fallback
    N_PROTOS_GLOBAL = None  # None means use N_PROTOS as fallback
    D_MODEL = 128
    PATCH_LEN = 4
    STRIDE = 1
    E_LAYERS_LOCAL = 1
    E_LAYERS_GLOBAL = 1

    
    # Priority 1: Use values from config object if available
    if dataset_block is not None:
        # OmegaConf supports both dict-style and attribute access
        try:
            NUM_CLASSES = int(dataset_block.get('num_classes', NUM_CLASSES))
            SEQ_LEN = int(dataset_block.get('seq_len', SEQ_LEN))
            PROTO_LEN = int(dataset_block.get('proto_len', PROTO_LEN))
            N_PROTOS = int(dataset_block.get('num_prototypes', N_PROTOS))
            # Support separate num_prototypes_local and num_prototypes_global
            # Check both dict-style and attribute access
            if 'num_prototypes_local' in dataset_block:
                N_PROTOS_LOCAL = int(dataset_block.get('num_prototypes_local'))
            elif hasattr(dataset_block, 'num_prototypes_local'):
                N_PROTOS_LOCAL = int(getattr(dataset_block, 'num_prototypes_local'))
            if 'num_prototypes_global' in dataset_block:
                N_PROTOS_GLOBAL = int(dataset_block.get('num_prototypes_global'))
            elif hasattr(dataset_block, 'num_prototypes_global'):
                N_PROTOS_GLOBAL = int(getattr(dataset_block, 'num_prototypes_global'))
            # Read d_model, patch_len, stride from config
            if 'd_model' in dataset_block:
                D_MODEL = int(dataset_block.get('d_model'))
            elif hasattr(dataset_block, 'd_model'):
                D_MODEL = int(getattr(dataset_block, 'd_model'))
            if 'patch_len' in dataset_block:
                PATCH_LEN = int(dataset_block.get('patch_len'))
            elif hasattr(dataset_block, 'patch_len'):
                PATCH_LEN = int(getattr(dataset_block, 'patch_len'))
            if 'stride' in dataset_block:
                STRIDE = int(dataset_block.get('stride'))
            elif hasattr(dataset_block, 'stride'):
                STRIDE = int(getattr(dataset_block, 'stride'))
            if 'e_layers_local' in dataset_block:
                E_LAYERS_LOCAL = int(dataset_block.get('e_layers_local'))
            elif hasattr(dataset_block, 'e_layers_local'):
                E_LAYERS_LOCAL = int(getattr(dataset_block, 'e_layers_local'))

            if 'e_layers_global' in dataset_block:
                E_LAYERS_GLOBAL = int(dataset_block.get('e_layers_global'))
            elif hasattr(dataset_block, 'e_layers_global'):
                E_LAYERS_GLOBAL = int(getattr(dataset_block, 'e_layers_global'))
            # Read perturbation evaluation parameters
            PERTURBATION_NUM_STEPS = 20
            PERTURBATION_POSITIVE_THRESHOLD = 0.75
            if 'perturbation_num_steps' in dataset_block:
                PERTURBATION_NUM_STEPS = int(dataset_block.get('perturbation_num_steps'))
            elif hasattr(dataset_block, 'perturbation_num_steps'):
                PERTURBATION_NUM_STEPS = int(getattr(dataset_block, 'perturbation_num_steps'))
            if 'perturbation_positive_threshold' in dataset_block:
                PERTURBATION_POSITIVE_THRESHOLD = float(dataset_block.get('perturbation_positive_threshold'))
            elif hasattr(dataset_block, 'perturbation_positive_threshold'):
                PERTURBATION_POSITIVE_THRESHOLD = float(getattr(dataset_block, 'perturbation_positive_threshold'))
            print(f"[Config] Loaded from config object: n_local={N_PROTOS_LOCAL}, n_global={N_PROTOS_GLOBAL}, n_protos={N_PROTOS}")
            print(f"[Config] d_model={D_MODEL}, patch_len={PATCH_LEN}, stride={STRIDE}, e_layers_local={E_LAYERS_LOCAL}, e_layers_global={E_LAYERS_GLOBAL}")
            print(f"[Config] perturbation_num_steps={PERTURBATION_NUM_STEPS}, perturbation_positive_threshold={PERTURBATION_POSITIVE_THRESHOLD}")
            print(f"[Config] dataset_block keys: {list(dataset_block.keys()) if hasattr(dataset_block, 'keys') else 'N/A'}")
        except (AttributeError, TypeError) as e:
            # Fallback to attribute access if dict access fails
            NUM_CLASSES = int(getattr(dataset_block, 'num_classes', NUM_CLASSES))
            SEQ_LEN = int(getattr(dataset_block, 'seq_len', SEQ_LEN))
            PROTO_LEN = int(getattr(dataset_block, 'proto_len', PROTO_LEN))
            N_PROTOS = int(getattr(dataset_block, 'num_prototypes', N_PROTOS))
            if hasattr(dataset_block, 'num_prototypes_local'):
                N_PROTOS_LOCAL = int(getattr(dataset_block, 'num_prototypes_local'))
            if hasattr(dataset_block, 'num_prototypes_global'):
                N_PROTOS_GLOBAL = int(getattr(dataset_block, 'num_prototypes_global'))
            # Read d_model, patch_len, stride from config (attribute access)
            if hasattr(dataset_block, 'd_model'):
                D_MODEL = int(getattr(dataset_block, 'd_model'))
            if hasattr(dataset_block, 'patch_len'):
                PATCH_LEN = int(getattr(dataset_block, 'patch_len'))
            if hasattr(dataset_block, 'stride'):
                STRIDE = int(getattr(dataset_block, 'stride'))
            if hasattr(dataset_block, 'e_layers_local'):
                E_LAYERS_LOCAL = int(getattr(dataset_block, 'e_layers_local'))
            if hasattr(dataset_block, 'e_layers_global'):
                E_LAYERS_GLOBAL = int(getattr(dataset_block, 'e_layers_global'))
            # Read perturbation evaluation parameters
            PERTURBATION_NUM_STEPS = 20
            PERTURBATION_POSITIVE_THRESHOLD = 0.75
            if hasattr(dataset_block, 'perturbation_num_steps'):
                PERTURBATION_NUM_STEPS = int(getattr(dataset_block, 'perturbation_num_steps'))
            if hasattr(dataset_block, 'perturbation_positive_threshold'):
                PERTURBATION_POSITIVE_THRESHOLD = float(getattr(dataset_block, 'perturbation_positive_threshold'))
            print(f"[Config] Loaded from config object (attr): n_local={N_PROTOS_LOCAL}, n_global={N_PROTOS_GLOBAL}, n_protos={N_PROTOS}, error={e}")
            print(f"[Config] d_model={D_MODEL}, patch_len={PATCH_LEN}, stride={STRIDE}, e_layers_local={E_LAYERS_LOCAL}, e_layers_global={E_LAYERS_GLOBAL}")
            print(f"[Config] perturbation_num_steps={PERTURBATION_NUM_STEPS}, perturbation_positive_threshold={PERTURBATION_POSITIVE_THRESHOLD}")
    
    # Priority 2: Fall back to YAML file if config object doesn't have values
    # Only read from YAML if config object was not provided or doesn't have the dataset block
    if dataset_block is None:
        ds_cfg_path = f"./configs/{dataset_name}.yaml"
        if os.path.exists(ds_cfg_path):
            ds_cfg = OmegaConf.load(ds_cfg_path)
            if dataset_name in ds_cfg:
                block = ds_cfg[dataset_name]
                NUM_CLASSES = int(block.get('num_classes', NUM_CLASSES))
                SEQ_LEN = int(block.get('seq_len', SEQ_LEN))
                PROTO_LEN = int(block.get('proto_len', PROTO_LEN))
                N_PROTOS = int(block.get('num_prototypes', N_PROTOS))
                if 'num_prototypes_local' in block:
                    N_PROTOS_LOCAL = int(block.get('num_prototypes_local'))
                if 'num_prototypes_global' in block:
                    N_PROTOS_GLOBAL = int(block.get('num_prototypes_global'))
                # Read d_model, patch_len, stride from YAML
                if 'd_model' in block:
                    D_MODEL = int(block.get('d_model'))
                if 'patch_len' in block:
                    PATCH_LEN = int(block.get('patch_len'))
                if 'stride' in block:
                    STRIDE = int(block.get('stride'))
                if 'e_layers_local' in block:
                    E_LAYERS_LOCAL = int(block.get('e_layers_local'))
                if 'e_layers_global' in block:
                    E_LAYERS_GLOBAL = int(block.get('e_layers_global'))
                # Read perturbation evaluation parameters
                PERTURBATION_NUM_STEPS = 20
                PERTURBATION_POSITIVE_THRESHOLD = 0.75
                if 'perturbation_num_steps' in block:
                    PERTURBATION_NUM_STEPS = int(block.get('perturbation_num_steps'))
                if 'perturbation_positive_threshold' in block:
                    PERTURBATION_POSITIVE_THRESHOLD = float(block.get('perturbation_positive_threshold'))
                print(f"[YAML] Loaded from YAML: n_local={N_PROTOS_LOCAL}, n_global={N_PROTOS_GLOBAL}, n_protos={N_PROTOS}")
                print(f"[YAML] d_model={D_MODEL}, patch_len={PATCH_LEN}, stride={STRIDE}, e_layers_local={E_LAYERS_LOCAL}, e_layers_global={E_LAYERS_GLOBAL}")
                print(f"[YAML] perturbation_num_steps={PERTURBATION_NUM_STEPS}, perturbation_positive_threshold={PERTURBATION_POSITIVE_THRESHOLD}")

    parser = argparse.ArgumentParser()
    # Experiment settings
    # Task settings
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--model_id", type=str,
                        default="dataset_name")  # Replace 'dataset_name' with the actual variable value
    parser.add_argument("--model", type=str, default="ProtoPTST")

    # Data settings
    parser.add_argument("--data", type=str,
                        default=config.dataset.name)  # Replace 'dataset_name' with the actual variable value
    parser.add_argument("--dataset_name", type=str, default=config.dataset.name)
    parser.add_argument("--root_path", type=str, default=config.base.root_dir)
    parser.add_argument("--two_class", type=int, default=0)
    parser.add_argument("--saliency", type=bool, default=False)
    parser.add_argument("--s_no", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)

    # Model settings for classification
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--e_layers_local", type=int, default=E_LAYERS_LOCAL)
    parser.add_argument("--e_layers_global", type=int, default=E_LAYERS_GLOBAL)
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", type=bool, default=True)
    parser.add_argument("--no_inter_attn", type=bool, default=False)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--patch_len", type=int, default=PATCH_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--sampling_rate", type=int, default=256)
    parser.add_argument("--patch_len_list", type=list, default=[2, 4, 8])
    parser.add_argument("--single_channel", type=bool, default=False)
    parser.add_argument("--augmentations", type=str, default="flip,shuffle,jitter,mask,drop")

    # Optimization settings
    parser.add_argument("--batch_size", type=int, default=8)  # Replace 'BATCH_SIZE' with the actual variable value
    parser.add_argument("--train_epochs", type=int, default=20)  # 100
    # parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)  # 5
    parser.add_argument("--des", type=str, default="Exp")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--use_amp", type=bool, default=False)
    parser.add_argument("--swa", type=bool, default=False)
    parser.add_argument("--d_layers", type=int, default=1)

    # GPU settings
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=str, default=0)  # Replace 'gpu_n' with the actual variable value
    parser.add_argument("--use_multi_gpu", type=bool, default=False)
    parser.add_argument("--devices", type=str, default=1)  # Replace 'gpu_n' with the actual variable value
    parser.add_argument("--device", type=str, default="cuda:0")  # Replace 'gpu_n' with the actual variable value

    # De-stationary projector parameters
    parser.add_argument("--p_hidden_dims", type=list, default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--num_workers", type=int, default=1)

    # Building model args
    parser.add_argument("--pred_len", type=int, default=0)
    parser.add_argument("--enc_in", type=int, default=1)  # Replace 'CHANNEL' with the actual variable value
    parser.add_argument("--num_class", type=int,
                        default=NUM_CLASSES)  # Replace 'CLASS_NUMBER' with the actual variable value

    # ProtoPTST args
    parser.add_argument("--num_prototypes", "-n_pro", type=int, default=N_PROTOS)  # YAML-overridable (for backward compatibility)
    parser.add_argument("--num_prototypes_local", type=int,
                        default=N_PROTOS_LOCAL if N_PROTOS_LOCAL is not None else N_PROTOS)  # Local prototypes count
    parser.add_argument("--num_prototypes_global", type=int,
                        default=N_PROTOS_GLOBAL if N_PROTOS_GLOBAL is not None else N_PROTOS)  # Global prototypes count
    parser.add_argument("--prototype_len", "-pro_len", type=int,
                        default=PROTO_LEN)  # Replace 'proto_len' with the actual variable value
    parser.add_argument("--prototype_init", default="kaiming", type=str)
    parser.add_argument("--prototype_activation", default="linear", type=str)
    parser.add_argument("--ablation", default="none", type=str,
                        help=["no_matching_loss", "no_variances_loss", 'no_prototype_layer'])
    parser.add_argument("--class_model_type", default="cnn")
    parser.add_argument("--equal_seg_len", default=0, type=int)
    parser.add_argument("--seq_method", default=None, type=str)
    
    # Perturbation-based evaluation parameters (for datasets without ground truth)
    parser.add_argument("--perturbation_num_steps", type=int, default=20,
                       help="Number of perturbation steps for evaluation (more steps = finer but slower)")
    parser.add_argument("--perturbation_positive_threshold", type=float, default=0.75,
                       help="Threshold for positive class in AUPRC calculation (0.75 = top 25%% as positive, range [0, 1])")

    ############ Parse arguments
    # do not read CLI; stick to defaults and YAML overrides
    args = parser.parse_args(args=[])
    
    # Override with config values if they exist (for perturbation parameters)
    try:
        if dataset_block is not None:
            if 'perturbation_num_steps' in dataset_block or (hasattr(dataset_block, 'perturbation_num_steps') and getattr(dataset_block, 'perturbation_num_steps', None) is not None):
                if 'perturbation_num_steps' in dataset_block:
                    args.perturbation_num_steps = int(dataset_block.get('perturbation_num_steps'))
                else:
                    args.perturbation_num_steps = int(getattr(dataset_block, 'perturbation_num_steps'))
            if 'perturbation_positive_threshold' in dataset_block or (hasattr(dataset_block, 'perturbation_positive_threshold') and getattr(dataset_block, 'perturbation_positive_threshold', None) is not None):
                if 'perturbation_positive_threshold' in dataset_block:
                    args.perturbation_positive_threshold = float(dataset_block.get('perturbation_positive_threshold'))
                else:
                    args.perturbation_positive_threshold = float(getattr(dataset_block, 'perturbation_positive_threshold'))
    except Exception as e:
        print(f"Warning: Could not load perturbation parameters from config: {e}")
    
    ## print(args)
    print("#" * 20)
    print(f"Args after parsing: num_prototypes={args.num_prototypes}, num_prototypes_local={args.num_prototypes_local}, num_prototypes_global={args.num_prototypes_global}")
    print(f"Perturbation params: num_steps={args.perturbation_num_steps}, positive_threshold={args.perturbation_positive_threshold}")
    print("#" * 20)
    ############ Load data

    # Add dataset to args
    args.dataset = config.dataset.name
    args.meta_dataset = config.dataset.meta_dataset

    # Inject base root path only (no other path params)
    base_cfg = getattr(config, 'base', {})
    args.root_path = getattr(base_cfg, 'root_dir', None)
    
    # Ensure proto_len is set for compatibility (YAML uses proto_len, argparse uses prototype_len)
    if not hasattr(args, 'proto_len') or args.proto_len is None:
        args.proto_len = args.prototype_len

    return args

## Module is intended to be imported by orchestration scripts; no CLI entry.