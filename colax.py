
import torch
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def _get_device(module):
    """Return the actual device of an nn.Module (works across PyTorch versions)."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device('cpu')



def moving_average_centered(time_series, window_size):
    """
        Centered moving average: window centered alignment.
    """
    padded_series = np.pad(time_series, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(padded_series, np.ones(window_size), 'valid') / window_size


def fill_short_negative_sequences(time_series, threshold=7):
    """
        Fill negative subsequences shorter than `threshold` with value 1.

        Args:
            time_series (np.ndarray): Input time series.
            threshold (int): Negative subsequences shorter than this will be filled to 1.

        Returns:
            np.ndarray: The modified time series.
        """
    filled_series = time_series - 1
    start = None
    for i, value in enumerate(filled_series):
        if value < 0:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                length = end - start
                if length < threshold:
                    time_series[start:end] = 2
                start = None
    return time_series


def segment_sequence_ones(a):

    segments = []
    start = None
    for i in range(len(a)):
        if a[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(a) - 1))
    torch.cuda.empty_cache()
    return segments


def get_equal_segments(seq_len, seg_length):

    segments = []
    start = 0
    while start < seq_len:
        end = min(start + seg_length, seq_len)
        segments.append([start, end - 1])
        start = end
    return segments


def _interpolate_to_len(tensor_2d, target_len):

    src_len = tensor_2d.shape[1]
    if src_len == target_len:
        return tensor_2d
    # [bs, src_len] → [bs, 1, src_len] → interpolate → [bs, 1, target_len] → [bs, target_len]
    out = torch.nn.functional.interpolate(
        tensor_2d.unsqueeze(1),
        size=target_len,
        mode='linear',
        align_corners=True,
    )
    return out.squeeze(1)


def map_patch_attn_to_time(memory_attn, seq_len, patch_len, stride):
    """
    Map patch-level memory_attn back to the original time-step space
    """
    bs, patch_num, num_global = memory_attn.shape

    patch_importance = memory_attn.max(dim=-1).values  # [bs, patch_num]

    global_importance = _interpolate_to_len(patch_importance, seq_len)

    return global_importance  # [bs, seq_len]


def map_patch_attn_to_time_fast(memory_attn, seq_len, patch_len, stride):
    """
    [fast] Map patch-level memory_attn back to the original time-step space
    """
    bs, patch_num, num_global = memory_attn.shape

    patch_importance = memory_attn.max(dim=-1).values  # [bs, patch_num]

    global_importance = _interpolate_to_len(patch_importance, seq_len)

    return global_importance  # [bs, seq_len]



def fuse_local_global_actions(actions_local, memory_attn, seq_len, patch_len, stride,
                               fuse_method='hybrid', alpha=0.5):
    """

    Args:

        fuse_method:
            - 'multiply':  fused = local * (1 + alpha * global_weight)
            - 'add':       fused = local + alpha * global_weight
            - 'gate':      fused = local * sigmoid(alpha * (global_weight - 0.5))
            - 'hybrid':    fused = local * (1 + alpha * global_weight) + beta * global_weight

    Returns:
        fused_actions: [bs, seq_len, num_prototypes_local] — 融合后的激活
    """
    if memory_attn is None or fuse_method == 'none':
        return actions_local

    actual_seq_len = actions_local.shape[1]

    try:
        global_importance = map_patch_attn_to_time_fast(
            memory_attn, actual_seq_len, patch_len, stride
        )
    except Exception:
        global_importance = map_patch_attn_to_time(
            memory_attn, actual_seq_len, patch_len, stride
        )

    
    if global_importance.shape[1] != actual_seq_len:
        global_importance = _interpolate_to_len(global_importance, actual_seq_len)

    
    g_min = global_importance.min(dim=1, keepdim=True).values
    g_max = global_importance.max(dim=1, keepdim=True).values
    g_range = (g_max - g_min).clamp(min=1e-8)
    global_weight = (global_importance - g_min) / g_range  # [bs, seq_len]

  
    global_weight = global_weight.unsqueeze(-1)  # [bs, seq_len, 1]

 
    # if fuse_method == 'multiply':
    #     fused = actions_local * (1.0 + alpha * global_weight)
    # elif fuse_method == 'add':
    #     fused = actions_local + alpha * global_weight
    # elif fuse_method == 'gate':
       
    #     gate = torch.sigmoid((global_weight - 0.5) * alpha * 10)
    #     fused = actions_local * gate
    # elif fuse_method == 'hybrid':
        
    #     beta = alpha * 0.3J
    #     fused = actions_local * (1.0 + alpha * global_weight)
    # else:
    #     fused = actions_local
    # fused = actions_local * (1.0 + alpha * global_weight)
    fused = actions_local * (1.0 + global_weight)

    return fused



def get_seg_fused_ProtopTST(signal, seg_model, fuse_method='hybrid', alpha=0.5):

    threshold = 1

    if seg_model is None:
        print("no seg model chosen")
        return [], None

    device = _get_device(seg_model)
    x_in = signal.reshape(1, -1, 1).to(device)

    model_outputs = seg_model(x_in, x_in, x_in, x_in)

    if len(model_outputs) == 6:
        out, actions, prototype, memory_attn, z_global, global_prototypes = model_outputs
    elif len(model_outputs) == 4:
        out, actions, prototype, memory_attn = model_outputs
    else:
        out, actions, prototype = model_outputs
        memory_attn = None

    seq_len = signal.shape[0]
    patch_len = getattr(seg_model, 'patch_len', 4)
    stride = getattr(seg_model, 'stride', 1)

    fused_actions = fuse_local_global_actions(
        actions, memory_attn, seq_len, patch_len, stride,
        fuse_method=fuse_method, alpha=alpha
    )


    actions_sum = torch.sum(fused_actions, dim=-1).reshape(-1)
    actions_sum = fill_short_negative_sequences(actions_sum)
    actions_sum_step = (actions_sum > threshold).int()
    segs = segment_sequence_ones(actions_sum_step)

    torch.cuda.empty_cache()
    return segs, actions_sum


def get_seg_fused_ProtopTST_SNC(signal, seg_model, fuse_method='hybrid', alpha=0.5):

    threshold = 0.4

    if seg_model is None:
        print("no seg model chosen")
        return [], None

    device = _get_device(seg_model)
    x_in = signal.reshape(1, -1, 1).to(device)


    model_outputs = seg_model(x_in, x_in, x_in, x_in)

    if len(model_outputs) == 6:
        out, actions, prototype, memory_attn, z_global, global_prototypes = model_outputs
    elif len(model_outputs) == 4:
        out, actions, prototype, memory_attn = model_outputs
    else:
        out, actions, prototype = model_outputs
        memory_attn = None

    seq_len = signal.shape[0]
    patch_len = getattr(seg_model, 'patch_len', 4)
    stride = getattr(seg_model, 'stride', 1)

    fused_actions = fuse_local_global_actions(
        actions, memory_attn, seq_len, patch_len, stride,
        fuse_method=fuse_method, alpha=alpha
    )

    actions_sum = fused_actions[:, :, 0].reshape(-1).cpu().detach().numpy()
    actions_sum = moving_average_centered(actions_sum, 100)
    actions_sum_step = (actions_sum > threshold)
    segs = segment_sequence_ones(actions_sum_step)

    torch.cuda.empty_cache()
    return segs, actions_sum


def get_seg_fused_unified(signal, seg_model, dataset_name,
                           fuse_method='hybrid', alpha=0.5):

    name = (dataset_name or "").lower() if isinstance(dataset_name, str) else ""
    if (name == "mitecg") or ("ecg" in name):
        segs, actions_sum = get_seg_fused_ProtopTST(
            signal, seg_model, fuse_method=fuse_method, alpha=alpha
        )
    else:
        segs, actions_sum = get_seg_fused_ProtopTST_SNC(
            signal, seg_model, fuse_method=fuse_method, alpha=alpha
        )

    if isinstance(actions_sum, np.ndarray):
        actions_sum = torch.tensor(
            actions_sum,
            dtype=signal.dtype,
            device=signal.device if hasattr(signal, 'device') else None,
        )
    return segs, actions_sum


class ScoreSubsequencesMine:


    def __init__(self, args, class_model, seg_model=None, seq_method="prototype"):
        self.class_model = class_model
        self.seq_method = seq_method
        self.seg_model = seg_model
        self.args = args

        self.equal_seg_len = args.equal_seg_len
        self.device = args.device


        self.fuse_method = getattr(args, 'fuse_method', 'hybrid')
        self.fuse_alpha = getattr(args, 'fuse_alpha', 0.5)

    def normalize_scores(self, score_list):
        if not score_list:
            return []
        max_score = max(score_list)
        if max_score == 0:
            return [0 for _ in score_list]
        return [score / max_score for score in score_list]

    def get_classification(self, signal):
        """signal: [T, dim] → classification logits"""
        signal = signal.reshape(-1, 1, 1)
        times = torch.arange(1, signal.shape[0] + 1, dtype=torch.float16).reshape(-1, 1)
        classification = self.class_model(signal.to(self.device), times.to(self.device))[0, :2]
        return classification

    def get_score_vector(self, signal, signal_idx, return_place_holder=True):

        if self.seq_method == "equal_seg":
            return_place_holder = False
            seq_lst = get_equal_segments(signal.shape[0], self.args.equal_seg_len)
            actions_sum = None
        elif self.seq_method == "prototype":
            seq_lst, actions_sum = get_seg_fused_unified(
                signal, self.seg_model,
                dataset_name=getattr(self.args, 'data', None),
                fuse_method=self.fuse_method,
                alpha=self.fuse_alpha,
            )
        else:
            print("!!!! seq_method error !!!")
            seq_lst, actions_sum = [], None

        score_list = []
        score_vector = torch.zeros((signal.shape))
        original_prediction = self.get_classification(signal)
        modified_series_list = []

        seq_lst = np.clip(seq_lst, None, self.seq_len - 1)

        for i, (start_idx, end_idx) in enumerate(seq_lst):
            modified_series = signal.clone()

            if self.args.ablation == "no_linaer_zero":
                modified_series[start_idx:end_idx, :] = 0
            elif self.args.ablation == "no_linear_random":
                modified_series[start_idx:end_idx, :] = torch.rand(
                    (end_idx - start_idx, 1, 1)
                )
            elif self.args.ablation == "no_linear_mean":
                modified_series[start_idx:end_idx, :] = torch.mean(signal)
            else:

                start_val = modified_series[start_idx]
                end_val = modified_series[end_idx]
                t = torch.linspace(0, 1, end_idx - start_idx,
                                   device=start_val.device, dtype=start_val.dtype)
                interpolated = start_val * (1 - t.unsqueeze(-1)) + end_val * t.unsqueeze(-1)
                modified_series[start_idx:end_idx, :] = interpolated

            modified_prediction = self.get_classification(modified_series)
            score = (original_prediction - modified_prediction).abs().sum().item()
            score_list.append(score)
            # modified_series_list.append(modified_series)

        normalized_scores = self.normalize_scores(score_list)

        for i, (start_idx, end_idx) in enumerate(seq_lst):
            score_vector[start_idx:end_idx, :] = normalized_scores[i]

        if self.args.ablation == "no_segment" and actions_sum is not None:
            actions_max = torch.max(actions_sum)
            actions_min = torch.min(actions_sum)
            actions_sum = (actions_sum - actions_min) / (actions_max - actions_min) + 1e-6
            action_len = actions_sum.shape[0]
            base = torch.zeros_like(score_vector)
            base[:action_len, :] = actions_sum.reshape(-1, 1)
            score_vector = base

        if return_place_holder:
            return score_vector, actions_sum if actions_sum is not None else torch.zeros(1)
        else:
            return score_vector

    def get_score_as_GTEXP(self, X, if_place_hodler=False, only_max=False):

        self.seq_len = X.shape[1]
        print("X.shape:", X.shape)

        scores_list = []
        place_holder_list = []
        sample_n = X.shape[0]

        with torch.no_grad():
            for i in tqdm(range(sample_n), desc="counting saliency score (fused)", position=0):
                signal = X[i, :, :]  # (T, dim)

                if self.args.seq_method == "equal_seg":
                    score_vector = self.get_score_vector(signal, i, return_place_holder=False)
                    place_holder = torch.zeros(1)
                else:
                    score_vector, place_holder = self.get_score_vector(signal, i)

                    if isinstance(place_holder, np.ndarray):
                        place_holder = torch.from_numpy(place_holder).float()

                    if place_holder.numel() > 0 and place_holder.max() > 0:
                        place_holder = place_holder.reshape(-1, 1)
                        place_holder = place_holder / torch.max(place_holder)

                        T = score_vector.shape[0]
                        if T < place_holder.shape[0]:
                            place_holder = place_holder[:T]
                        else:
                            padding_size = T - place_holder.shape[0]
                            place_holder = torch.cat([
                                place_holder,
                                torch.zeros(padding_size, 1, device=place_holder.device)
                            ], dim=0)

                        place_holder = place_holder.to(score_vector.device)
                        score_vector = score_vector + 0.1 * place_holder

                scores_list.append(score_vector.to(dtype=torch.float16))
                place_holder_list.append(place_holder)

                torch.cuda.empty_cache()
                del signal, score_vector, place_holder
                torch.cuda.empty_cache()

        GT_EXP = torch.stack(scores_list, dim=0)
        print("GT_EXP.shape:", GT_EXP.shape)
        return GT_EXP

    def get_score_as_GTEXP_para(self, X, place_hodler=False, only_max=False):

        self.seq_len = X.shape[1]
        sample_n = X.shape[0]

        def process_sample(i):
            signal = X[i, :, :]
            return self.get_score_vector(signal, i)

        results = Parallel(n_jobs=-1)(
            delayed(process_sample)(i)
            for i in tqdm(range(sample_n), desc="counting saliency score (fused, parallel)")
        )

        score_vectors, place_holder_list = zip(*results)
        GT_EXP = torch.stack(score_vectors, dim=0)
        return GT_EXP


def eval_colax_fused(pipeline, fuse_method='hybrid', alpha=0.5):

    import os


    pipeline.args.fuse_method = fuse_method
    pipeline.args.fuse_alpha = alpha

    seg_model = pipeline.load_seg_model()
    pipeline.args.saliency = True

    from get_data import get_saliency_data
    X, y, times, gt_exps = get_saliency_data(pipeline.args, return_dict=False)
    X, y, times, gt_exps = X.cpu(), y.cpu(), times.cpu(), gt_exps.cpu()
    class_model = pipeline.get_class_model(X)


    Scorer = ScoreSubsequencesMine(pipeline.args, class_model, seg_model)
    score = Scorer.get_score_as_GTEXP(X)


    from txai.utils.evaluation import ground_truth_xai_eval
    results_dict = ground_truth_xai_eval(score, gt_exps)
    for k, v in results_dict.items():
        print('\t{} = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))


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
            f"_fuse_{fuse_method}_alpha{alpha}{ablation_str}"
        )
        for k, v in results_dict.items():
            f.write(f' {k}:{np.mean(v):.4f}-{(np.std(v) / np.sqrt(len(v))):.4f}')
        f.write("\n")

    return results_dict


def interpolate_tensor(input_tensor, new_T=480, dim_to_change=0):

    # Swap dims to move T to the last position for interpolate
    last_dim = len(input_tensor.shape) - 1
    input_tensor = input_tensor.transpose(dim_to_change, last_dim)  # Now (batch_size, D, T)

    # Use interpolate
    output_tensor = torch.nn.functional.interpolate(input_tensor, size=new_T, mode='linear', align_corners=True)
    print(output_tensor.shape)

    # Swap dims back
    output_tensor = output_tensor.transpose(dim_to_change, last_dim)  # Now (batch_size, new_T, D)

    return output_tensor


def get_seg_ProtopTST(signal, seg_model):  # for ECG dataset
    """
    signal: [seq_len, dim]
    """
    threshold = 1

    if seg_model is not None:
        ProtopTST = seg_model
    else:
        print("no seg model chooesed")

    device = ProtopTST.device

    x_in = signal.reshape(1, -1, 1).to(device)

    # Handle both old model (3 returns) and hybrid model (4 returns)
    model_outputs = ProtopTST(x_in, x_in, x_in, x_in)

    if len(model_outputs) == 4:
        # ProtoPTST_hybrid: output, activations_local, prototype_local, memory_attn
        out, actions, prototype, memory_attn = model_outputs
    elif len(model_outputs) == 6:
        # ProtoPTST_hybrid_new: output, activations_local, prototype_local, memory_attn, z_global, global_prototypes
        out, actions, prototype, memory_attn, _, _ = model_outputs
    else:
        # Original ProtoPTST: output, activations, prototype
        out, actions, prototype = model_outputs

    actions_sum = torch.sum(actions, dim=-1).reshape(-1)

    actions_sum = fill_short_negative_sequences(actions_sum)

    actions_sum_step = (actions_sum > threshold).int()

    segs = segment_sequence_ones(actions_sum_step)

    torch.cuda.empty_cache()

    return segs, actions_sum


def get_seg_ProtopTST_SNC(signal, seg_model):  # for mcce,mcch,mtce,mtch dataset
    """
    signal: [seq_len, dim]
    """
    threshold = 0.4

    if seg_model is not None:
        ProtopTST = seg_model
    else:
        print("no seg model chooesed")

    device = ProtopTST.device

    x_in = signal.reshape(1, -1, 1).to(device)

    # Handle both old model (3 returns) and hybrid model (4 returns)
    model_outputs = ProtopTST(x_in, x_in, x_in, x_in)
    if len(model_outputs) == 6:
        out, actions, prototype, memory_attn, z_global, global_prototypes = model_outputs
    elif len(model_outputs) == 4:
        out, actions, prototype, memory_attn = model_outputs
    else:
        out, actions, prototype = model_outputs

    actions_sum = actions[:, :, 0].reshape(-1).cpu().detach().numpy()

    actions_sum = moving_average_centered(actions_sum, min(100, max(5, len(actions_sum) // 5)))

    actions_sum_step = (actions_sum > threshold)

    segs = segment_sequence_ones(actions_sum_step)

    torch.cuda.empty_cache()

    return segs, actions_sum


def get_seg_unified(signal, seg_model, dataset_name):
    """
    Unified segmentation wrapper that dispatches by dataset name and
    standardizes the return types.

    Returns:
        segs: list of (start_idx, end_idx)
        actions_sum: torch.Tensor of shape [T]
    """
    name = (dataset_name or "").lower() if isinstance(dataset_name, str) else ""
    if (name == "mitecg") or ("ecg" in name):
        segs, actions_sum = get_seg_ProtopTST(signal, seg_model)
    elif name in ("mcce", "mcch", "mtce", "mtch"):
        segs, actions_sum = get_seg_ProtopTST_SNC(signal, seg_model)
    else:
        segs, actions_sum = get_seg_ProtopTST(signal, seg_model)

    if isinstance(actions_sum, np.ndarray):
        actions_sum = torch.tensor(
            actions_sum,
            dtype=signal.dtype,
            device=signal.device if hasattr(signal, 'device') else None,
        )
    return segs, actions_sum


class ScoreSubsequences(object):
    def __init__(self, args, class_model, seg_model=None, seq_method="prototype") -> None:
        self.class_model = class_model
        self.seq_method = seq_method
        self.seg_model = seg_model
        self.args = args

        self.equal_seg_len = args.equal_seg_len
        self.device = args.device

    def normalize_scores(self, score_list):
        if not score_list:
            return []
        max_score = max(score_list)
        if max_score == 0:
            return [0 for _ in score_list]
        return [score / max_score for score in score_list]

    def get_classification(self, signal):
        ## signal: [T,dim]
        signal = signal.reshape(-1, 1, 1)

        times = torch.arange(1, signal.shape[0] + 1, dtype=torch.float16).reshape(-1, 1)

        classification = self.class_model(signal.to(self.device), times.to(self.device))[0, :2]
        return classification

    def get_score_vector(self, signal, signal_idx, return_place_holder=True):
        ## torch tensor: signal: [T,dim]
        if self.seq_method == "equal_seg":
            return_place_holder = False
            seq_lst = get_equal_segments(signal.shape[0], self.args.equal_seg_len)
        elif self.seq_method == "prototype":
            seq_lst, actions_sum = get_seg_unified(signal, self.seg_model, getattr(self.args, 'data', None))
        else:
            print("!!!! seq_method error !!!")

        score_list = []
        score_vector = torch.zeros((signal.shape))

        original_prediction = self.get_classification(signal)

        modified_series_list = []

        seq_lst = np.clip(seq_lst, None, self.seq_len - 1)

        for i, (start_idx, end_idx) in enumerate(seq_lst):
            modified_series = signal.clone()

            if self.args.ablation == "no_linaer_zero":
                modified_series[start_idx:end_idx, :] = 0
            if self.args.ablation == "no_linear_random":
                modified_series[start_idx:end_idx, :] = torch.rand((end_idx - start_idx, 1, 1))
            if self.args.ablation == "no_linear_mean":
                modified_series[start_idx:end_idx, :] = torch.mean(signal)
            else:
                start_val = modified_series[start_idx]
                end_val = modified_series[end_idx]
                t = torch.linspace(0, 1, end_idx - start_idx, device=start_val.device, dtype=start_val.dtype)
                interpolated = start_val * (1 - t.unsqueeze(-1)) + end_val * t.unsqueeze(-1)
                modified_series[start_idx:end_idx, :] = interpolated

            modified_prediction = self.get_classification(modified_series)

            score = (original_prediction - modified_prediction).abs().sum().item()

            score_list.append(score)

            modified_series_list.append(modified_series)

        normalized_scores = self.normalize_scores(score_list)

        for i, (start_idx, end_idx) in enumerate(seq_lst):
            score_vector[start_idx:end_idx, :] = normalized_scores[i]

        if self.args.ablation == "no_segment":
            max = torch.max(actions_sum)
            min = torch.min(actions_sum)

            actions_sum = (actions_sum - min) / (max - min) + 1e-6

            action_len = actions_sum.shape[0]
            base = torch.zeros_like(score_vector)

            base[:action_len, :] = actions_sum.reshape(-1, 1)
            score_vector = base

        if return_place_holder:
            return score_vector, actions_sum
        else:
            return score_vector

    def get_score_as_GTEXP(self, X, if_place_hodler=False, only_max=False):
        ## X:[N,T,dim] batch-first
        self.seq_len = X.shape[1]
        print("X.shape:", X.shape)

        scores_list = []
        place_holder_list = []

        sample_n = X.shape[0]

        with torch.no_grad():
            for i in tqdm(range(sample_n), desc="counting saliency score", position=0):
                signal = X[i, :, :]
                if only_max:
                    score_vector, place_holder = self.get_score_vector_max(signal, i)
                if self.args.seq_method == "equal_seg":
                    score_vector = self.get_score_vector(signal, i)
                    place_holder = []
                else:
                    score_vector, place_holder = self.get_score_vector(signal, i)
                    place_holder = place_holder.reshape(-1, 1)

                    if isinstance(place_holder, np.ndarray):
                        place_holder = torch.from_numpy(place_holder).float()

                    place_holder = place_holder / torch.max(place_holder)

                    T = score_vector.shape[0]
                    if T < place_holder.shape[0]:
                        place_holder = place_holder[:T]
                    else:
                        padding_size = T - place_holder.shape[0]
                        place_holder = torch.cat([place_holder, torch.zeros(padding_size, 1, device=place_holder.device)], dim=0)

                    place_holder = place_holder.to(score_vector.device)
                    score_vector = score_vector + 0.1 * place_holder

                scores_list.append(score_vector.to(dtype=torch.float16))
                place_holder_list.append(place_holder)

                torch.cuda.empty_cache()

                del signal, score_vector, place_holder
                torch.cuda.empty_cache()

        GT_EXP = torch.stack(scores_list, dim=0)
        print("GT_EXP.shape:", GT_EXP.shape)
        return GT_EXP

    def get_score_as_GTEXP_para(self, X, place_hodler=False, only_max=False):
        self.seq_len = X.shape[1]
        sample_n = X.shape[0]

        def process_sample(i):
            signal = X[i, :, :]
            if only_max:
                return self.get_score_vector_max(signal, i)
            else:
                return self.get_score_vector(signal, i)

        results = Parallel(n_jobs=-1)(delayed(process_sample)(i) for i in tqdm(range(sample_n), desc="counting saliency score"))

        score_vectors, place_holder_list = zip(*results)

        GT_EXP = torch.stack(score_vectors, dim=0)

        return GT_EXP



