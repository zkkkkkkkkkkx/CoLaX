"""
ProtoPTST_hybrid_separated with Global Segment Extraction

This file extends ProtoPTST_hybrid_separated to support segment extraction
from global prototypes, providing complementary interpretability to local prototypes.

Key additions:
1. Global segment extraction based on memory_attn
2. Patch-to-timestep mapping utilities
3. Global prototype loss similar to local prototype loss
"""

import torch
from torch import nn
import sys

from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import PatchEmbedding
from .models import PatchTST
from torch.nn import functional as F
from ..MemoryModule import MemoryModule
import numpy as np


class Model(PatchTST.Model):
    """
    Hybrid Prototype Model with SEPARATED embedding and encoder + Global Segment Extraction:
    1. Local Prototypes: Patch-based time series segments (original ProtoPTST)
       - Uses patch_embedding_local and encoder_local
       - Supports segment extraction from activations
    2. Global Prototypes: MemoryModule-based global feature vectors
       - Uses patch_embedding_global and encoder_global
       - NOW SUPPORTS segment extraction from memory_attn

    This version adds global segment extraction capabilities for enhanced interpretability.
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding = configs.stride
        self.patch_num = (self.seq_len - self.patch_len + self.padding) // self.stride + 1
        self.d_model = configs.d_model
        self.device = configs.device
        self.prototype_num_selected = 1

        # ========== Local Prototypes (Patch-based) ==========
        # Support num_prototypes_local if provided, otherwise use num_prototypes for backward compatibility
        self.num_prototypes_local = getattr(configs, 'num_prototypes_local', configs.num_prototypes)
        self.num_classes = configs.num_class
        self.prototype_len = configs.prototype_len
        # Use num_prototypes_local for prototype_shape
        num_protos_for_shape = getattr(configs, 'num_prototypes_local', configs.num_prototypes)
        self.prototype_shape = [num_protos_for_shape, self.prototype_len,
                                configs.enc_in]  # [num_prototypes_local x prototype_len x enc_in]
        self.prototype_activation_function = 'log'

        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape),
                                              requires_grad=True)  # trainable parameter

        if configs.prototype_init == 'kaiming':
            self.prototype_vectors = nn.init.kaiming_normal_(self.prototype_vectors)
        elif configs.prototype_init == 'xavier':
            self.prototype_vectors = nn.init.xavier_normal_(self.prototype_vectors)
        else:
            print("random init prototype")

        print("self.seq_len", self.seq_len)

        # ========== SEPARATED Local Embedding and Encoder ==========
        # Create independent embedding and encoder for local prototypes
        self.patch_embedding_local = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.stride, configs.stride, configs.dropout
        )

        self.encoder_local = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers_local)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # ========== Global Prototypes (MemoryModule-based) ==========
        # Number of global memory items (prototypes)
        self.num_prototypes_global = getattr(configs, 'num_prototypes_global', configs.num_prototypes)

        # Initialize MemoryModule for global prototypes
        # MemoryModule expects: n_memory, fea_dim, shrink_thres, device, phase_type
        phase_type = getattr(configs, 'phase_type', 'train')
        self.memory_module = MemoryModule(
            n_memory=self.num_prototypes_global,
            fea_dim=self.d_model,  # Use d_model as feature dimension
            shrink_thres=getattr(configs, 'memory_shrink_thres', 0.0025),
            device=self.device,
            memory_init_embedding=None,
            phase_type=phase_type,
            dataset_name=getattr(configs, 'dataset_name', None)
        )

        # ========== SEPARATED Global Embedding and Encoder ==========
        # Create independent embedding and encoder for global prototypes
        self.patch_embedding_global = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.stride, configs.stride, configs.dropout
        )

        self.encoder_global = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers_global)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # ========== Projection Layers ==========
        # Compute expected input dimension for local prototypes
        expected_input_dim_local = (self.seq_len - 1) * self.num_prototypes_local

        # Add adaptive pooling to handle dynamic-length input
        self.adaptive_pool = nn.AdaptiveAvgPool1d(expected_input_dim_local // self.num_prototypes_local)


        # Dimension for global features after aggregation
        global_feature_dim = self.d_model * 2  # MemoryModule concatenates query + memory: 2*d_model

        # Fusion layer: combine local and global features
        # Option 1: Concatenate
        fusion_dim = expected_input_dim_local + global_feature_dim

        # ========== Temporal Evidence Accumulator ==========
        # Whether to use temporal evidence accumulation between prototype activations and classification
        # When enabled, probabilities are based on accumulated evidence over time instead of instantaneous activations.
        self.use_temporal_evidence_accumulator = getattr(
            configs, 'use_temporal_evidence_accumulator', True
        )

        # Projection layer: maps weighted local activations to class logits
        self.projection = nn.Linear(
            expected_input_dim_local, configs.num_class
        )

        self.linear_1 = nn.Linear(configs.d_model, 1)

        # Attention mechanism: compute similarity between global and local features
        # Query: projected from global feature, Key: projected from local feature
        # Then compute attention weights based on similarity
        self.linear_global_q = nn.Linear(global_feature_dim, expected_input_dim_local)  # Query projection
        self.linear_global_k = nn.Linear(expected_input_dim_local,
                                         expected_input_dim_local)  # Key projection (optional, can use identity)

        # Temperature parameter for attention (optional, for scaling similarity)
        self.attention_temp = getattr(configs, 'attention_temp', 1.0)

        # CNN branch for local prototypes to capture local temporal patterns
        prototype_cnn_kernel = getattr(configs, 'prototype_cnn_kernel', 3)
        prototype_cnn_layers = getattr(configs, 'prototype_cnn_layers', 2)
        prototype_cnn_dropout = getattr(configs, 'prototype_cnn_dropout', 0.0)

        if prototype_cnn_layers <= 0:
            self.prototype_local_cnn = None
        else:
            cnn_blocks = []
            in_channels = configs.enc_in
            for _ in range(prototype_cnn_layers):
                cnn_blocks.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=configs.enc_in,
                        kernel_size=prototype_cnn_kernel,
                        padding=prototype_cnn_kernel // 2,
                        bias=True,
                    )
                )
                cnn_blocks.append(nn.ReLU(inplace=True))
                in_channels = configs.enc_in
            self.prototype_local_cnn = nn.Sequential(*cnn_blocks)

        self.prototype_cnn_dropout = (
            nn.Dropout(prototype_cnn_dropout)
            if prototype_cnn_dropout > 0 and self.prototype_local_cnn is not None
            else None
        )

        self.args = configs

        # ========== Global Segment Extraction Configuration ==========
        # Enable/disable global segment extraction
        self.enable_global_segment = getattr(configs, 'enable_global_segment', True)
        # Global segment length (can be different from local prototype_len)
        self.global_segment_len = getattr(configs, 'global_segment_len', self.patch_len)


    def patch_to_timestep_range(self, patch_idx):
        """
        Map patch index to original timestep range.

        Args:
            patch_idx: Patch index (0-indexed)

        Returns:
            start: Start timestep (inclusive)
            end: End timestep (exclusive)
        """
        start = patch_idx * self.stride
        end = start + self.patch_len
        return start, end

    def prototype_layer(self, prototype, prototype_patch=True, if_normalize=False):
        """
        Process local prototypes: [num_prototypes x prototype_len x enc_in]
        Uses LOCAL embedding and encoder (separated from global)
        """
        # Normalization from Non-stationary Transformer
        if if_normalize:
            means = prototype.mean(dim=1, keepdim=True).detach()
            prototype = prototype - means
            stdev = prototype.std(dim=1, keepdim=True, unbiased=False) + 1e-5
            prototype = prototype / stdev

        prototype_enc = prototype  # [num_prototypes, prototype_len, enc_in]

        if prototype_patch:
            # do patching and embedding
            prototype_enc = prototype_enc.permute(0, 2, 1)  # [num_prototypes, enc_in, prototype_len]

            # CNN for local temporal patterns
            if hasattr(self, 'prototype_local_cnn') and self.prototype_local_cnn is not None:
                local_features = self.prototype_local_cnn(prototype_enc)
                if self.prototype_cnn_dropout is not None:
                    local_features = self.prototype_cnn_dropout(local_features)
                prototype_enc = prototype_enc + local_features

            # Use LOCAL embedding and encoder
            prototype_out, n_vars = self.patch_embedding_local(prototype_enc)  # [num_prototypes, patch_num, d_model]

            # Encoder (LOCAL)
            prototype_out, attns = self.encoder_local(prototype_out)  # [num_prototypes, patch_num, d_model]
            prototype_out = self.linear_1(prototype_out)  # [num_prototypes, patch_num, 1]

        return prototype_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        input:
        x_enc: [bs x seq_len x nvars]

        output: [bs x num_classes]
        """
        x = x_enc  # [bs, seq_len, enc_in]

        # ========== Local Prototypes Processing ==========
        prototype_local = self.prototype_layer(self.prototype_vectors)  # [num_prototypes, patch_num, 1]
        if self.args.ablation == 'no_prototype_layer':
            prototype_local = self.prototype_vectors

        # Local prototype distance calculation
        activations_local = self.convolution_distance_x_as_input(x, prototype_local)  # [bs x seq_len x num_prototypes]

        # Note: Local prototypes are updated via gradient descent through seg_prototype_loss (backpropagation)
        # Global prototypes (MemoryModule) are updated automatically in memory_module.forward() via update()

        # Reorder to [bs x num_prototypes x seq_len] for temporal processing
        activations_for_pool = activations_local.permute(0, 2, 1)  # [bs x num_prototypes x seq_len]

        # ========== Temporal Evidence Accumulation ==========
        # Instead of using raw instantaneous activations, accumulate evidence over time.
        # This makes the downstream probability depend on integrated evidence, not just a snapshot.
        if self.use_temporal_evidence_accumulator:
            # Cumulative sum along the temporal dimension acts as an evidence integrator.
            # Each timestep t now represents sum_{τ<=t} activation(τ).
            evidence_seq = torch.cumsum(activations_for_pool, dim=2)  # [bs x num_prototypes x seq_len]
        else:
            # Fallback: keep original instantaneous activations
            evidence_seq = activations_for_pool  # [bs x num_prototypes x seq_len]

        # Use adaptive pooling to standardize sequence length on accumulated evidence
        expected_seq_len = (self.seq_len - 1)
        pooled_evidence = self.adaptive_pool(evidence_seq)  # [bs x num_prototypes x expected_seq_len]

        # Reshape local features (now evidence-based) to a flat vector
        features_local = pooled_evidence.reshape(
            pooled_evidence.shape[0], -1
        )  # [bs x (num_prototypes * expected_seq_len)]

        # ========== Global Prototypes Processing (MemoryModule) ==========
        # Extract global features from encoder
        # Process input through encoder to get features
        x_for_encoder = x.permute(0, 2, 1)  # [bs, enc_in, seq_len]

        # Patching and embedding (using GLOBAL embedding and encoder)
        enc_out, n_vars = self.patch_embedding_global(x_for_encoder)  # [bs * nvars, patch_num, d_model]

        # Encoder (GLOBAL)
        enc_out, attns = self.encoder_global(enc_out)  # [bs * nvars, patch_num, d_model]

        # Reshape: [bs * nvars, patch_num, d_model] -> [bs, nvars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # For MemoryModule, we need [bs, L, C] where L can be patch_num
        # Use the first variable if multi-variate, or average across variables
        if n_vars > 1:
            # Average across variables or take first variable
            enc_out_global = enc_out.mean(dim=1)  # [bs, patch_num, d_model]
        else:
            enc_out_global = enc_out.squeeze(1)  # [bs, patch_num, d_model]

        # Pass through MemoryModule
        memory_output = self.memory_module(enc_out_global)  # dict with 'output', 'attn', 'memory_init_embedding'
        features_global = memory_output['output']  # [bs, patch_num, 2*d_model]
        memory_attn = memory_output['attn']  # [bs, patch_num, num_prototypes_global]

        features_global = features_global.mean(dim=1)  # [bs, 2*d_model] - global representation g


        query = self.linear_global_q(features_global)  # [bs, expected_input_dim_local] - query from global

        key = self.linear_global_k(features_local)  # [bs, expected_input_dim_local] - key from local

        similarity = query * key  # [bs, expected_input_dim_local] - element-wise similarity

        gate_weights = torch.sigmoid(similarity / self.attention_temp)  # [bs, expected_input_dim_local]

        weighted_local_activation = gate_weights * features_local  # [bs, expected_input_dim_local]


        # ========== Classification ==========
        output = self.projection(weighted_local_activation)  # [bs, num_classes]

        z_global = enc_out_global.mean(dim=1)  # [bs, d_model]

        # Get global prototypes from memory module
        global_prototypes = self.memory_module.mem
        return output, activations_local, prototype_local, memory_attn, z_global, global_prototypes

    def convolution_distance_x_as_input(self, x, prototype_vectors):
        """
        Compute local prototype activations using convolution
        x: [bs x seq_len x nvars]
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        x = x.permute(0, 2, 1)  # [bs, nvars, seq_len]

        prototype_vectors = prototype_vectors.permute(0, 2, 1)  # [num_prototypes, 1, patch_num]
        prototype_vectors = prototype_vectors.to(x.device)

        # Perform convolution
        conv_out = F.conv1d(x, prototype_vectors, padding=(prototype_vectors.shape[-1] - 1) // 2,
                            stride=1)  # [bs, num_prototypes, seq_len]

        # Adjust output shape and apply normalization
        conv_out = conv_out.permute(0, 2, 1)  # [bs, seq_len, num_prototypes]

        layer_norm = nn.LayerNorm(conv_out.shape[1:]).to(self.device)
        conv_out = layer_norm(conv_out)

        conv_out = F.leaky_relu(conv_out, negative_slope=0.01)

        return conv_out

    def seg_prototype_loss(self, activations, x, prototype_vectors, outputs, select_prototype=True,
                           no_variance_loss=False):
        """
        Loss for local prototypes (same as original)
        activation: [bs x seq_len x num_prototypes]
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        if self.args.ablation == 'no_variances_loss':
            no_variance_loss = True

        batch_size = activations.shape[0]
        prototype_len = prototype_vectors.shape[1]
        prototype_num = self.num_prototypes_local

        half_len = prototype_len // 2
        x = x.reshape(batch_size, -1)

        subsequence_all_prototype = torch.tensor([]).to(self.device)
        for p_index in range(prototype_num):
            activation = activations[:, :, p_index].reshape(batch_size, -1)

            top_k_values, top_k_indices = torch.topk(activation, 1, dim=1)  # [bs x 1]

            subsequences = torch.zeros(batch_size, prototype_len).to(self.device)

            for i in range(batch_size):
                center = top_k_indices[i, 0].item()
                start = max(0, center - half_len)
                end = min(self.seq_len, center + half_len)

                sub_seq = x[i, start:end]

                if len(sub_seq) < prototype_len:
                    if start == 0:
                        sub_seq = torch.cat((torch.zeros(prototype_len - len(sub_seq)).to(self.device), sub_seq))
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(prototype_len - len(sub_seq)).to(self.device)))

                mean = sub_seq.mean()
                sub_seq = sub_seq - mean

                subsequences[i] = sub_seq

            subsequences = subsequences.unsqueeze(0)  # [1 x bs x prototype_len]

            subsequence_all_prototype = torch.cat((subsequence_all_prototype, subsequences),
                                                  dim=0)  # [num_prototypes x bs x prototype_len]

        subsequence_all_prototype = subsequence_all_prototype.permute(1, 0, 2)  # [bs x num_prototypes x prototype_len]

        prototype_repeat = prototype_vectors.permute(2, 0, 1)  # [1, num_prototypes, patch_num]
        prototype_repeat = prototype_repeat.repeat(batch_size, 1, 1)

        if select_prototype and self.num_prototypes_local > 1:
            indices = self.prototype_selection_class(outputs, prototype_vectors)  # [bs x 1]
            indices = indices.unsqueeze(-1).expand(-1, -1, prototype_len)

            subsequence_all_prototype = torch.gather(subsequence_all_prototype, dim=1, index=indices)
            prototype_repeat = torch.gather(prototype_repeat, dim=1, index=indices)

        prototype_loss = torch.nn.functional.mse_loss(subsequence_all_prototype, prototype_repeat)
        variance_loss = self.total_variance_loss(prototype_vectors)

        if self.args.ablation == 'no_matching_loss':
            prototype_loss = 0

        if self.num_prototypes_local == 1:
            return prototype_loss

        if no_variance_loss:
            return prototype_loss
        else:
            return self.dynamic_loss(prototype_loss, variance_loss)

    def seg_global_prototype_loss(self, memory_attn, x, global_prototypes, outputs, select_prototype=True):
        """
        Loss for global prototypes based on segment extraction.
        Similar to seg_prototype_loss but works on patch-level attention.

        Args:
            memory_attn: [bs, patch_num, num_prototypes_global] - Attention weights from memory module
            x: [bs, seq_len, nvars] - Original input time series
            global_prototypes: [num_prototypes_global, d_model] - Global prototype vectors
            outputs: [bs, num_classes] - Model predictions
            select_prototype: Whether to select one prototype per sample based on output

        Returns:
            prototype_loss: MSE loss between extracted segments and prototype representations
        """
        if not self.enable_global_segment:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        batch_size = memory_attn.shape[0]
        patch_num = memory_attn.shape[1]
        num_prototypes = memory_attn.shape[2]

        # Aggregate attention over patches to get sample-level prototype weights
        attn_weights = memory_attn.mean(dim=1)  # [bs, num_prototypes_global]

        # Find the most relevant prototype for each sample
        if select_prototype and num_prototypes > 1:
            # Option 1: Use output class to select prototype
            prototype_indices = self.prototype_selection_class(outputs, None)  # [bs, 1]
            # Option 2: Use attention weights to select prototype
            # prototype_indices = torch.argmax(attn_weights, dim=1, keepdim=True)  # [bs, 1]
        else:
            # Use all prototypes
            prototype_indices = None

        # Find the patch with highest attention for each prototype
        # memory_attn: [bs, patch_num, num_prototypes_global]
        if prototype_indices is not None:
            # Select specific prototype for each sample
            selected_attn = torch.gather(
                memory_attn,
                dim=2,
                index=prototype_indices.unsqueeze(1).expand(-1, patch_num, -1)
            ).squeeze(2)  # [bs, patch_num]
        else:
            # Use prototype with highest attention
            selected_attn = memory_attn.max(dim=2)[0]  # [bs, patch_num]

        # Find top-k patches with highest attention
        top_k_patches = 1  # Can be made configurable
        top_k_values, top_k_indices = torch.topk(selected_attn, top_k_patches, dim=1)  # [bs, top_k]

        # Extract segments from original time series based on patch indices
        x_flat = x.reshape(batch_size, -1)  # [bs, seq_len] (assuming single variable or averaged)
        if x.dim() == 3 and x.shape[2] > 1:
            # Multi-variate: average across variables
            x_flat = x.mean(dim=2)  # [bs, seq_len]

        subsequence_all_prototype = torch.tensor([]).to(self.device)

        for p_idx in range(num_prototypes):
            if prototype_indices is not None:
                # Only process selected prototype
                mask = (prototype_indices.squeeze() == p_idx)
                if not mask.any():
                    continue
                batch_indices = torch.where(mask)[0]
            else:
                batch_indices = torch.arange(batch_size, device=self.device)

            subsequences = torch.zeros(batch_size, self.global_segment_len).to(self.device)

            for i in batch_indices:
                # Get patch index with highest attention
                patch_idx = top_k_indices[i, 0].item()

                # Map patch index to timestep range
                start, end = self.patch_to_timestep_range(patch_idx)

                # Extract segment from original time series
                # Use center of patch as reference, extract segment_len around it
                center = (start + end) // 2
                half_seg_len = self.global_segment_len // 2
                seg_start = max(0, center - half_seg_len)
                seg_end = min(self.seq_len, center + half_seg_len)

                sub_seq = x_flat[i, seg_start:seg_end]

                # Pad if necessary
                if len(sub_seq) < self.global_segment_len:
                    pad_len = self.global_segment_len - len(sub_seq)
                    if seg_start == 0:
                        sub_seq = torch.cat((torch.zeros(pad_len).to(self.device), sub_seq))
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(pad_len).to(self.device)))

                # Normalize segment
                mean = sub_seq.mean()
                sub_seq = sub_seq - mean

                subsequences[i] = sub_seq

            subsequences = subsequences.unsqueeze(0)  # [1, bs, global_segment_len]
            subsequence_all_prototype = torch.cat(
                (subsequence_all_prototype, subsequences),
                dim=0
            )  # [num_prototypes, bs, global_segment_len]

        subsequence_all_prototype = subsequence_all_prototype.permute(1, 0,
                                                                      2)  # [bs, num_prototypes, global_segment_len]


        # For now, we'll use a simplified loss: encourage segments to be diverse
        # This is a placeholder - can be enhanced with learned projections
        if num_prototypes > 1:
            # Compute pairwise similarity between segments
            segments_flat = subsequence_all_prototype.reshape(batch_size, num_prototypes, -1)
            # Normalize
            segments_norm = F.normalize(segments_flat, p=2, dim=2)
            # Compute similarity matrix
            similarity = torch.bmm(segments_norm, segments_norm.transpose(1, 2))  # [bs, num_prototypes, num_prototypes]
            # Remove diagonal
            mask = ~torch.eye(num_prototypes, dtype=torch.bool, device=self.device).unsqueeze(0)
            off_diag_sim = similarity[mask.expand(batch_size, -1, -1)]
            # Encourage diversity (minimize similarity)
            prototype_loss = off_diag_sim.mean()
        else:
            prototype_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        return prototype_loss

    def extract_global_segments(self, memory_attn, x, top_k=1):
        """
        Extract time series segments corresponding to global prototype activations.

        Args:
            memory_attn: [bs, patch_num, num_prototypes_global] - Attention weights
            x: [bs, seq_len, nvars] - Original input time series
            top_k: Number of top patches to consider per prototype

        Returns:
            segments: Dict with keys:
                - 'segments': [bs, num_prototypes_global, segment_len] - Extracted segments
                - 'patch_indices': [bs, num_prototypes_global, top_k] - Patch indices used
                - 'timestep_ranges': List of (start, end) tuples for each segment
        """
        if not self.enable_global_segment:
            return None

        batch_size = memory_attn.shape[0]
        patch_num = memory_attn.shape[1]
        num_prototypes = memory_attn.shape[2]

        # Prepare input
        x_flat = x.reshape(batch_size, -1)
        if x.dim() == 3 and x.shape[2] > 1:
            x_flat = x.mean(dim=2)  # [bs, seq_len]

        # Find top-k patches for each prototype
        top_k_values, top_k_indices = torch.topk(memory_attn, top_k, dim=1)  # [bs, top_k, num_prototypes_global]
        top_k_indices = top_k_indices.permute(0, 2, 1)  # [bs, num_prototypes_global, top_k]

        # Extract segments
        segments = torch.zeros(batch_size, num_prototypes, self.global_segment_len).to(self.device)
        timestep_ranges = []

        for i in range(batch_size):
            sample_ranges = []
            for p_idx in range(num_prototypes):
                # Get patch index
                patch_idx = top_k_indices[i, p_idx, 0].item()

                # Map to timestep range
                start, end = self.patch_to_timestep_range(patch_idx)
                center = (start + end) // 2
                half_seg_len = self.global_segment_len // 2
                seg_start = max(0, center - half_seg_len)
                seg_end = min(self.seq_len, center + half_seg_len)

                sample_ranges.append((seg_start, seg_end))

                # Extract segment
                sub_seq = x_flat[i, seg_start:seg_end]

                # Pad if necessary
                if len(sub_seq) < self.global_segment_len:
                    pad_len = self.global_segment_len - len(sub_seq)
                    if seg_start == 0:
                        sub_seq = torch.cat((torch.zeros(pad_len).to(self.device), sub_seq))
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(pad_len).to(self.device)))

                # Normalize
                mean = sub_seq.mean()
                sub_seq = sub_seq - mean

                segments[i, p_idx] = sub_seq

            timestep_ranges.append(sample_ranges)

        return {
            'segments': segments,  # [bs, num_prototypes_global, segment_len]
            'patch_indices': top_k_indices,  # [bs, num_prototypes_global, top_k]
            'timestep_ranges': timestep_ranges,  # List of lists of (start, end) tuples
            'attention_weights': top_k_values.permute(0, 2, 1)  # [bs, num_prototypes_global, top_k]
        }

    def prototype_selection_class(self, output, prototype_vectors):
        """
        output: [bs x num_classes]
        prototype_vectors: [num_prototypes, porto_len, 1] (can be None for global)
        """
        n_selected = self.prototype_num_selected
        indices = torch.argmax(output, dim=1)  # [bs]
        indices = indices.unsqueeze(-1)
        return indices  # [bs x 1]

    def total_variance_loss(self, prototype_vectors):
        """
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        prototype_vectors = prototype_vectors.reshape(prototype_vectors.shape[0], -1)

        mean_embedding = torch.mean(prototype_vectors, dim=0, keepdim=True)
        deviations = prototype_vectors - mean_embedding
        variance = torch.mean(torch.sum(deviations ** 2, dim=1))
        loss = -torch.log(variance)

        return loss

    def dynamic_loss(self, loss1, loss2):
        epsilon = 1e-5

        loss1 = abs(loss1)
        loss2 = abs(loss2)

        w1 = 1.0 / (loss1 + epsilon)
        w2 = 1.0 / (loss2 + epsilon)

        sum_w = w1 + w2
        w1 = w1 / sum_w
        w2 = w2 / sum_w

        normalized_loss1 = w1 * loss1
        normalized_loss2 = -(w2 * loss2)

        total_loss = normalized_loss1 + normalized_loss2

        return total_loss

    def smoothness_loss(self, sequence):
        """
        not in use
        """
        sequence = sequence.reshape(-1)
        diff = sequence[1:] - sequence[:-1]
        return torch.mean(diff ** 2)

    def activation_cosine_loss(self, activations):
        """
        activations: [bs x seq_len x num_prototypes]
        """
        if self.num_prototypes_local == 1:
            return 0
        else:
            X = activations[:, :, 0]
            Y = activations[:, :, 1]
            cosine_sim = F.cosine_similarity(X, Y, dim=1)
            loss = 1 - cosine_sim.mean()
            return loss

    def global_prototype_alignment_loss(self, z_global, global_prototypes, memory_attn, temperature=0.07):
        """
        Alignment loss for global prototypes (InfoNCE-style).
        Encourages each sample's global representation z to be close to its matched global prototype.

        Args:
            z_global: [bs, d_model] - Global representation of each sample
            global_prototypes: [num_prototypes_global, d_model] - Global prototype vectors
            memory_attn: [bs, patch_num, num_prototypes_global] - Attention weights from memory module
            temperature: Temperature parameter for InfoNCE (default: 0.07)

        Returns:
            alignment_loss: Scalar loss value
        """
        bs = z_global.shape[0]
        num_prototypes = global_prototypes.shape[0]

        # Normalize z_global and global_prototypes for cosine similarity
        z_global_norm = F.normalize(z_global, p=2, dim=1)  # [bs, d_model]
        global_prototypes_norm = F.normalize(global_prototypes, p=2, dim=1)  # [num_prototypes_global, d_model]

        # Compute similarity matrix: [bs, num_prototypes_global]
        # Each row represents similarity between one sample and all prototypes
        similarity_matrix = torch.matmul(z_global_norm, global_prototypes_norm.t())  # [bs, num_prototypes_global]

        # Get attention weights aggregated over patches: [bs, num_prototypes_global]
        # Average attention across patch dimension to get sample-level prototype weights
        attn_weights = memory_attn.mean(dim=1)  # [bs, num_prototypes_global]

        # Method 1: Weighted InfoNCE loss
        # Use attention weights to find the most relevant prototype for each sample
        # Positive prototype: the one with highest attention weight
        # Negative prototypes: all others

        # Find positive prototype index for each sample
        pos_prototype_idx = torch.argmax(attn_weights, dim=1)  # [bs]

        # Compute InfoNCE loss
        # For each sample, the positive prototype is the one with highest attention
        # All other prototypes are negatives

        # Scale similarities by temperature
        similarity_matrix_scaled = similarity_matrix / temperature  # [bs, num_prototypes_global]

        # Vectorized InfoNCE loss computation
        # Get positive similarities: [bs]
        pos_sims = similarity_matrix_scaled[torch.arange(bs, device=z_global.device), pos_prototype_idx]

        # Compute exp for all similarities: [bs, num_prototypes_global]
        exp_sims = torch.exp(similarity_matrix_scaled)

        # Sum of exp for all prototypes (including positive): [bs]
        sum_exp_all = exp_sims.sum(dim=1)

        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        # = -pos + log(sum(exp(all)))
        alignment_loss = (-pos_sims + torch.log(sum_exp_all + 1e-8)).mean()

        return alignment_loss

    def global_prototype_alignment_loss_soft(self, z_global, global_prototypes, memory_attn, temperature=0.07):

        bs = z_global.shape[0]


        z_global_norm = F.normalize(z_global, p=2, dim=1)  # [B, D]
        proto_norm = F.normalize(global_prototypes, p=2, dim=1)  # [M, D]

        logits = torch.matmul(z_global_norm, proto_norm.t()) / temperature

        log_p = F.log_softmax(logits, dim=-1)  # [B, M]

        with torch.no_grad():
            q_logits = memory_attn.mean(dim=1)  # [B, M]
            q = F.softmax(q_logits, dim=-1)  # [B, M]

        align_loss = F.kl_div(log_p, q, reduction='batchmean')

        return align_loss

    def global_prototype_alignment_loss_weighted(self, z_global, global_prototypes, memory_attn, temperature=0.07):
        """
        Alternative alignment loss using weighted soft assignment.
        Uses attention weights to softly assign samples to prototypes.

        Args:
            z_global: [bs, d_model] - Global representation of each sample
            global_prototypes: [num_prototypes_global, d_model] - Global prototype vectors
            memory_attn: [bs, patch_num, num_prototypes_global] - Attention weights from memory module
            temperature: Temperature parameter (default: 0.07)

        Returns:
            alignment_loss: Scalar loss value
        """
        bs = z_global.shape[0]

        # Normalize z_global and global_prototypes
        z_global_norm = F.normalize(z_global, p=2, dim=1)  # [bs, d_model]
        global_prototypes_norm = F.normalize(global_prototypes, p=2, dim=1)  # [num_prototypes_global, d_model]

        # Compute similarity matrix: [bs, num_prototypes_global]
        similarity_matrix = torch.matmul(z_global_norm, global_prototypes_norm.t())

        # Get attention weights aggregated over patches: [bs, num_prototypes_global]
        attn_weights = memory_attn.mean(dim=1)  # [bs, num_prototypes_global]

        # Weighted alignment loss: minimize distance to prototypes weighted by attention
        # For each sample, compute weighted distance to all prototypes
        # Loss = -sum(attn_weight * similarity) for each sample
        weighted_similarity = (attn_weights * similarity_matrix).sum(dim=1)  # [bs]
        alignment_loss = -weighted_similarity.mean()  # Negative because we want to maximize similarity

        return alignment_loss

    def global_prototype_diversity_loss(self, global_prototypes, method='variance'):
        """
        Diversity loss for global prototypes to ensure they are not redundant.
        Encourages prototypes to be diverse and cover different regions of the feature space.

        Args:
            global_prototypes: [num_prototypes_global, d_model] - Global prototype vectors
            method: Method to compute diversity loss ('variance' or 'cosine')

        Returns:
            diversity_loss: Scalar loss value (lower is better, so we minimize similarity)
        """
        num_prototypes = global_prototypes.shape[0]

        if num_prototypes <= 1:
            return torch.tensor(0.0, device=global_prototypes.device, requires_grad=True)

        if method == 'cosine':
            # Method 2: Cosine similarity-based diversity
            # Normalize prototypes
            prototypes_norm = F.normalize(global_prototypes, p=2, dim=1)  # [num_prototypes_global, d_model]

            # Compute pairwise cosine similarity matrix
            similarity_matrix = torch.matmul(prototypes_norm, prototypes_norm.t())  # [num_prototypes, num_prototypes]

            # Remove diagonal (self-similarity, which is always 1)
            mask = ~torch.eye(num_prototypes, dtype=torch.bool, device=global_prototypes.device)
            off_diagonal_similarities = similarity_matrix[mask]  # [num_prototypes * (num_prototypes - 1)]

            # Minimize average pairwise similarity (maximize diversity)
            # Use mean similarity as diversity loss (higher similarity = lower diversity)
            diversity_loss = off_diagonal_similarities.mean()

        elif method == 'orthogonal':
            # Method 3: Encourage orthogonality between prototypes
            prototypes_norm = F.normalize(global_prototypes, p=2, dim=1)  # [num_prototypes_global, d_model]

            # Compute pairwise cosine similarity
            similarity_matrix = torch.matmul(prototypes_norm, prototypes_norm.t())  # [num_prototypes, num_prototypes]

            # Remove diagonal
            mask = ~torch.eye(num_prototypes, dtype=torch.bool, device=global_prototypes.device)
            off_diagonal_similarities = similarity_matrix[mask]

            # Penalize high similarity (encourage orthogonality)
            # Use squared similarities to penalize large deviations from orthogonality more
            diversity_loss = (off_diagonal_similarities ** 2).mean()

        else:
            raise ValueError(f"Unknown diversity method: {method}. Choose from 'variance', 'cosine', or 'orthogonal'")

        return diversity_loss

    def global_prototype_diversity_loss_margin(self, global_prototypes, margin=0.2):

        num_prototypes = global_prototypes.shape[0]
        if num_prototypes <= 1:
            return torch.tensor(0.0, device=global_prototypes.device, requires_grad=True)

        prototypes_norm = F.normalize(global_prototypes, p=2, dim=1)  # [M, D]
        sim = prototypes_norm @ prototypes_norm.t()  # [M, M]

        mask = ~torch.eye(num_prototypes, dtype=torch.bool, device=global_prototypes.device)
        off_diag = sim[mask]  # [M*(M-1)]

        loss = F.relu(off_diag - margin).mean()
        return loss

    def local_prototype_diversity_loss(self, prototype_vectors=None, activations=None, method='variance'):
        """
        Comprehensive diversity loss for local prototypes to ensure they are not redundant.
        Can work on prototype vectors directly or on activations.

        Args:
            prototype_vectors: [num_prototypes, patch_num, 1] - Local prototype vectors (optional)
            activations: [bs, seq_len, num_prototypes] - Activation patterns (optional)
            method: Method to compute diversity loss ('variance', 'cosine', 'orthogonal', or 'activation_cosine')

        Returns:
            diversity_loss: Scalar loss value (lower is better, encourages diversity)
        """
        if prototype_vectors is None and activations is None:
            raise ValueError("Either prototype_vectors or activations must be provided")

        if self.num_prototypes_local <= 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if method == 'cosine':
            # Method 2: Cosine similarity-based diversity (works on prototype vectors)
            if prototype_vectors is None:
                raise ValueError("method='cosine' requires prototype_vectors")

            # Reshape prototypes to [num_prototypes, total_dim]
            prototypes_flat = prototype_vectors.reshape(prototype_vectors.shape[0], -1)  # [num_prototypes, total_dim]

            # Normalize prototypes
            prototypes_norm = F.normalize(prototypes_flat, p=2, dim=1)  # [num_prototypes, total_dim]

            # Compute pairwise cosine similarity matrix
            similarity_matrix = torch.matmul(prototypes_norm, prototypes_norm.t())  # [num_prototypes, num_prototypes]

            # Remove diagonal (self-similarity, which is always 1)
            mask = ~torch.eye(self.num_prototypes_local, dtype=torch.bool, device=prototype_vectors.device)
            off_diagonal_similarities = similarity_matrix[mask]  # [num_prototypes * (num_prototypes - 1)]

            # Minimize average pairwise similarity (maximize diversity)
            diversity_loss = off_diagonal_similarities.mean()

        elif method == 'orthogonal':
            # Method 3: Encourage orthogonality between prototypes
            if prototype_vectors is None:
                raise ValueError("method='orthogonal' requires prototype_vectors")

            prototypes_flat = prototype_vectors.reshape(prototype_vectors.shape[0], -1)
            prototypes_norm = F.normalize(prototypes_flat, p=2, dim=1)

            # Compute pairwise cosine similarity
            similarity_matrix = torch.matmul(prototypes_norm, prototypes_norm.t())

            # Remove diagonal
            mask = ~torch.eye(self.num_prototypes_local, dtype=torch.bool, device=prototype_vectors.device)
            off_diagonal_similarities = similarity_matrix[mask]

            # Penalize high similarity (encourage orthogonality)
            # Use squared similarities to penalize large deviations from orthogonality more
            diversity_loss = (off_diagonal_similarities ** 2).mean()

        elif method == 'activation_cosine':
            # Method 4: Diversity based on activation patterns (works on activations)
            if activations is None:
                raise ValueError("method='activation_cosine' requires activations")

            # activations: [bs, seq_len, num_prototypes]
            bs, seq_len, num_proto = activations.shape

            # Average over batch and sequence to get prototype-level representations
            # Get average activation per prototype: [num_prototypes]
            prototype_activations = activations.mean(dim=(0, 1))  # [num_prototypes]

            # Compute variance of activations across prototypes
            # Higher variance means more diverse activation patterns
            variance = torch.var(prototype_activations, dim=0)

            # Maximize variance (minimize negative variance)
            diversity_loss = -variance

        elif method == 'activation_pairwise':
            # Method 5: Pairwise cosine similarity on activation patterns
            if activations is None:
                raise ValueError("method='activation_pairwise' requires activations")

            # activations: [bs, seq_len, num_prototypes]
            # Reshape to [bs * seq_len, num_prototypes] for easier processing
            bs, seq_len, num_proto = activations.shape
            activations_reshaped = activations.permute(2, 0, 1).reshape(num_proto, -1)  # [num_prototypes, bs * seq_len]

            # Normalize each prototype's activation pattern
            activations_norm = F.normalize(activations_reshaped, p=2, dim=1)  # [num_prototypes, bs * seq_len]

            # Compute pairwise cosine similarity matrix
            similarity_matrix = torch.matmul(activations_norm, activations_norm.t())  # [num_prototypes, num_prototypes]

            # Remove diagonal
            mask = ~torch.eye(self.num_prototypes_local, dtype=torch.bool, device=activations.device)
            off_diagonal_similarities = similarity_matrix[mask]

            # Minimize average pairwise similarity
            diversity_loss = off_diagonal_similarities.mean()

        else:
            raise ValueError(
                f"Unknown diversity method: {method}. Choose from 'variance', 'cosine', 'orthogonal', 'activation_cosine', or 'activation_pairwise'")

        return diversity_loss



