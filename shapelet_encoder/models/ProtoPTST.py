import torch
from torch import nn
import sys

from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import PatchEmbedding
from .models import PatchTST
from torch.nn import functional as F
import numpy

# 在原有的基础上加入了CNN

# class GradientReversal(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, lambd=1.0):
#         ctx.lambd = lambd
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return -ctx.lambd * grad_output, None
#
#
# def GRL(x, lambd=1.0):
#     return GradientReversal.apply(x, lambd)


# # correlation penalty helper
# def corr_squared_loss(activations, confounder):
#     # activations: [bs, seq_len, K]
#     # confounder: [bs] or [bs,1]
#     agg = activations.max(dim=1).values  # [bs, K]
#     c = confounder.view(-1, 1).float()
#     agg_center = agg - agg.mean(dim=0, keepdim=True)
#     c_center = c - c.mean()
#     cov = (agg_center * c_center).mean(dim=0)  # [K]
#     std_a = agg_center.std(dim=0) + 1e-8
#     std_c = c_center.std() + 1e-8
#     corr = cov / (std_a * std_c)
#     return (corr ** 2).mean()


class Model(PatchTST.Model):
    
    
    def __init__(self, configs):
        super().__init__(configs)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding = configs.stride
        self.patch_num = (self.seq_len - self.patch_len + self.padding) // self.stride+1
        self.d_model = configs.d_model
        self.device= configs.device
        self.prototype_num_selected = 1
        
        self.num_prototypes = configs.num_prototypes
        self.num_classes = configs.num_class
        self.prototype_len = configs.prototype_len
        self.prototype_shape = [configs.num_prototypes,self.prototype_len,configs.enc_in] # [num_prototypes x d_model x enc_in]
        self.prototype_activation_function ='log'
        
        
        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape), requires_grad=True) # trainable parameter
        #self.prototype_vectors = nn.Parameter(torch.zeros(self.prototype_shape), requires_grad=True)
        
        if configs.prototype_init == 'kaiming':
            self.prototype_vectors = nn.init.kaiming_normal_(self.prototype_vectors)
        elif configs.prototype_init == 'xavier':
            self.prototype_vectors = nn.init.xavier_normal_(self.prototype_vectors)
        else:
            
            print("random init prototype")
            
        
        # Xavier initialization
        print("self.seq_len",self.seq_len)
        
        # Compute expected input dimension
        expected_input_dim = (self.seq_len-1) * self.num_prototypes
        
        # Add adaptive pooling to handle dynamic-length input
        self.adaptive_pool = nn.AdaptiveAvgPool1d(expected_input_dim // self.num_prototypes)
        
        self.projection = nn.Linear(
                expected_input_dim, configs.num_class  #### !!!
            )
        
        self.projection_1= nn.Linear(
                 self.num_prototypes*101, configs.num_class)
        
        self.linear_1 = nn.Linear(configs.d_model, 1)

        # CNN branch for prototypes to capture local temporal patterns
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

        # bookkeeping for class templates
        self.current_class_templates = {i: torch.zeros(self.seq_len).to(self.device) for i in range(self.num_classes)}
        
        print("!!!!prototype_shape",self.prototype_shape)
        print("!!!!self.patch_num",self.patch_num)
        print("configs.d_model",configs.d_model)
        print("!!!!expected_input_dim", expected_input_dim)
        
    def prototype_layer(self,prototype, prototype_patch=True, if_normalize=False):
        """
        prototyes: [num_prototypes x prototype_len x enc_in]: torch.tensor
        """
        # Normalization from Non-stationary Transformer
        if if_normalize:
            means = prototype.mean(dim=1, keepdim=True).detach()
            prototype = prototype - means
            stdev = prototype.std(dim=1, keepdim=True, unbiased=False) + 1e-5
            prototype = prototype / stdev 
            
        
        prototype_enc=prototype # [num_prototypes, prototype_len, enc_in]
        ##print("!!1!enc_out.shape",prototype_enc.shape)#

        if prototype_patch:
            # do patching and embedding
            prototype_enc = prototype_enc.permute(0, 2, 1) # [num_prototypes,enc_in, prototype_len]

            # 加入的CNN部分，卷积在 prototype_len 维度上进行，用于提取每个 prototype 在时间维度上的局部特征
            if hasattr(self, 'prototype_local_cnn') and self.prototype_local_cnn is not None:
                local_features = self.prototype_local_cnn(prototype_enc)
                if self.prototype_cnn_dropout is not None:
                    local_features = self.prototype_cnn_dropout(local_features)
                prototype_enc = prototype_enc + local_features

            ##print("!!1.5!enc_out.shape",prototype_enc.shape)
            prototype_out, n_vars = self.patch_embedding(prototype_enc)# [num_prototypes, patch_num, d_model]
            
            ##print("!!2!enc_out.shape",prototype_out.shape)

            #  Encoder
            # z: [bs * nvars x patch_num x d_model]
            prototype_out, attns = self.encoder(prototype_out) # [num_prototypes, patch_num, d_model]
            ##print("!!3!prototype.shape",prototype_out.shape)
            prototype_out=self.linear_1(prototype_out)  # [num_prototypes, patch_num, 1]
    
        
            #prototype=self.prototype_vectors # !!! test
            
            ##print("!!4!prototype.shape",prototype_out.shape)
        
        return prototype_out
        
#
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        input:
        x_enc: [bs x seq_len x nvars ]

        output: [bs x num_classes]
        """
        # x_enc=x_enc+2.5### ！ for freaqshape
        x=x_enc # ([4, 1400, 1])
        #print("x.shape",x.shape)


        # z: [bs x nvars x patch_num x d_model]

        # prototype_vectors [num_prototypes, prototype_len, enc_in变量现在应该都是1]

        prototype=self.prototype_layer(self.prototype_vectors) # [num_prototypes, patch_num, 1]
        if self.args.ablation == 'no_prototype_layer':
            prototype = self.prototype_vectors

        # prototype layer distance calculation
        activations=self.convolution_distance_x_as_input(x,prototype) # [bs x seq_len x num_prototypes]
        #activations=self.convolution_distance(prototype_out,prototype)

        # Reshape to [batch_size, num_prototypes, seq_len] for 1D adaptive pooling
        activations_for_pool = activations.permute(0, 2, 1)  # [bs x num_prototypes x seq_len]

        # Use adaptive pooling to standardize sequence length
        expected_seq_len = (self.seq_len-1)

        # Apply adaptive pooling
        # 变成seq_len-1的长度
        pooled_activations = self.adaptive_pool(activations_for_pool)  # [bs x num_prototypes x expected_seq_len]

        # Reshape to a flat vector
        output = pooled_activations.reshape(pooled_activations.shape[0], -1)  # [bs x (num_prototypes * expected_seq_len)] 16, 1399*2

        if False:
            indices=self.prototype_selection(activations,prototype) #[bs x n_selected]
            indices= indices.unsqueeze(-1).expand(-1, -1, self.seq_len) # #[bs x n_selected,seq_len]
            indices=indices.permute(0,2,1) # [bs x seq_len x n_selected]
            output= torch.gather(activations, dim=2, index=indices).reshape(activations.shape[0],-1)


        # Dimension adaptation (fallback)
        expected_dim = self.projection.weight.shape[1]
        current_dim = output.shape[1]

        if current_dim != expected_dim:
            print(f"Warning: Dimension mismatch even after pooling. Adapting input from {current_dim} to {expected_dim}")

            if current_dim > expected_dim:
                # Truncate if input dimension is too large
                output = output[:, :expected_dim]
            else:
                # Pad with zeros if input dimension is too small
                padding = torch.zeros(output.shape[0], expected_dim - current_dim, device=output.device)
                output = torch.cat([output, padding], dim=1)

        output = self.projection(output)

        #output=self.projection_1(x.reshape(x.shape[0],-1)) # test

        # activation是相似度， prototype是原型， output是分类结果（exp中用了softmax）
        # 多个 prototype 可以学习不同的模式，而单个长 prototype 只能学习一种模式，但时间跨度更长
        return output, activations, prototype


    def convolution_distance_x_as_input(self, x, prototype_vectors):
        """
        x: [bs x patch_num x nvars]
        prototype_vectors: [num_prototypes, porto_len, 1]
        """
        # 计算输入时间序列与原型（prototypes）的相似度/匹配度，输出激活值（activations）

        x=x.permute(0,2,1) # [bs, nvars, patch_num  ]但是感觉没有patch  16,1,800
        
  
        prototype_vectors = prototype_vectors.permute(0, 2, 1)  # [num_prototypes, 1 , pro_patch_num] 2,1,98
        prototype_vectors=prototype_vectors.to(x.device)
        # Perform convolution
        ##print("In convolution_distance, x.shape",x.shape,"prototype_vectors.shape",prototype_vectors.shape) # devie is x.device

        # 滑动窗口匹配, prototype_vectors作为卷积核, 在每个位置计算内积（互相关），得到相似度
        # 在序列上滑动一个与 prototype 等长的窗口，逐位置计算匹配度。每prototype个计算一次，padding保证补齐
        # 每个prototype应该都有一个
        conv_out = F.conv1d(x, prototype_vectors, padding=(prototype_vectors.shape[-1]-1)//2 ,stride=1)  # 输出 [bs, num_prototypes, patch_num] 16,2,799
        
        # Adjust output shape and apply normalization
        conv_out = conv_out.permute(0, 2, 1)  # [bs, patch_num, num_prototypes]

        # 这边好奇怪！！！使用定义的self.activation_norm = nn.LayerNorm(self.num_prototypes)反而结果差
        layer_norm = nn.LayerNorm(conv_out.shape[1:]).to(self.device)  # LayerNorm based on conv_out shape
        conv_out = layer_norm(conv_out)
        # conv_out = torch.sigmoid(conv_out)
        conv_out=F.leaky_relu(conv_out, negative_slope=0.01) ## !!!
        # conv_out=F.leaky_relu(conv_out, negative_slope=0.0001) ## !!!
        # softmax across prototype dim to match paper's Eq.(2) if desired
        # conv_out = F.softmax(conv_out, dim=-1) 不加反而大
        ##print("conv_out.shape",conv_out.shape) # [bs x patch_num x num_prototypes]
        return conv_out

  
    def smoothness_loss(self,sequence):
        """
        not in use
        """
        
        sequence=sequence.reshape(-1)
        # sequence shape: [batch_size, sequence_length]
        diff = sequence[:1:] - sequence[:-1]  # Compute differences between adjacent elements
        return torch.mean(diff ** 2)  # Square differences and take the mean
    
    
    def seg_prototype_loss(self,activations,x,prototype_vectors,outputs,select_prototype=True,no_variance_loss=False):
        
        """
        activatetion:  [bs x seq_len x num_prototypes]
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        if self.args.ablation == 'no_variances_loss':
            no_variance_loss = True
        
        
        batch_size = activations.shape[0]
        prototype_len = prototype_vectors.shape[1]
        prototype_num = self.num_prototypes
       
        half_len = prototype_len // 2
        x=x.reshape(batch_size,-1)
        
        subsequence_all_prototype = torch.tensor([]).to(self.device)
        for p_index in range(prototype_num):
            ##print("p_index",p_index)
            # Use torch.topk to get top-1 max value and its index per sample
            activation = activations[:,:,p_index].reshape(batch_size, -1)
            
            #get the top values and indices
            ##print("activations.shape",activation.shape)
            top_k_values, top_k_indices = torch.topk(activation, 1, dim=1)  # [bs x 1]
            # Placeholder for per-sample subsequences (zeros)


            # Initialize a tensor to store extracted subsequences
            subsequences = torch.zeros(batch_size,prototype_len).to(self.device)
            

            # Extract subsequence around the max index for each sample
            for i in range(batch_size):
                center = top_k_indices[i, 0].item()  # center position
                # Compute subsequence start/end within bounds
                start = max(0, center - half_len)
                end = min(self.seq_len, center + half_len)
                
                # Pad to prototype_len at sequence boundaries if needed
                sub_seq = x[i, start:end]
                
                if len(sub_seq) < prototype_len:
                    # Pad at the beginning if start == 0
                    if start == 0:
                        sub_seq = torch.cat((torch.zeros(prototype_len - len(sub_seq)).to(self.device), sub_seq))
                    # Otherwise pad at the end
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(prototype_len - len(sub_seq)).to(self.device)))
                
                # Store into subsequences
                mean = sub_seq.mean()
                # Normalize to zero-mean
                sub_seq = sub_seq - mean
                
                subsequences[i] = sub_seq
            
            subsequences=subsequences.unsqueeze(0) # [1 x bs x prototype_len]
            
            subsequence_all_prototype=torch.cat((subsequence_all_prototype,subsequences),dim=0) # [num_prototypes x bs x prototype_len]
            
        
        subsequence_all_prototype=subsequence_all_prototype.permute(1,0,2) # [ bs x num_prototypes x prototype_len]
        
        prototype_repeat=prototype_vectors.permute(2,0,1) #  [1,num_prototypes, patch_num ]
        prototype_repeat=prototype_repeat.repeat(batch_size, 1,1)
        ##print("subsequence_all_prototype.shape",subsequence_all_prototype.shape,"prototype_vectors.shape",prototype_vectors.shape)
        
        
        if select_prototype and self.num_prototypes>1:
            
            #indices=self.prototype_selection(activations,prototype_vectors) #[bs x n_selected]
            
            indices=self.prototype_selection_class(outputs,prototype_vectors) #[bs x 1]
            #print("############indices",indices)
            
            indices= indices.unsqueeze(-1).expand(-1, -1, prototype_len) #
            
            #print("indices.shape",indices[:,0,0])
            
            
       
            
            
            subsequence_all_prototype= torch.gather(subsequence_all_prototype, dim=1, index=indices)
            prototype_repeat= torch.gather(prototype_repeat, dim=1, index=indices)
            
            #print("subsequence_all_prototype.shape",subsequence_all_prototype.shape)
            
        
        #print("subsequence_all_prototype.shape",prototype_repeat)
        
        prototype_loss = torch.nn.functional.mse_loss(subsequence_all_prototype,prototype_repeat)
        #print("@@@prototype_loss",prototype_loss)
        variance_loss = self.total_variance_loss(prototype_vectors)
        
        
        if  self.args.ablation == 'no_matching_loss':
            prototype_loss = 0
        
        
        #print("proto_loss:",prototype_loss,"variance_loss:",variance_loss)
        if self.num_prototypes==1:
            return  prototype_loss
        
        if no_variance_loss:
            return prototype_loss
        else:
            return self.dynamic_loss(prototype_loss,variance_loss)

  
    def seg_activations(self,activations,prototype_vectors):
        
        """
        activatetion:  [bs x seq_len x num_prototypes]
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        
        batch_size = activations.shape[0]
        prototype_len = prototype_vectors.shape[1]
        prototype_num = self.num_prototypes
       
        half_len = prototype_len // 2
        #x=x.reshape(batch_size,-1)
        
        subsequence_all_prototype = torch.tensor([]).to(self.device)
        for p_index in range(prototype_num):
            ##print("p_index",p_index)
            # Use torch.topk to get top-1 max value and its index per sample
            activation = activations[:,:,p_index].reshape(batch_size, -1)
            
            #get the top values and indices
            ##print("activations.shape",activation.shape)
            top_k_values, top_k_indices = torch.topk(activation, 1, dim=1)  # [bs x 1]
            # Placeholder for per-sample subsequences (zeros)


            # Initialize a tensor to store extracted subsequences
            subsequences = torch.zeros(batch_size,prototype_len).to(self.device)
            

            # Extract subsequence around the max index for each sample
            for i in range(batch_size):
                center = top_k_indices[i, 0].item()  # center position
                # Compute subsequence start/end within bounds
                start = max(0, center - half_len)
                end = min(self.seq_len, center + half_len)
                
                # Pad to prototype_len at sequence boundaries if needed
                sub_seq =  activations[i,start:end,p_index]  #x[i, start:end] !!!
                
                if len(sub_seq) < prototype_len:
                    # Pad at the beginning if start == 0
                    if start == 0:
                        sub_seq = torch.cat((torch.zeros(prototype_len - len(sub_seq)).to(self.device), sub_seq))
                    # Otherwise pad at the end
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(prototype_len - len(sub_seq)).to(self.device)))
                
                # Store into subsequences
                mean = sub_seq.mean()
                
                # Normalize to zero-mean
                sub_seq = sub_seq - mean
                
                subsequences[i] = sub_seq
            
            subsequences=subsequences.unsqueeze(0) # [1 x bs x prototype_len]
            
            subsequence_all_prototype=torch.cat((subsequence_all_prototype,subsequences),dim=0) # [num_prototypes x bs x prototype_len]
            
        
        subsequence_all_prototype=subsequence_all_prototype.permute(1,2,0) # [ bs x  prototype_len x num_prototypes]
        
       
        ##print("subsequence_all_prototype.shape",subsequence_all_prototype.shape,"prototype_vectors.shape",prototype_vectors.shape)
        
        return subsequence_all_prototype

    def prototype_selection(self,conv_out,prototype_vectors):
        """
        conv_out: [bs x patch_num x num_prototypes]
        prototype_vectors: [num_prototypes, porto_len, 1]
        """
        # Select top-n prototypes
        n_selected = self.prototype_num_selected
        
        top_k_values, _ = torch.topk(conv_out, 1, dim=1) # [bs x 1 x num_prototypes]
        
        top_k_values, indices = torch.topk(top_k_values, n_selected, dim=2) # [bs x 1 x n_selected]
        
        indices = indices.squeeze(1) # [bs x n_selected]
        
        #selected_prototypes = torch.gather(conv_out, 1, indices)

        # selected_prototypes has shape [batch_size, n_selected, proto_len]
        
        #print('indices:',indices)
        return indices #[bs x n_selected]
    
    
    def prototype_selection_class(self,output,prototype_vectors):
        """
        output: [bs x num_classes]
        prototype_vectors: [num_prototypes, porto_len, 1]

        """

        n_selected = self.prototype_num_selected

        indices = torch.argmax(output, dim=1) % self.num_prototypes  # [bs] — clamp to valid prototype range

        indices = indices.unsqueeze(-1)

        return indices #[bs x 1]
        




    def total_variance_loss(self,prototype_vectors):
        """
            prototype_vectors: [num_prototypes, patch_num, 1]
            """
        prototype_vectors=prototype_vectors.reshape(prototype_vectors.shape[0],-1)
        
        
        
        mean_embedding = torch.mean(prototype_vectors, dim=0, keepdim=True)
        deviations = prototype_vectors - mean_embedding
        variance = torch.mean(torch.sum(deviations ** 2, dim=1))
        # Maximize variance (i.e., minimize negative variance)
        loss = -torch.log(variance)
        
        return loss
    
    def dynamic_loss(self,loss1,loss2):
        
        epsilon = 1e-5
        
        loss1=abs(loss1)
        loss2=abs(loss2)

        # Compute weights using inverse of loss values
        w1 = 1.0 / (loss1 + epsilon)
        w2 = 1.0 / (loss2 + epsilon)

        # Normalize weights
        sum_w = w1 + w2
        w1 = w1 / sum_w
        w2 = w2 / sum_w

        # Compute total loss
        
        normalized_loss1 = w1 * loss1
        normalized_loss2 = -(w2 * loss2)
        
        #print("normalized_loss1",normalized_loss1,"normalized_loss2",normalized_loss2)
        total_loss = normalized_loss1 + normalized_loss2
        
        return total_loss
    
    def activation_cosine_loss(self,activations):
        """
        activations: [bs x seq_len x num_prototypes]
        """
        # If only one prototype, return 0 loss
        if self.num_prototypes==1:
            return 0
        
        else:
        
            X=activations[:,:,0]
            Y=activations[:,:,1]
            
            cosine_sim = F.cosine_similarity(X, Y, dim=1)
            
            #print("mean",mean,"variance",variance)
            # Compute loss on activations
            loss = 1 - cosine_sim.mean()
            return loss
    
    
    
    def interpolate_tensor(input_tensor,new_T=480,dim_to_change=0):

        # Swap dims to move T to last, easier for interpolate
        last_dim=len(input_tensor.shape)-1
        input_tensor = input_tensor.transpose(dim_to_change,last_dim)  # Now (batch_size, D, T)

        # Use interpolate
        output_tensor = torch.nn.functional.interpolate(input_tensor, size=new_T, mode='linear', align_corners=True)
        print(output_tensor.shape)

        # Swap dims back to original order
        output_tensor = output_tensor.transpose(dim_to_change,last_dim)  # Now (batch_size, new_T, D)

        return output_tensor

# newly add

# compute class templates across dataset (fast form: pass lists or tensors)
#     def compute_class_templates(self, Xm_list, labels_list):
#         # Xm_list: iterable of tensors [seq_len]
#         class_accum = {i: [] for i in range(self.num_classes)}
#         for xm, y in zip(Xm_list, labels_list):
#             class_accum[int(y)].append(xm.detach().cpu())
#         c_list = {}
#         for cls in range(self.num_classes):
#             if len(class_accum[cls]) == 0:
#                 c_list[cls] = torch.zeros(self.seq_len).to(self.device)
#             else:
#                 avg = torch.stack(class_accum[cls], dim=0).mean(dim=0).to(self.device)
#                 c_list[cls] = avg
#         self.current_class_templates = c_list
#         return c_list
#
#     # convenience wrapper for online/EMA update of templates
#     def update_class_templates_ema(self, Xm_batch, labels_batch, momentum=0.9):
#         # Xm_batch: [bs, seq_len]
#         for i in range(Xm_batch.shape[0]):
#             cls = int(labels_batch[i].item())
#             new = Xm_batch[i].detach().cpu().to(self.device)
#             old = self.current_class_templates.get(cls, torch.zeros_like(new))
#             self.current_class_templates[cls] = momentum * old + (1.0 - momentum) * new
#         return self.current_class_templates