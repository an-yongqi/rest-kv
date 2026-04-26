import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import re

from typing import List


from typing import List, Optional, Tuple
from transformers.cache_utils import Cache

def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, keep_idx, 1)
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k), kv_states[:, :, seqlen - recent_size:, :], ~mask

class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            import nvtx
            copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')

    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent


class PyramidKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None, merge = None):

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.steps = -1
        self.beta = beta

        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num


        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num


        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps

        # print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}")
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


def ema(input_tensor, dim, alpha=None):
    if alpha is None:
        alpha = 2. / (input_tensor.size(dim) + 1)
    # Initialize accumulator with the first element along the given dimension
    cumulative_tensor = input_tensor.select(dim, 0).unsqueeze(dim).clone()
    for i in range(1, input_tensor.size(dim)):
        current_slice = input_tensor.select(dim, i).unsqueeze(dim)
        cumulative_tensor = alpha * current_slice + (1 - alpha) * cumulative_tensor

    return cumulative_tensor.squeeze(dim)


class AblationCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, ss=None, ts=None, indicator=None):
        self.max_capacity_prompt = max_capacity_prompt
        self.window_size = window_size
        assert self.max_capacity_prompt - self.window_size > 0
        self.ss = ss
        self.ts = ts
        self.indicator = indicator

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, Wo):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            if self.indicator == 'random':
                # [bsz, num_heads, max_capacity_prompt]
                indices = torch.argsort(torch.rand(bsz, num_heads, q_len), dim=-1)[..., :self.max_capacity_prompt]
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim).to(key_states.device)
                key_states = key_states.gather(dim = 2, index = indices)  # [bsz, num_heads, max_capacity_prompt, head_dim]
                value_states = value_states.gather(dim = 2, index = indices)    # [bsz, num_heads, max_capacity_prompt, head_dim]
                return key_states, value_states
                
            elif self.indicator == 'attn' or self.indicator == 'attn-v':
                attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim) # [bsz, num_heads, window_size, q_len]
                mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)    # [window_size, window_size]
                mask_cond = torch.arange(mask.size(-1), device=attn_weights.device) 
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)    # [window_size, window_size]
                mask = mask.to(attn_weights.device)
                attention_mask = mask[None, None, :, :] # [1, 1, window_size, window_size]
                attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask    # mask the diagonal part
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # [bsz, num_heads, window_size, q_len]
                attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)    # [bsz, num_heads, q_len]
                if self.indicator == 'attn-v':
                    attn_weights_sum *= torch.norm(value_states[..., :-self.window_size, :], p=2, dim=-1) # [bsz, num_heads, q_len]
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=5, padding=5//2, stride=1)  # [bsz, num_heads, q_len]
                indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices  # [bsz, num_heads, max_capacity_prompt - window_size]
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)    # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)  # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)    # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
                k_cur = key_states[:, :, -self.window_size:, :] # [bsz, num_heads, window_size, head_dim]
                v_cur = value_states[:, :, -self.window_size:, :]   # [bsz, num_heads, window_size, head_dim]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)   # [bsz, num_heads, max_capacity_prompt, head_dim]
                value_states = torch.cat([v_past_compress, v_cur], dim = 2) # [bsz, num_heads, max_capacity_prompt, head_dim]
                return key_states, value_states
            elif self.indicator == 'reconstruction':
                attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim) # [bsz, num_heads, window_size, q_len]
                mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)    # [window_size, window_size]
                mask_cond = torch.arange(mask.size(-1), device=attn_weights.device) 
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)    # [window_size, window_size]
                mask = mask.to(attn_weights.device)
                attention_mask = mask[None, None, :, :] # [1, 1, window_size, window_size]
                attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask    # mask the diagonal part
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # [bsz, num_heads, window_size, q_len]
                
                if self.ts == 'ema':
                    attn_weights_mean = ema(attn_weights[:, :, -self.window_size:, : -self.window_size], dim=-2, alpha=0.3)  # [bsz, num_heads, q_len - window_size]
                elif self.ts == 'mean':
                    attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)    # [bsz, num_heads, q_len - window_size]
                elif self.ts == 'inv-ema':
                    attn_weights_mean = ema(attn_weights[:, :, -self.window_size:, : -self.window_size].flip(dims=[-2]), dim=-2, alpha=0.3)
                elif self.ts == 'none':
                    attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size][:, :, -1, :]    # [bsz, num_heads, q_len - window_size]
                
                attn_outputs_mean = torch.einsum("bhl,bhld->bhd", attn_weights_mean, value_states[:, :, :-self.window_size]) # [bsz, num_heads, head_dim]
                important_score = attn_weights_mean / (1 - attn_weights_mean)   # [bsz, num_heads, q_len - window_size]
                Wo = Wo.reshape(-1, num_heads, head_dim)
                projection_outputs = torch.einsum("bhd,chd->bhc", attn_outputs_mean, Wo) # [bsz, num_heads, C_out]
                projection_values = torch.einsum("bhld,chd->bhlc", value_states, Wo) # [bsz, num_heads, q_len, C_out]
                important_score *= torch.pow(torch.norm((projection_outputs.unsqueeze(2) - projection_values[:, :, :-self.window_size]), p=2, dim=-1), 1.0)   # [bsz, num_heads, q_len - window_size]
    
                if self.ss == 'adaptive':
                    t_shift_score = (attn_weights[0,:,:self.window_size//2].sort()[1][:,:,self.window_size-self.max_capacity_prompt:] - attn_weights[0,:,self.window_size//2:].sort()[1][:,:,self.window_size-self.max_capacity_prompt:]).float().mean(dim=(1,2))
                    kernel_sizes = 2 * (t_shift_score.abs() // 2000) + 1 
                    shifts = torch.where(t_shift_score > 0, t_shift_score // 2000, t_shift_score // 2000 + 1)
                    tmp = []
                    for i in range(num_heads):
                        kernel_size = max(5, int(kernel_sizes[i]))
                        shift = int(shifts[i])
                        padding_left = kernel_size//2 - shift
                        padding_right = kernel_size//2 + shift
                        padded = F.pad(important_score[:,i], (padding_left, padding_right), mode='replicate')  # asymmetric padding
                        tmp.append(F.avg_pool1d(padded, kernel_size=kernel_size, stride=1))
                    attn_cache = torch.stack(tmp, dim=1)
                elif self.ss == 'avgpool':
                    attn_cache = F.avg_pool1d(important_score, kernel_size=5, padding=5//2, stride=1)  # [bsz, num_heads, q_len - window_size]
                elif self.ss == 'maxpool':
                    attn_cache = F.max_pool1d(important_score, kernel_size=5, padding=5//2, stride=1)  # [bsz, num_heads, q_len - window_size]
                elif self.ss == 'none':
                    attn_cache = important_score
                
                indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices  # [bsz, num_heads, max_capacity_prompt - window_size]
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)    # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]

                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)  # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)    # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
                k_cur = key_states[:, :, -self.window_size:, :] # [bsz, num_heads, window_size, head_dim]
                v_cur = value_states[:, :, -self.window_size:, :]   # [bsz, num_heads, window_size, head_dim]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)   # [bsz, num_heads, max_capacity_prompt, head_dim]
                value_states = torch.cat([v_past_compress, v_cur], dim = 2) # [bsz, num_heads, max_capacity_prompt, head_dim]
                return key_states, value_states
                

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, recent_size = 32, ratio =  0.4, mode=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio
        self.mode = mode

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.ratio = ratio
        self.recent_size = recent_size

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim) # [bsz, num_heads, window_size, q_len]
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)    # [window_size, window_size]
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device) 
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)    # [window_size, window_size]
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :] # [1, 1, window_size, window_size]
            # Masking before softmax acts as a form of normalization (SnapKV approach)
            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask    # mask the diagonal part

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # [bsz, num_heads, window_size, q_len]
            # [(attn_weights[0][i][:16].sort()[1][:, -96:].sum() / 96 / 16 - attn_weights[0][i][-16:].sort()[1][:, -96:].sum() / 96 / 16).item() for i in range(32)]
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)    # [bsz, num_heads, q_len]
            # attn_weights_sum = ema(attn_weights[:, :, -self.window_size:, : -self.window_size], dim=-2, alpha=0.5)    # [bsz, num_heads, q_len]
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)  # [bsz, num_heads, q_len]
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)  # [bsz, num_heads, q_len]
            else:
                raise ValueError('Pooling method not supported')
            # attn_cache *= torch.norm(value_states[..., :-self.window_size, :], p=2, dim=-1) # [bsz, num_heads, q_len]
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices  # [bsz, num_heads, max_capacity_prompt - window_size]
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)    # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)  # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)    # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
            k_cur = key_states[:, :, -self.window_size:, :] # [bsz, num_heads, window_size, head_dim]
            v_cur = value_states[:, :, -self.window_size:, :]   # [bsz, num_heads, window_size, head_dim]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)   # [bsz, num_heads, max_capacity_prompt, head_dim]
            value_states = torch.cat([v_past_compress, v_cur], dim = 2) # [bsz, num_heads, max_capacity_prompt, head_dim]
            return key_states, value_states

    def update_think(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]
            
            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio)
            return kv_pruned, kv_recent, mask, value_states


class RestKVCluster():
    """KV cache compression using importance-aware selection (ReST-KV).

    Importance score = odds_ratio(attention) * L2_norm(output_diff)
    This captures both "how much attention a key receives" and "how unique
    the corresponding value's contribution is to the output".

    Key configuration knobs:
      metric_mode: "before" computes the importance metric per query then
          aggregates (mean or EMA); "after" aggregates attention first, then
          computes the metric once on the aggregated result.
      use_wo: when True, projects through the output projection W_o before
          computing the value difference, so the metric is measured in
          output space rather than value space.
      use_norm: when True, normalizes the importance score by the norm of
          the projected output, preventing high-norm heads from dominating.
      use_ema: when True, uses exponential moving average (with parameter
          alpha) instead of simple mean when aggregating across queries.
      pooling / kernel_size: smooths importance scores via pooling before
          top-k selection.  'adaptive' adjusts kernel size and asymmetric
          shift per head based on how the attention distribution drifts.

    Overall flow:
      1. Compute attention weights for the recent window of queries.
      2. Derive per-position importance scores (odds-ratio * L2 diff).
      3. Smooth via pooling (avg / max / adaptive).
      4. Select top-k positions from the non-window (past) KV cache.
      5. Concatenate selected past KV with the recent window.
    """

    def __init__(self, num_hidden_layers = 32, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool', beta = 20, num_layers = 80, layer_idx=None, merge=None, use_wo=False, use_norm=False, use_ema=False, alpha=0.5, metric_mode=None, tau=1, scale=200, use_pyramid=False):
        # SnapKV base parameters
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0

        # PyramidKV layer-adaptive budget
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.steps = -1
        self.beta = beta

        # ReST-KV-specific parameters
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.use_wo = use_wo
        self.use_norm = use_norm
        self.use_ema = use_ema
        self.use_pyramid = use_pyramid
        self.alpha = alpha
        self.tau = tau
        self.metric_mode = metric_mode
        self.scale = scale

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, Wo):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")
        # PyramidKV
        if self.use_pyramid:
            min_num = (self.max_capacity_prompt - self.window_size) // self.beta
            max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num


            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num


            steps = (max_num - min_num) // (self.num_hidden_layers - 1)
            max_capacity_prompt = max_num - self.layer_idx * steps

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim) # [bsz, num_heads, window_size, q_len]
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)    # [window_size, window_size]
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device) 
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)    # [window_size, window_size]
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :] # [1, 1, window_size, window_size]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask    # mask the diagonal part
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # [bsz, num_heads, window_size, q_len]
            
            # --- Compute importance score: odds_ratio(attn) * L2_norm(value_diff) ---
            if self.metric_mode == "before":    # per-query metric, then aggregate across queries
                attn_outputs = torch.matmul(attn_weights, value_states) # [bsz, num_heads, window_size, head_dim]
                important_score = attn_weights / (1 - attn_weights) # odds ratio [bsz, num_heads, window_size, q_len]
                if self.use_wo: # project through W_o before computing value difference
                    Wo = Wo.reshape(-1, num_heads, head_dim)
                    projection_outputs = torch.einsum("bhwd,chd->bhwc", attn_outputs, Wo) # [bsz, num_heads, window_size, C_out]
                    projection_values = torch.einsum("bhld,chd->bhlc", value_states, Wo) # [bsz, num_heads, q_len, C_out]
                    chunk_size = 1024 * 4  # chunk size tunable based on available GPU memory
                    norm_result = []
                    for i in range(0, q_len, chunk_size):
                        q_len_chunk = slice(i, min(i + chunk_size, q_len))  # slice along q_len dimension
                        norm_chunk = torch.norm(projection_outputs.unsqueeze(3) - projection_values[:, :, q_len_chunk].unsqueeze(2), p=2, dim=-1)  # L2 norm per chunk
                        norm_result.append(norm_chunk)

                    important_score *= torch.cat(norm_result, dim=-1)
                    if self.use_norm:   # normalize by output projection norm
                        important_score /= torch.norm(projection_outputs, p=2, dim=-1).unsqueeze(-1) # [bsz, num_heads, window_size, q_len]
                else:   # compute value diff directly in value space
                    important_score *= torch.norm((attn_outputs.unsqueeze(3) - value_states.unsqueeze(2)), p=2, dim=-1) # [bsz, num_heads, window_size, q_len]
                    if self.use_norm:   # normalize by attention output norm
                        important_score /= torch.norm(attn_outputs, p=2, dim=-1).unsqueeze(-1) # [bsz, num_heads, window_size, q_len]
                if self.use_ema:    # aggregate across queries via exponential moving average
                    important_score = ema(important_score[:, :, -self.window_size:, : -self.window_size], dim=-2, alpha=self.alpha) # [bsz, num_heads, q_len - window_size]
                else:   # aggregate across queries via simple mean
                    important_score = important_score[:, :, -self.window_size:, : -self.window_size].mean(dim = -2) # [bsz, num_heads, q_len - window_size]
            else:   # "after" mode: aggregate attention first, then compute metric once
                if self.use_ema:
                    attn_weights_mean = ema(attn_weights[:, :, -self.window_size:, : -self.window_size], dim=-2, alpha=self.alpha)  # [bsz, num_heads, q_len - window_size]
                else:
                    attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)    # [bsz, num_heads, q_len - window_size]
                attn_outputs_mean = torch.einsum("bhl,bhld->bhd", attn_weights_mean, value_states[:, :, :-self.window_size]) # [bsz, num_heads, head_dim]
                important_score = attn_weights_mean / (1 - attn_weights_mean)   # odds ratio [bsz, num_heads, q_len - window_size]
                if self.use_wo: # project through W_o before computing value difference
                    Wo = Wo.reshape(-1, num_heads, head_dim)
                    projection_outputs = torch.einsum("bhd,chd->bhc", attn_outputs_mean, Wo) # [bsz, num_heads, C_out]
                    projection_values = torch.einsum("bhld,chd->bhlc", value_states, Wo) # [bsz, num_heads, q_len, C_out]
                    important_score *= torch.pow(torch.norm((projection_outputs.unsqueeze(2) - projection_values[:, :, :-self.window_size]), p=2, dim=-1), self.tau)   # [bsz, num_heads, q_len - window_size]
                else:   # compute value diff directly in value space
                    important_score *= torch.norm((attn_outputs_mean.unsqueeze(2) - value_states[:, :, :-self.window_size]), p=2, dim=-1)   # [bsz, num_heads, q_len - window_size]
   
            # --- Smooth importance scores via pooling ---
            # Adaptive pooling: kernel size and shift vary per head based on
            # how much the attention distribution shifts between the first and
            # second halves of the window (larger shift -> larger kernel).
            if self.pooling == 'adaptive':
                t_shift_score = (attn_weights[0,:,:self.window_size//2].sort()[1][:,:,self.window_size-self.max_capacity_prompt:] - attn_weights[0,:,self.window_size//2:].sort()[1][:,:,self.window_size-self.max_capacity_prompt:]).float().mean(dim=(1,2))
                kernel_sizes = 2 * (t_shift_score.abs() // self.scale) + 1 
                shifts = torch.where(t_shift_score > 0, t_shift_score // self.scale, t_shift_score // self.scale + 1)
                tmp = []
                for i in range(num_heads):
                    kernel_size = max(self.kernel_size, int(kernel_sizes[i]))
                    shift = int(shifts[i])
                    padding_left = kernel_size//2 - shift
                    padding_right = kernel_size//2 + shift
                    padded = F.pad(important_score[:,i], (padding_left, padding_right), mode='replicate')  # asymmetric padding
                    tmp.append(F.avg_pool1d(padded, kernel_size=kernel_size, stride=1))
                attn_cache = torch.stack(tmp, dim=1)
            elif self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(important_score, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)  # [bsz, num_heads, q_len - window_size]
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(important_score, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)  # [bsz, num_heads, q_len - window_size]
            else:
                attn_cache = important_score
            
            # --- Top-k selection on smoothed scores, then concat with recent window ---
            if self.use_pyramid and q_len >= self.max_capacity_prompt:
                indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            else:
                indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices  # [bsz, num_heads, max_capacity_prompt - window_size]
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)    # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)  # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)    # [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
            k_cur = key_states[:, :, -self.window_size:, :] # [bsz, num_heads, window_size, head_dim]
            v_cur = value_states[:, :, -self.window_size:, :]   # [bsz, num_heads, window_size, head_dim]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)   # [bsz, num_heads, max_capacity_prompt, head_dim]
            value_states = torch.cat([v_past_compress, v_cur], dim = 2) # [bsz, num_heads, max_capacity_prompt, head_dim]
            return key_states, value_states


class L2NormCluster():
    def __init__(self, max_capacity_prompt:int=256+64, layer_idx:int=0, skip_layers: List[int] = []):
        self.max_capacity_prompt = max_capacity_prompt
        self.layer_idx = layer_idx
        self.skip_layers = skip_layers

    def reset(self, max_capacity_prompt:int=256+64, layer_idx:int=0, skip_layers: List[int] = []):
        self.max_capacity_prompt = max_capacity_prompt
        self.layer_idx = layer_idx
        self.skip_layers = skip_layers

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f"L2Norm max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif self.layer_idx in self.skip_layers:
            return key_states, value_states
        else:
            head_dim = key_states.size(-1)
            token_norms = torch.norm(key_states, p=2, dim=-1)
            sorted_indices = token_norms.squeeze(-1).argsort(dim=-1)
            sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            sorted_key_states = key_states.gather(dim=2, index=sorted_indices_expanded)
            sorted_value_states = value_states.gather(dim=2, index=sorted_indices_expanded)

            key_states = sorted_key_states[:, :, :self.max_capacity_prompt, :]
            value_states = sorted_value_states[:, :, :self.max_capacity_prompt, :]

            return key_states, value_states

class CAMKVCluster:
    def __init__(self, start_budget_ratio = 0.1, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.start_budget_ratio = start_budget_ratio
        self.merge = merge

    def reset(self, start_budget_ratio = 0.1, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.start_budget_ratio = start_budget_ratio
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f"CAM max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum

            # merge recent tokens
            start_budget = math.ceil(self.start_budget_ratio * q_len)
            recent_budget = self.window_size
            # start_budget = math.ceil(self.start_budget_ratio * attn_weights.shape[-1])
            # recent_budget = math.ceil(self.recent_budget_ratio * attn_weights.shape[-1])
            # print(f"start_budget {start_budget}")
            # print(f"recent_budget {recent_budget}")

            # CAM merge
            seq_length = attn_weights.shape[-1]
            padding_length = 0
            merge_budget = recent_budget
            for token_index in range(start_budget + padding_length + recent_budget, seq_length):
                if token_index - recent_budget < 0 or token_index - recent_budget >= value_states.shape[2]:
                    continue
                attn_score = torch.mean(attn_weights[:, :, :token_index, :token_index], dim=-2)
                mean_attn = torch.max(torch.cat((attn_score[0, :, :start_budget], attn_score[0, :, token_index - recent_budget:token_index]), dim=-1), dim=-1)[0]
                merge_prob = attn_score[0, :, token_index - recent_budget] / mean_attn
                if torch.isnan(merge_prob).any(): merge_prob[torch.isnan(merge_prob)] = 0
                if torch.isinf(merge_prob).any(): merge_prob[torch.isinf(merge_prob)] = 1
                merge_mask = torch.bernoulli(merge_prob.clamp(min=0, max=1))
                score1 = value_states[:, :, token_index - recent_budget, ...].clone() * merge_mask.unsqueeze(-1) / merge_budget
                value_states[:, :, token_index - recent_budget + 1:token_index - recent_budget + merge_budget + 1, :] += score1.unsqueeze(2)

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            return key_states, value_states


class H2OKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # print(f"H2O max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2) # [bsz, num_heads, q_len-window_size]

            attn_cache = attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


class StreamingLLMKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:

            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class AdaKVCluster():
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',max_capacity_prompt=None,floor = None,normalize=None, layer_idx = None, num_hidden_layers=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = max_capacity_prompt - window_size
        self.floor_ratio = floor
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.num_hidden_layers = num_hidden_layers

        self.normalize = normalize
        self.layer_idx = layer_idx

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None


    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        adaptive_attn_score = sorted_attn_score
        length = adaptive_attn_score.size(dim=-1)
        if self.normalize:
            ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
            adaptive_attn_score = adaptive_attn_score*ratio_weight
        adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads)
        sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity,dim=-1).indices
        sorted_indices = sorted_indices//length
        # floor capacity set
        head_adaptive_capacity = torch.zeros((bsz,num_heads),device=_device,dtype = sorted_indices.dtype)
        head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
        assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity
        head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads):
            cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states,heads_value_states


class HeadKVCluster():
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',max_capacity_prompt=None, layer_idx = None, num_hidden_layers=None, head_capacity=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = max_capacity_prompt - window_size
        self.head_adaptive_capacity = head_capacity
        self.num_hidden_layers = num_hidden_layers

        self.layer_idx = layer_idx

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        _,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads):
            cache_index = sorted_attn_score_indices[head_idx][...,:self.head_adaptive_capacity[self.layer_idx][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states,heads_value_states

def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None


    self.kv_cluster = PyramidKVCluster(
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None


    self.kv_cluster = SnapKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )
    
def init_restkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'merge'):
            self.config.merge = None


    self.kv_cluster = RestKVCluster(
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        use_wo = self.config.use_wo,
        use_norm = self.config.use_norm,
        use_ema = self.config.use_ema,
        use_pyramid = self.config.use_pyramid,
        alpha = self.config.alpha,
        metric_mode = self.config.metric_mode,
        tau = self.config.tau,
        scale = self.config.scale,
    )


def init_ablation(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096


    self.kv_cluster = AblationCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        ss = self.config.ss,
        ts = self.config.ts,
        indicator = self.config.indicator,
    )

def init_think(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
        if not hasattr(self.config, 'recent_size'):
            self.config.recent_size = 32
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = 0.4


    self.kv_cluster = SnapKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        recent_size = self.config.recent_size,
        ratio = self.config.ratio
        )

def init_l2norm(self):

    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'layer_idx'):
            self.config.layer_idx = 0
        if not hasattr(self.config, 'skip_layers'):
            self.config.skip_layers = [0,1]

    self.kv_cluster = L2NormCluster(
        max_capacity_prompt = self.config.max_capacity_prompt,
        layer_idx = self.layer_idx,
        skip_layers = self.config.skip_layers
    )

def init_CAM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'


    self.kv_cluster = CAMKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_H2O(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = H2OKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None


    self.kv_cluster = StreamingLLMKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_adakv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'floor_ratio'):
            self.config.floor_ratio = 0.2
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = True
    # max_capacity_prompt --> base_capacity
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaKVCluster(
            num_hidden_layers = self.config.num_hidden_layers,
            layer_idx = self.layer_idx,
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor = self.config.floor,
            normalize = self.config.normalize
            )


def init_headkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'head_capacity'):
            raise ValueError("Must have head_capacity")
    # max_capacity_prompt --> base_capacity
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = HeadKVCluster(
            num_hidden_layers = self.config.num_hidden_layers,
            layer_idx = self.layer_idx,
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            head_capacity=self.config.head_capacity
            )
