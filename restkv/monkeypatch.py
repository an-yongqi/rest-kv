from importlib.metadata import version
import transformers

from restkv.llama_model import llama_flash_attn2_forward_HeadKV, llama_flash_attn2_forward_AdaKV, llama_flash_attn2_forward_PyramidKV,llama_flash_attn2_forward_CAM,llama_flash_attn2_forward_H2O,llama_flash_attn2_forward_SnapKV,llama_flash_attn2_forward_StreamingLLM, llama_flash_attn2_forward_L2Norm, llama_flash_attn2_forward_RestKV, llama_flash_attn2_forward_Ablation
from restkv.llama_model import llama_attn_forward_PyramidKV,llama_attn_forward_CAM,llama_attn_forward_H2O,llama_attn_forward_SnapKV,llama_attn_forward_StreamingLLM, llama_attn_forward_L2Norm, llama_attn_forward_RestKV
from restkv.llama_model import llama_sdpa_attn_forward_PyramidKV,llama_sdpa_attn_forward_CAM,llama_sdpa_attn_forward_H2O,llama_sdpa_attn_forward_SnapKV,llama_sdpa_attn_forward_StreamingLLM, llama_sdpa_attn_forward_L2Norm, llama_sdpa_attn_forward_RestKV
from restkv.llama_model import adaptive_LlamaModel_forward
from restkv.llama_model_think import llama_attn_forward_SnapKV_ThinK, think_model_forward

from restkv.mistral_model import mistral_flash_attn2_forward_HeadKV, mistral_flash_attn2_forward_PyramidKV,mistral_flash_attn2_forward_CAM,mistral_flash_attn2_forward_H2O,mistral_flash_attn2_forward_SnapKV,mistral_flash_attn2_forward_StreamingLLM, mistral_flash_attn2_forward_L2Norm, mistral_flash_attn2_forward_RestKV, mistral_flash_attn2_forward_Ablation
from restkv.mistral_model import mistral_attn_forward_PyramidKV,mistral_attn_forward_CAM,mistral_attn_forward_H2O,mistral_attn_forward_SnapKV,mistral_attn_forward_StreamingLLM, mistral_attn_forward_L2Norm, mistral_attn_forward_RestKV
from restkv.mistral_model import mistral_sdpa_attn_forward_PyramidKV,mistral_sdpa_attn_forward_CAM,mistral_sdpa_attn_forward_H2O,mistral_sdpa_attn_forward_SnapKV,mistral_sdpa_attn_forward_StreamingLLM, mistral_sdpa_attn_forward_L2Norm,  mistral_sdpa_attn_forward_RestKV
from restkv.mistral_model import adaptive_MistralModel_forward

from restkv.qwen_model import qwen_flash_attn2_forward_StreamingLLM, qwen_flash_attn2_forward_SnapKV, qwen_flash_attn2_forward_RestKV
from restkv.gemma_model import gemma_flash_attn2_forward_StreamingLLM, gemma_flash_attn2_forward_SnapKV, gemma_flash_attn2_forward_RestKV

from restkv.llama_model import prepare_inputs_for_generation_llama, prepare_inputs_for_generation_llama_new
from restkv.mistral_model import prepare_inputs_for_generation_mistral, prepare_inputs_for_generation_mistral_new
from restkv.qwen_model import prepare_inputs_for_generation_qwen_new
from restkv.gemma_model import prepare_inputs_for_generation_gemma_new

def replace_llama(method, model_name=None):

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_PyramidKV

    elif method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM

    elif method == "h2o":
        print("Using H2O!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_H2O
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_H2O

    elif method == "cam":
        print("Using CAM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_CAM
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_CAM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_CAM

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SnapKV

    elif method == "restkv":
        print("Using ReST-KV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_RestKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_RestKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_RestKV

    elif method == "ablation":
        print("Using Ablation!")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_Ablation
    
    elif method == "minference":
        print("Using MInference!")
        from .minference import minference_attn_forward, init_minference
        init_minference(model_name)
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new
        transformers.models.llama.modeling_llama.LlamaAttention.forward = minference_attn_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = minference_attn_forward
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = minference_attn_forward

    elif method == "l2norm":
        print("Using L2Norm!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_L2Norm
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_L2Norm
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_L2Norm

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_AdaKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_AdaKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_AdaKV

    elif method == "headkv":
        print("Using HeadKV!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_HeadKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_HeadKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_HeadKV

    elif method == "think":
        print("Using Think!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = think_model_forward
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV_ThinK


    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new


def replace_qwen(method, model_name=None):

    if method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen_flash_attn2_forward_StreamingLLM

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen_flash_attn2_forward_SnapKV

    elif method == "restkv":
        print("Using ReST-KV!")
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen_flash_attn2_forward_RestKV

    if method not in ["fullkv"]:
        transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen_new
        
def replace_gemma(method, model_name=None):
    
    if method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.gemma.modeling_gemma.GemmaFlashAttention2.forward = gemma_flash_attn2_forward_StreamingLLM

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.gemma.modeling_gemma.GemmaFlashAttention2.forward = gemma_flash_attn2_forward_SnapKV

    elif method == "restkv":
        print("Using ReST-KV!")
        transformers.models.gemma.modeling_gemma.GemmaFlashAttention2.forward = gemma_flash_attn2_forward_RestKV

    if method not in ["fullkv"]:
        transformers.models.gemma.modeling_gemma.GemmaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_gemma_new

def replace_mistral(method):

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_PyramidKV

    elif method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_StreamingLLM

    elif method == "h2o":
        print("Using H2O!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_H2O

    elif method == "cam":
        print("Using CAM!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_CAM
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_CAM
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_CAM

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_SnapKV
    
    elif method == "restkv":
        print("Using ReST-KV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_RestKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_RestKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_RestKV

    elif method == "ablation":
        print("Using Ablation!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_Ablation

    elif method == "l2norm":
        print("Using L2Norm!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_L2Norm
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_L2Norm
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_L2Norm

    # elif method == "adakv":
    #     print("Using AdaKV!")
    #     transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
    #     transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_AdaKV
    #     transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_AdaKV
    #     transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_flash_attn2_forward_AdaKV

    elif method == "headkv":
        print("Using HeadKV!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_HeadKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_HeadKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_flash_attn2_forward_HeadKV

    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new
