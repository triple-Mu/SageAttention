from . import _qattn_sm80
from ._registration import register_qattn_ops

_OPS = register_qattn_ops(_qattn_sm80, "sageattention", [
    ("qk_int8_sv_f16_accum_f16_attn", "base",
     "Custom CUDA kernel for SageAttention with INT8 quantization for Q and K, FP16 PV with FP16 accumulation."),
    ("qk_int8_sv_f16_accum_f32_attn", "base",
     "Custom CUDA kernel for SageAttention with INT8 quantization for Q and K, FP16 PV with FP32 accumulation."),
    ("qk_int8_sv_f16_accum_f16_attn_inst_buf", "base",
     "Custom CUDA kernel for SageAttention with INT8 quantization for Q and K, FP16 PV with FP16 accumulation."),
    ("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn", "vmean",
     "Custom CUDA kernel for SageAttention with INT8 quantization for Q and K, FP16 PV with FP16 accumulation."),
])

qk_int8_sv_f16_accum_f16_attn = _OPS["qk_int8_sv_f16_accum_f16_attn"]
qk_int8_sv_f16_accum_f32_attn = _OPS["qk_int8_sv_f16_accum_f32_attn"]
qk_int8_sv_f16_accum_f16_attn_inst_buf = _OPS["qk_int8_sv_f16_accum_f16_attn_inst_buf"]
qk_int8_sv_f16_accum_f16_fuse_v_mean_attn = _OPS["qk_int8_sv_f16_accum_f16_fuse_v_mean_attn"]
