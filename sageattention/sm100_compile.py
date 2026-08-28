from . import _qattn_sm100
from ._registration import register_qattn_ops

_OPS = register_qattn_ops(_qattn_sm100, "sageattention_sm100", [
    ("qk_int8_sv_f8_accum_f32_attn", "base"),
    ("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", "vscale"),
])

qk_int8_sv_f8_accum_f32_attn = _OPS["qk_int8_sv_f8_accum_f32_attn"]
qk_int8_sv_f8_accum_f32_fuse_v_scale_attn = _OPS["qk_int8_sv_f8_accum_f32_fuse_v_scale_attn"]
