from . import _qattn_sm120
from ._registration import register_qattn_ops

_OPS = register_qattn_ops(_qattn_sm120, "sageattention_sm120", [
    ("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", "vscale"),
    ("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf", "vscale"),
    ("qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn", "vscale_vmean"),
])

qk_int8_sv_f8_accum_f32_fuse_v_scale_attn = _OPS["qk_int8_sv_f8_accum_f32_fuse_v_scale_attn"]
qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf = _OPS["qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf"]
qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn = _OPS["qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn"]
