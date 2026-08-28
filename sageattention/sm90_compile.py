from . import _qattn_sm90
from ._registration import register_qattn_ops

_OPS = register_qattn_ops(_qattn_sm90, "sageattention_sm90", [
    ("qk_int8_sv_f8_accum_f32_attn_inst_buf", "base"),
    ("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", "vscale"),
])

qk_int8_sv_f8_accum_f32_attn_inst_buf = _OPS["qk_int8_sv_f8_accum_f32_attn_inst_buf"]
qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf = _OPS["qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"]
