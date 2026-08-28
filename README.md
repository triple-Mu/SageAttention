# SageAttention
<!-- We are continuously updating more features. You could **Star** and **Watch** our repository to stay updated.

--- -->
This repository provides the official implementation of SageAttention, SageAttention2, and SageAttention2++, which achieve surprising speedup on most GPUs without lossing accuracy across all models in a plug-and-play way.

**SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration**  
Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen  
Paper: https://arxiv.org/abs/2410.02367

**SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization**  
Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen  
Paper: https://arxiv.org/abs/2411.10958

**SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training**  
Jintao Zhang, Jia Wei, Haoxu Wang, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Kai Jiang, Jianfei Chen, Jun Zhu  
Paper: https://arxiv.org/abs/2505.11594


![Local Image](./assets/2.png)
*Note: [SageAttention2++](https://arxiv.org/pdf/2505.21136) achieves higher speed while maintaining the same accuracy performance.*

## Current Features
<!-- This is a beta release of SageAttention2. We welcome any feedback on accuracy, performance issues, bugs, feature requests, or suggestions. Please feel free to open an issue or launch a pull request! -->

+ Optmized kernels for **Ampere, Ada and Hopper GPUs.**
+ INT8 quantization and smoothing for $QK^\top$ with support for varying granularities.
+ FP8 quantization for $PV$, and FP16 accumulator for FP8/FP16 $PV$.
+ Two-level accumulation strategy for $PV$ to improve accuracy in FP8 MMA and WGMMA.
+ Support `torch.compile` (fullgraph, zero graph breaks) including CUDA graphs / `mode="reduce-overhead"`, and distributed inference.


## Project Updates
- [2025-09-27]: 🎉 [SageAttention3](https://arxiv.org/abs/2505.11594) is accepted by NeurIPS 2025 as a **Spotlight** paper! 
- [2025-09-27]: The code of [SageAttention3](https://arxiv.org/abs/2505.11594) is released in this repository at  [sageattention3_blackwell](./sageattention3_blackwell/). We would still greatly appreciate it if you could take a moment to fill out the Form in [Huggingface](https://huggingface.co/jt-zhang/SageAttention3). Please note that since SageAttention2 is more accurate, we still recommend using SageAttention2 for precision-sensitive applications.
- [2025-07-01]: The code of [SageAttention2++](https://arxiv.org/pdf/2505.21136) is released in this repository. We would still greatly appreciate it if you could take a moment to fill out the Form in [Huggingface](https://huggingface.co/jt-zhang/SageAttention2_plus). Thank you very much!

![Local Image](./assets/5090_sageattn2++.png)

![Local Image](./assets/4090_sageattn2++.png)

- [2025-06-19]: [Sparse SageAttention1 API](https://github.com/jt-zhang/Sparse_SageAttention_API) and [Sparse SageAttention2 API](https://github.com/thu-ml/SpargeAttn) can compute attention with any block sparse pattern very fast.
- [2025-05-02]: 🎉SageAttention2 and [SpargeAttn](https://github.com/thu-ml/SpargeAttn) are accepted by ICML 2025! 
- [2025-02-25]: 🔥 We release [SpargeAttn](https://github.com/thu-ml/SpargeAttn), a sparse attention based on SageAttention2, which could acclerate any model without training.
- [2025-02-15]: 🔥 The compilation code is updated to support RTX5090! On RTX5090, SageAttention reaches 560T, 2.7x faster than FlashAttention2!
- [2025-01-28]: 🔥⚡SageAttention is now available on Hopper GPUs (H100, H800, H20)! It matches the speed of FlashAttention3-FP8 but offers **much better accuracy!**

| **FlashAttention2** | **FlashAttention3** | **FlashAttention3-FP8** | **SageAttention** |
|----------------------|----------------------|----------------------|----------------------|
| ![FlashAttention2](assets/cogvideox1.5_fa2_example.gif) | ![FlashAttention3](assets/cogvideox1.5_fa3_example.gif)  | ![FlashAttention3-FP8](assets/cogvideox1.5_fa3fp8_example.gif) | ![SageAttention](assets/cogvideox1.5_sage_example.gif) |
| **25'34''** | **17'32''** | **12'14''** | **12'07''** |

*Results for [CogVideoX1.5-5B](https://huggingface.co/THUDM/CogVideoX1.5-5B) on NVIDIA H20 GPU*

![Local Image](./assets/H100_hd128.png)

![Local Image](./assets/H20_hd128.png)

- [2025-01-24]: 🎉SageAttention is accepted by ICLR 2025! 
- [2024-12-20]: 🔥Update the [SageAttention2 Paper](https://arxiv.org/abs/2411.10958).   

- [2024-12-20]: 🔥Release SageAttention 2.0.1 Beta! In this version, we introduce a new feature: per-thread quantization, which offers finer granularity while maintaining hardware efficiency.
- [2024-11-21]: 🔥SageAttention 2.0.0 beta is released! Now SageAttention has measured speedup on L20, L40, A100, A800, and A6000, RTX3090 and RTX4090.
- [2024-11-11]: Support for different sequence lengths between `q` and `k,v`,  `(batch_size, head_num, seq_len, head_dim)` or `(batch_size, seq_len, head_num, head_dim)` input shapes, and `group-query attention` is available now.


## Installation
### Base environment
+ `python>=3.9`, `torch>=2.6.0` (the wheel is `cp39-abi3`: one build covers every Python >= 3.9, but stays tied to the torch major.minor it was built against)
+ `cmake>=3.26` and `ninja` in the build environment (`pip install "cmake>=3.31" ninja`; pulled in automatically under build isolation)
- `CUDA`:
  + `>=13.0` for sm110 (tcgen05)
  + `>=12.8` for Blackwell or SageAttention2++
  + `>=12.4` for fp8 support on Ada
  + `>=12.3` for fp8 support on Hopper
  + `>=12.0` for Ampere
+ `flash-attn` for benchmarking

### Install Package

For SageAttention V1 in Triton (slower than SageAttention V2/V2++/V3), refer to [SageAttention-1](https://github.com/thu-ml/SageAttention/tree/sageattention-1) branch and install using pip: `pip install sageattention==1.0.6`

Compile from source (CMake drives the CUDA build; setup.py stays the entry
point, so `pip install -e .`, `pip install .`, `python setup.py install` and
`python setup.py bdist_wheel` all work):
```
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
# arch list: defaults to the local GPUs; set explicitly for cross-compilation
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"   # Optional
pip install -e . --no-build-isolation
```
Useful knobs: `MAX_JOBS` (cmake build parallelism), `NVCC_THREADS` (per-TU nvcc
parallelism; note `MAX_JOBS x NVCC_THREADS` is the real process count),
`SAGEATTN_PTXAS_VERBOSE=1`, `SAGEATTN_LINEINFO=1`, `DEBUG=1`,
`SAGEATTN_CMAKE_ARGS` (extra cmake args).

All kernels land in a single `sageattention/_C.abi3.so`; the operators are
exposed as `torch.ops.sageattention.*` (TORCH_LIBRARY, no pybind11) and the
right kernel is dispatched automatically from the GPU's compute capability,
like `torch.ops.aten._scaled_dot_product_cudnn_attention`.

To benchmark the speed against FlashAttention3, please compile FlashAttention3 from source:
```
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
git checkout b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7 # 2.7.2 release
cd hopper
python setup.py install
```

## How to Use
```python
from sageattention import sageattn
attn_output = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
```
+ `q, k, v` are **FP16/BF16** dtype with the shape `(batch_size, head_num, seq_len, head_dim)` using default `tensor_layout="HND"`. For shape `(batch_size, seq_len, head_num, head_dim)`, set `tensor_layout="NHD"`. 
+ `is_causal` determines the use of a causal mask.

### Available APIs:
+ `sageattn` is the single public entry point: it picks the kernel from the
  GPU's compute capability (sm80/sm86 -> INT8+FP16PV, sm89/sm12x -> INT8+FP8PV,
  sm90 -> Hopper wgmma path, sm100/sm110 -> tcgen05 opt-in). The per-arch
  wrapper functions of v2.x (`sageattn_qk_int8_pv_fp16_cuda`,
  `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp8_cuda_sm90`, ...)
  and the Triton-based `sageattn_varlen` were removed.
+ Advanced knobs are explicit keyword arguments of `sageattn`
  (`qk_quant_gran`, `pv_accum_dtype` — `"fp32+fp16"` is SageAttention2++ —,
  `smooth_k`, `smooth_v`); `None` keeps the per-device default. Unknown kwargs
  now raise instead of being silently ignored.
+ The quantization building blocks are available as
  `torch.ops.sageattention.*` (`quant_qk`, `quant_v_fp8`, `fwd`, ...), all
  registered with fake kernels so `torch.compile(fullgraph=True)` and CUDA
  graphs work end to end.

For optimal speed and accuracy performance on custom devices and models, we strongly recommend referring to the [this file](./sageattention/core.py) for detailed guidance.

> **Note:**
Support for different sequence lengths between `q` and `k,v` and `group-query attention` is available.


### Plug-and-play Example

We can replace `scaled_dot_product_attention` easily. 
We will take [CogvideoX](https://huggingface.co/zai-org/CogVideoX-2b) as an example:

Add the following codes and run
```diff
import torch.nn.functional as F

+ from sageattention import sageattn
+ F.scaled_dot_product_attention = sageattn

```

Specifically,

```bash
cd example
python cogvideox_infer.py --model cogvideox-2b --compile --attention_type sage
```

**You can get a lossless video in** `./example/videos/<model>/<attention_type>/` **faster than by using** `--attention_type sdpa`. More examples and guidance can be found under the `example/` directory.

> **Note:** Not all models works with `F.scaled_dot_product_attention = sageattn`. Technically, you should replace the original Attention by modifying the `Attention Class` of the target model. For image and video models, we suggest only replacing the attention in DiT (see `example/modify_mochi.py` for detail).

### Kernel Benchmarking
We provide a benchmarking script to compare the speed of different kernels including SageAttention, FlashAttention2 and FlashAttention3. Please refer to the `benchmark/` directory for more details.
 
## Performance
### Speed of Kernels

`8+8` means the kernel with INT8 quantization for $QK^\top$ and FP8 quantization for $PV$. `8+16` uses FP16 with FP16 accumulator for $PV$.

![Local Image](./assets/5090_sageattn2++.png)

![Local Image](./assets/4090_sageattn2++.png)

![Local Image](./assets/4090_hd128.png)

![Local Image](./assets/L20_hd128.png)

![Local Image](./assets/H100_hd128.png)

![Local Image](./assets/H20_hd128.png)

![Local Image](./assets/A100_hd128.png)

![Local Image](./assets/3090_hd128.png)

> **Note:** The TOPS results refer only to the Attention Kernel, excluding the quantization and smoothing.

### End-to-end Performance
#### **End-to-End Accuracy:**

![Local Image](./assets/22.png)

![Local Image](./assets/23.png)

![Local Image](./assets/24.png)

![Local Image](./assets/25.png)

#### **End-to-End Speedup:**

![Local Image](./assets/26.png)
*Note: SageAttention2++ achieves higher speed.*

## Citation
**If you use this code or find our work valuable, please cite:**
```
@inproceedings{zhang2025sageattention,
  title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
@inproceedings{zhang2024sageattention2,
  title={Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization},
  author={Zhang, Jintao and Huang, Haofeng and Zhang, Pengle and Wei, Jia and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
@article{zhang2025sageattention2++,
  title={Sageattention2++: A more efficient implementation of sageattention2},
  author={Zhang, Jintao and Xu, Xiaoming and Wei, Jia and Huang, Haofeng and Zhang, Pengle and Xiang, Chendong and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.21136},
  year={2025}
}
@article{zhang2025sageattention3,
  title={SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training},
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Xu, Xiaoming and Huang, Haofeng and Wang, Haoxu and Jiang, Kai and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.11594},
  year={2025}
}
```
