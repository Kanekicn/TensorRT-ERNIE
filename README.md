# 基于 TensorRT 的 ERNIE + Aside 特征融合模型推理加速

基于 TensorRT API 手动搭建 **ERNIE + Aside** 融合模型推理网络，支持动态 shape 和 batch；

优化1: 实现 LayerNorm Plugin 进行算子融合，结合 **FP16** 混合精度推理，减少 kernel 数量与内存 IO

优化2: 输入合并优化与CUDA Graph实现

优化3: Attention 算子的子图融合

