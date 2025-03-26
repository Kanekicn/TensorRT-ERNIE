import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time

import paddle

paddle.enable_static()

# TensorRT
import tensorrt as trt
#from calibrator import ErnieCalibrator as ErnieCalibrator
from trt_helper import *

"""
这个文件是一个将Paddle模型转换为TensorRT引擎的工具。主要功能包括：
模型结构转换：将ERNIE模型的各个组件（嵌入层、多头注意力层、前馈网络等）转换为TensorRT层。
权重加载：从Paddle保存的模型中加载权重，并将其应用到TensorRT网络中。
优化配置：设置TensorRT的优化参数，如精度模式（FP16/FP32）、工作空间大小等。
动态形状支持：通过优化配置文件支持动态输入形状。
辅助网络集成：除了主要的ERNIE模型外，还包含一个辅助网络分支，两者的输出会合并产生最终预测。
该工具的主要目的是将Paddle训练的ERNIE模型转换为可以高效推理的TensorRT引擎，以获得更好的推理性能。
"""

import os
ld_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ('/root/TensorRT-8.5.3.1/targets/x86_64-linux-gnu/lib:'
                                 '/opt/orion/orion_runtime/gpu/cuda:'
                                 '/opt/orion/orion_runtime/lib:'
                                 '/usr/lib64:'
                                 '/usr/lib:'
                                 '~/TensorRT-8.5.3.1/lib:'
                                 '/usr/local/nvidia/lib:'
                                 '/usr/local/nvidia/lib64:' + ld_lib_path)


"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
# if not handle:
    # raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

slice_output_shape = None

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()


def set_tensor_name(tensor, prefix, name):
    """
    为张量设置名称，通过添加前缀和名称组合
    
    参数:
        tensor: TensorRT张量对象
        prefix: 名称前缀
        name: 张量名称
    """
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    """
    为层的输出张量设置名称
    
    参数:
        layer: TensorRT层对象
        prefix: 名称前缀
        name: 输出名称
        out_idx: 输出索引，默认为0
    """
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    """
    设置层输出的动态范围，用于量化
    
    参数:
        layer: TensorRT层对象
        maxval: 最大值，范围将设置为[-maxval, maxval]
        out_idx: 输出索引，默认为0
    """
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(config):
    """
    根据配置获取多头注意力机制的数据类型
    
    参数:
        config: 配置对象，包含fp16、int8等配置项
    
    返回:
        对应的TensorRT数据类型的整数表示
    """
    dtype = trt.float32
    if config.fp16:
        dtype = trt.float16
    # 多头注意力默认不使用INT8输入和输出，除非特别指定
    if config.int8 and config.use_int8_multihead and not config.is_calib_mode:
        dtype = trt.int8
    return int(dtype)
def build_attention_layer(network_helper, prefix, config, weights_dict, x, mask):
    """
    构建多头注意力层
    
    参数:
        network_helper: TensorRT网络辅助对象
        prefix: 权重名称前缀
        config: 配置对象
        weights_dict: 权重字典
        x: 输入张量
        mask: 注意力掩码张量
    
    返回:
        注意力层的输出张量
    
    步骤:
    1. 设置多头注意力参数
    2. 构建查询(Q)、键(K)、值(V)三个线性变换
    3. 重塑并转置Q、K、V以适应多头注意力计算
    4. 计算注意力分数并应用缩放和掩码
    5. 应用softmax获取注意力权重
    6. 与V相乘获取加权结果
    7. 重塑并应用输出线性变换
    """
    local_prefix = prefix + "multi_head_att_"

    # 设置多头注意力参数
    num_heads = 12
    head_size = 64  # 768 / 12

    # 构建查询(Q)线性变换
    q_w = weights_dict[local_prefix + "query_fc.w_0"]
    q_b = weights_dict[local_prefix + "query_fc.b_0"]
    q = network_helper.addLinear(x, q_w, q_b)
    # 重塑并转置Q以适应多头注意力计算: [batch, seq_len, hidden] -> [batch, heads, seq_len, head_size]
    q = network_helper.addShuffle(q, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_q_view_transpose")

    # 构建键(K)线性变换
    k_w = weights_dict[local_prefix + "key_fc.w_0"]
    k_b = weights_dict[local_prefix + "key_fc.b_0"]
    k = network_helper.addLinear(x, k_w, k_b)
    # 重塑并转置K: [batch, seq_len, hidden] -> [batch, heads, head_size, seq_len]
    k = network_helper.addShuffle(k, None, (0, -1, num_heads, head_size), (0, 2, 3, 1), "att_k_view_and transpose")

    # 构建值(V)线性变换
    v_w = weights_dict[local_prefix + "value_fc.w_0"]
    v_b = weights_dict[local_prefix + "value_fc.b_0"]
    v = network_helper.addLinear(x, v_w, v_b)
    # 重塑并转置V: [batch, seq_len, hidden] -> [batch, heads, seq_len, head_size]
    v = network_helper.addShuffle(v, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_v_view_and transpose")

    # 计算注意力分数: Q * K^T
    scores = network_helper.addMatMul(q, k, "q_mul_k")

    # 应用缩放因子: 1/sqrt(head_size)
    scores = network_helper.addScale(scores, 1/math.sqrt(head_size))

    # 应用注意力掩码
    scores = network_helper.addAdd(scores, mask)

    # 应用softmax获取注意力权重
    attn = network_helper.addSoftmax(scores, dim=-1)

    # 与V相乘获取加权结果
    attn = network_helper.addMatMul(attn, v, "matmul(p_attn, value)")

    # 重塑结果: [batch, heads, seq_len, head_size] -> [batch, seq_len, hidden]
    attn = network_helper.addShuffle(attn, (0, 2, 1, 3), (0, -1, 1, num_heads * head_size), None, "attn_transpose_and_reshape")

    # 应用输出线性变换
    out_w = weights_dict[local_prefix + "output_fc.w_0"]
    out_b = weights_dict[local_prefix + "output_fc.b_0"]
    attn_output = network_helper.addLinear(attn, out_w, out_b)

    return attn_output

def build_mlp_layer(network_helper, prefix, config, weights_dict, x):
    """
    构建前馈神经网络(MLP)层
    
    参数:
        network_helper: TensorRT网络辅助对象
        prefix: 权重名称前缀
        config: 配置对象
        weights_dict: 权重字典
        x: 输入张量
    
    返回:
        MLP层的输出张量
    
    步骤:
    1. 应用第一个线性变换
    2. 应用ReLU激活函数
    3. 应用第二个线性变换
    """
    local_prefix = prefix + "ffn_"
    
    # 第一个线性变换
    fc1_w = weights_dict[local_prefix + "fc_0.w_0"]
    fc1_b = weights_dict[local_prefix + "fc_0.b_0"]
    x = network_helper.addLinear(x, fc1_w, fc1_b)

    # ReLU激活
    x = network_helper.addReLU(x)

    # 第二个线性变换
    fc2_w = weights_dict[local_prefix + "fc_1.w_0"]
    fc2_b = weights_dict[local_prefix + "fc_1.b_0"]
    x = network_helper.addLinear(x, fc2_w, fc2_b)

    return x

def build_embeddings_layer(network_helper, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor):
    """
    构建嵌入层，包括词嵌入、位置嵌入和句子嵌入
    
    参数:
        network_helper: TensorRT网络辅助对象
        weights_dict: 权重字典
        src_ids_tensor: 源ID张量
        sent_ids_tensor: 句子ID张量
        pos_ids_tensor: 位置ID张量
    
    返回:
        嵌入层的输出张量
    
    步骤:
    1. 获取词嵌入、句子嵌入和位置嵌入权重
    2. 应用各种嵌入查找
    3. 将三种嵌入相加得到最终嵌入
    """
    # 获取嵌入权重
    word_embedding = weights_dict["word_embedding"]
    sent_embedding = weights_dict["sent_embedding"]
    pos_embedding = weights_dict["pos_embedding"]

    # 应用嵌入查找
    src_embedded = network_helper.addEmbedding(src_ids_tensor, word_embedding, "word_embedding")
    pos_embedded = network_helper.addEmbedding(pos_ids_tensor, pos_embedding, "pos_embedding")
    sent_embedded = network_helper.addEmbedding(sent_ids_tensor, sent_embedding, "sent_embedding")

    # 将三种嵌入相加
    x = network_helper.addAdd(src_embedded, pos_embedded)
    x = network_helper.addAdd(x, sent_embedded)

    return x

def build_block_layer(network_helper, prefix, config, weights_dict, x, mask):
    """
    构建Transformer块层，包括自注意力和前馈网络
    
    参数:
        network_helper: TensorRT网络辅助对象
        prefix: 权重名称前缀
        config: 配置对象
        weights_dict: 权重字典
        x: 输入张量
        mask: 注意力掩码张量
    
    返回:
        块层的输出张量
    
    步骤:
    1. 应用自注意力层
    2. 添加残差连接
    3. 应用层归一化
    4. 应用前馈网络
    5. 添加残差连接
    6. 应用层归一化
    """
    local_prefix = prefix
    h = x

    # 应用自注意力层
    x = build_attention_layer(network_helper, local_prefix, config, weights_dict, x, mask)

    # 添加残差连接
    x = network_helper.addAdd(x, h)

    # 应用注意力后的层归一化
    post_att_norm_weight = weights_dict[local_prefix + "post_att_layer_norm_scale"]
    post_att_norm_bias = weights_dict[local_prefix + "post_att_layer_norm_bias"]
    x = network_helper.addLayerNorm(x, post_att_norm_weight, post_att_norm_bias)

    h = x

    # 应用前馈网络
    x = build_mlp_layer(network_helper, local_prefix, config, weights_dict, x)

    # 添加残差连接
    x = network_helper.addAdd(x, h)

    # 应用前馈网络后的层归一化
    fnn_norm_weight = weights_dict[local_prefix + "post_ffn_layer_norm_scale"]
    fnn_norm_bias = weights_dict[local_prefix + "post_ffn_layer_norm_bias"]
    x = network_helper.addLayerNorm(x, fnn_norm_weight, fnn_norm_bias)

    return x

def build_encoder_layer(network_helper, prefix, config, weights_dict, x, mask):
    """
    构建编码器层，包含多个Transformer块
    
    参数:
        network_helper: TensorRT网络辅助对象
        prefix: 权重名称前缀
        config: 配置对象
        weights_dict: 权重字典
        x: 输入张量
        mask: 注意力掩码张量
    
    返回:
        编码器的输出张量
    
    步骤:
    1. 应用多个Transformer块
    2. 提取第一个token的表示(CLS token)
    3. 应用池化层线性变换
    """
    # 应用12个Transformer块
    for layer in range(0, 12):
        local_prefix = prefix + "layer_{}_".format(layer)
        x = build_block_layer(network_helper, local_prefix, config, weights_dict, x, mask)

    # 准备切片操作，提取第一个token的表示(CLS token)
    x_shape_len = len(x.shape)
    start = np.zeros(x_shape_len, dtype=np.int32)
    start_tensor = network_helper.addConstant(start)

    # 添加切片层
    slice_layer = network_helper.network.add_slice(x, start, start, (1, 1, 1, 1))
    slice_layer.set_input(1, start_tensor)
    slice_layer.set_input(2, slice_output_shape)
    sliced = slice_layer.get_output(0)
    print("sliced")

    # 应用池化层线性变换
    pooled_w = weights_dict["pooled_fc.w_0"]
    pooled_b = weights_dict["pooled_fc.b_0"]
    x = network_helper.addLinear(sliced, pooled_w, pooled_b)

    return x

def build_ernie_model(network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor):
    """
    构建ERNIE模型
    
    参数:
        network_helper: TensorRT网络辅助对象
        config: 配置对象
        weights_dict: 权重字典
        src_ids_tensor: 源ID张量
        sent_ids_tensor: 句子ID张量
        pos_ids_tensor: 位置ID张量
        input_mask_tensor: 输入掩码张量
    
    返回:
        ERNIE模型的输出张量
    
    步骤:
    1. 构建嵌入层
    2. 应用编码器前的层归一化
    3. 构建编码器层
    4. 应用tanh激活函数
    """
    prefix = "encoder_"
    
    # 构建嵌入层
    embeddings = build_embeddings_layer(network_helper, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor)

    # 应用编码器前的层归一化
    pre_encoder_norm_weight = weights_dict["pre_encoder_layer_norm_scale"]
    pre_encoder_norm_bias = weights_dict["pre_encoder_layer_norm_bias"]
    x = network_helper.addLayerNorm(embeddings, pre_encoder_norm_weight, pre_encoder_norm_bias)

    # 构建编码器层
    encoder_out = build_encoder_layer(network_helper, prefix, config, weights_dict, x, input_mask_tensor)

    # 应用tanh激活函数
    x = network_helper.addTanh(encoder_out)

    return x

def build_aside(network_helper, weights_dict, tensor_list):
    """
    构建辅助网络分支
    
    参数:
        network_helper: TensorRT网络辅助对象
        weights_dict: 权重字典
        tensor_list: 输入张量列表
    
    返回:
        辅助网络的输出张量
    
    步骤:
    1. 对多个字段应用嵌入
    2. 连接所有嵌入结果
    3. 重塑张量
    4. 应用两个线性变换和ReLU激活
    5. 应用最终的线性变换
    """
    # 获取多字段嵌入权重
    multi_field_0 = weights_dict["multi_field_0"]
    multi_field_1 = weights_dict["multi_field_1"]
    multi_field_2 = weights_dict["multi_field_2"]
    multi_field_3 = weights_dict["multi_field_3"]
    multi_field_4 = weights_dict["multi_field_4"]
    multi_field_5 = weights_dict["multi_field_5"]
    multi_field_6 = weights_dict["multi_field_6"]
    multi_field_7 = weights_dict["multi_field_7"]

    # 应用嵌入查找
    x0 = network_helper.addEmbedding(tensor_list[0], multi_field_0, "multi_field_0")
    x1 = network_helper.addEmbedding(tensor_list[1], multi_field_1, "multi_field_1")
    x2 = network_helper.addEmbedding(tensor_list[2], multi_field_2, "multi_field_2")
    x3 = network_helper.addEmbedding(tensor_list[3], multi_field_3, "multi_field_3")
    x4 = network_helper.addEmbedding(tensor_list[4], multi_field_4, "multi_field_4")
    x5 = network_helper.addEmbedding(tensor_list[5], multi_field_5, "multi_field_5")
    x6 = network_helper.addEmbedding(tensor_list[6], multi_field_6, "multi_field_6")
    x7 = network_helper.addEmbedding(tensor_list[7], multi_field_7, "multi_field_7")

    # 连接所有嵌入结果
    concat_tensors = [x0, x1, x2, x3, x4, x5, x6, x7]
    x = network_helper.addCat(concat_tensors, dim=1)

    # 重塑张量为4D格式
    x = network_helper.addShuffle(x, None, (-1, 1, 1, 160), None, "aside_reshape")

    # 第一个线性变换和ReLU激活
    feature_emb_fc_w = weights_dict["feature_emb_fc_w"]
    feature_emb_fc_b = weights_dict["feature_emb_fc_b"]
    x = network_helper.addLinear(x, feature_emb_fc_w, feature_emb_fc_b)
    x = network_helper.addReLU(x)

    # 保存输出形状，用于后续切片操作
    global slice_output_shape
    slice_output_shape = network_helper.network.add_shape(x).get_output(0)

    # 第二个线性变换和ReLU激活
    feature_emb_fc_w2 = weights_dict["feature_emb_fc_w2"]
    feature_emb_fc_b2 = weights_dict["feature_emb_fc_b2"]
    x = network_helper.addLinear(x, feature_emb_fc_w2, feature_emb_fc_b2)
    x = network_helper.addReLU(x)

    # 最终的线性变换
    cls_out_w_aside = weights_dict["cls_out_w_aside"]
    cls_out_b_aside = weights_dict["cls_out_b_aside"]
    x = network_helper.addLinear(x, cls_out_w_aside, cls_out_b_aside)
    return x

def process_input_mask(network_helper, input_mask_tensor):
    """
    处理输入掩码，生成注意力偏置
    
    参数:
        network_helper: TensorRT网络辅助对象
        input_mask_tensor: 输入掩码张量
    
    返回:
        处理后的注意力偏置张量
    
    步骤:
    1. 计算注意力偏置矩阵
    2. 将偏置转换为负值掩码
    3. 缩放偏置值
    4. 重塑张量以适应多头注意力
    """
    # 计算注意力偏置矩阵: input_mask * input_mask^T
    attn_bias = network_helper.addMatMul(input_mask_tensor, input_mask_tensor, False, True, "get attn_bias")

    # 将偏置转换为负值掩码: (1 - attn_bias) * -10000.0
    tmp_arr = np.array([[[-1.]]], dtype=np.float32)
    tmp_tensor = network_helper.addConstant(tmp_arr)
    attn_bias = network_helper.addAdd(attn_bias, tmp_tensor)

    # 缩放偏置值
    attn_bias = network_helper.addScale(attn_bias, 10000.0)
    
    # 重塑张量以适应多头注意力
    attn_bias = network_helper.addShuffle(attn_bias, None, (0, 1, 0, -1), None, "input_mask.unsqueeze(-1)")

    return attn_bias

def build_model(network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, aside_tensor_list):
    """
    构建完整模型，包括ERNIE模型和辅助网络
    
    参数:
        network_helper: TensorRT网络辅助对象
        config: 配置对象
        weights_dict: 权重字典
        src_ids_tensor: 源ID张量
        sent_ids_tensor: 句子ID张量
        pos_ids_tensor: 位置ID张量
        input_mask_tensor: 输入掩码张量
        aside_tensor_list: 辅助网络输入张量列表
    
    返回:
        模型的最终输出张量
    
    步骤:
    1. 构建辅助网络
    2. 处理输入掩码
    3. 构建ERNIE模型
    4. 应用分类器线性变换
    5. 将ERNIE输出和辅助网络输出相加
    6. 应用sigmoid激活函数
    """
    # 构建辅助网络
    cls_aside_out = build_aside(network_helper, weights_dict, aside_tensor_list)

    # 处理输入掩码
    input_mask_tensor = process_input_mask(network_helper, input_mask_tensor)

    # 构建ERNIE模型
    x = build_ernie_model(network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor)

    # 应用分类器线性变换
    cls_out_w = weights_dict["cls_out_w"]
    cls_out_b = weights_dict["cls_out_b"]
    cls_out = network_helper.addLinear(x, cls_out_w, cls_out_b)

    # 将ERNIE输出和辅助网络输出相加
    x = network_helper.addAdd(cls_out, cls_aside_out)

    # 应用sigmoid激活函数得到最终预测
    x = network_helper.addSigmoid(x)

    return x

def build_engine(args, config, weights_dict, calibrationCacheFile):
    """
    构建TensorRT引擎
    
    参数:
        args: 命令行参数
        config: 配置对象
        weights_dict: 权重字典
        calibrationCacheFile: 校准缓存文件
    
    返回:
        构建好的TensorRT引擎
    
    步骤:
    1. 创建TensorRT构建器、网络和配置
    2. 设置工作空间大小和精度模式
    3. 创建网络辅助对象
    4. 添加网络输入
    5. 构建模型
    6. 标记输出
    7. 创建优化配置文件
    8. 构建引擎
    """
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        # 设置工作空间大小
        builder_config.max_workspace_size = args.workspace_size * (1024 * 1024)

        # 设置精度模式
        plugin_data_type:int = 0
        if args.fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            plugin_data_type = 1

        if args.strict:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        #  if args.use_strict:
            #  builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # 创建网络辅助对象
        network_helper = TrtNetworkHelper(network, plg_registry, TRT_LOGGER, plugin_data_type)

        # 添加网络输入
        src_ids_tensor = network_helper.addInput(name="src_ids", dtype=trt.int32, shape=(-1, -1, 1))
        pos_ids_tensor = network_helper.addInput(name="pos_ids", dtype=trt.int32, shape=(-1, -1, 1))
        sent_ids_tensor = network_helper.addInput(name="sent_ids", dtype=trt.int32, shape=(-1, -1, 1))
        input_mask_tensor = network_helper.addInput(name="input_mask", dtype=trt.float32, shape=(-1, -1, 1))

        # 添加辅助网络输入
        tmp6_tensor = network_helper.addInput(name="tmp6", dtype=trt.int32, shape=(-1, 1, 1))
        tmp7_tensor = network_helper.addInput(name="tmp7", dtype=trt.int32, shape=(-1, 1, 1))
        tmp8_tensor = network_helper.addInput(name="tmp8", dtype=trt.int32, shape=(-1, 1, 1))
        tmp9_tensor = network_helper.addInput(name="tmp9", dtype=trt.int32, shape=(-1, 1, 1))
        tmp10_tensor = network_helper.addInput(name="tmp10", dtype=trt.int32, shape=(-1, 1, 1))
        tmp11_tensor = network_helper.addInput(name="tmp11", dtype=trt.int32, shape=(-1, 1, 1))
        tmp12_tensor = network_helper.addInput(name="tmp12", dtype=trt.int32, shape=(-1, 1, 1))
        tmp13_tensor = network_helper.addInput(name="tmp13", dtype=trt.int32, shape=(-1, 1, 1))

        aside_tensor_list = [tmp6_tensor, tmp7_tensor, tmp8_tensor, tmp9_tensor, tmp10_tensor, tmp11_tensor, tmp12_tensor, tmp13_tensor]

        # 构建模型
        out = build_model(network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, aside_tensor_list)

        # 标记输出
        network_helper.markOutput(out)

        if args.cuda_graph:
            batchs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            seq_lens = [1, 32, 64, 96, 128]
            for b in batchs:
                for s in seq_lens:
                    profile = builder.create_optimization_profile()
                    static_shape = (b, s, 1)
                    profile.set_shape("src_ids", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("sent_ids", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("pos_ids", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("input_mask", min=static_shape, opt=static_shape, max=static_shape)

                    static_shape = (b, 1, 1)
                    profile.set_shape("tmp6", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("tmp7", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("tmp8", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("tmp9", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("tmp10", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("tmp11", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("tmp12", min=static_shape, opt=static_shape, max=static_shape)
                    profile.set_shape("tmp13", min=static_shape, opt=static_shape, max=static_shape)
                    builder_config.add_optimization_profile(profile)
                else:
                    # 创建优化配置文件
                    profile = builder.create_optimization_profile()

                    # 设置主要输入的形状范围
                    min_shape = (1, 128, 1)
                    opt_shape = (5, 128, 1)
                    max_shape = (10, 128, 1)
                    profile.set_shape("src_ids", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("sent_ids", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("pos_ids", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("input_mask", min=min_shape, opt=opt_shape, max=max_shape)

                    # 设置辅助输入的形状范围
                    min_shape = (1, 1, 1)
                    opt_shape = (5, 1, 1)
                    max_shape = (10, 1, 1)
                    profile.set_shape("tmp6", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp7", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp8", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp9", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp10", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp11", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp12", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp13", min=min_shape, opt=opt_shape, max=max_shape)

                    # 添加优化配置文件
                    builder_config.add_optimization_profile(profile)

        # 构建引擎并计时
        build_start_time = time.time()
        engine = builder.build_engine(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))
        return engine

def load_paddle_weights(path_prefix):
    """
    从Paddle模型加载权重
    
    参数:
        path_prefix: Paddle模型路径前缀
    
    返回:
        包含所有权重的字典
    
    步骤:
    1. 创建Paddle执行器
    2. 加载推理模型
    3. 获取模型状态字典
    4. 将Paddle张量转换为NumPy数组
    5. 构建权重字典
    """
    # 创建CPU执行器
    exe = paddle.static.Executor(paddle.CPUPlace())
    
    # 加载推理模型
    [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(path_prefix, exe, model_filename="__model__", params_filename="__params__"))

    # 获取模型状态字典
    state_dict = inference_program.state_dict()

    print(feed_target_names)

    # 构建权重字典
    tensor_dict = {}
    for i in state_dict:
        # 将Paddle张量转换为NumPy数组
        arr = np.array(state_dict[i])
        # print(arr.shape)

        tensor_dict[i] = arr

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Load paddle model. Found {:} entries in weight map".format(len(tensor_dict)))

    return tensor_dict


def main():
    """
    主函数，解析命令行参数并构建TensorRT引擎
    
    步骤:
    1. 解析命令行参数
    2. 加载Paddle模型权重
    3. 构建TensorRT引擎
    4. 序列化并保存引擎
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--paddle", required=True, help="The paddle model dir path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-b", "--max_batch_size", default=1, type=int, help="max batch size")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true", help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=3000, help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-g", "--cuda_graph", action="store_true", help="Indicates that inference should be run with CUDA Graph", required=False)
    # parser.add_argument("-n", "--calib-num", help="calibration cache path", required=False)

    # 解析命令行参数
    args, _ = parser.parse_known_args()

    # 加载Paddle模型权重
    if args.paddle != None:
        weights_dict = load_paddle_weights(args.paddle)
    else:
        raise RuntimeError("You need either specify paddle using option --paddle to build TRT model.")

    # 配置对象，在此示例中为空列表
    config = []
    
    # 构建TensorRT引擎
    with build_engine(args, config, weights_dict, None) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        # 序列化引擎
        serialized_engine = engine.serialize()
        # 保存引擎到文件
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

    # if args.img_path is not None:
    #     infer_helper = InferHelper(args.output, TRT_LOGGER)
    #     test_case_data(infer_helper, args, args.img_path)

if __name__ == "__main__":
    main()
