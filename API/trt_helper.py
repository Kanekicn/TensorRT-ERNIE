#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time

from typing import Optional, Tuple
import os

os.environ['LD_LIBRARY_PATH'] = '/usr/local/TensorRT/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

"""_summary_
    这个文件是TensorRT的辅助工具类，提供了一系列方法来简化TensorRT网络的构建过程。主要功能包括：
基础操作：提供了设置张量名称、初始化插件等基础功能。
网络层构建：封装了TensorRT的各种层（如卷积、线性层、激活函数等）的创建过程，使网络构建更加简洁。
张量操作：提供了张量重塑、切片、连接等操作的简化接口。
插件支持：集成了对TensorRT插件的支持，如LayerNorm和DumpTensor插件。
复杂操作实现：通过组合基本操作实现了一些TensorRT没有直接支持的操作，如GELU激活函数和嵌入层。
该工具类大大简化了使用TensorRT构建深度学习模型的过程，特别是对于复杂模型如ERNIE/BERT等Transformer架构的模型。
    """

def set_tensor_name(tensor, prefix, name):
    """
    为TensorRT张量设置名称, 通过添加前缀和名称组合
    
    参数:
        tensor: TensorRT张量对象
        prefix: 名称前缀
        name: 张量名称
    """
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    """
    为TensorRT层的输出张量设置名称
    
    参数:
        layer: TensorRT层对象
        prefix: 名称前缀
        name: 输出名称
        out_idx: 输出索引，默认为0
    """
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    """
    设置TensorRT层输出的动态范围，用于INT8量化
    
    参数:
        layer: TensorRT层对象
        maxval: 最大值，范围将设置为[-maxval, maxval]
        out_idx: 输出索引，默认为0
    """
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(args):
    """
    根据配置获取多头注意力机制的数据类型
    
    参数:
        args: 配置对象，包含fp16、int8等配置项
    
    返回:
        对应的TensorRT数据类型的整数表示
    """
    dtype = trt.float32
    if args.fp16:
        dtype = trt.float16
    # 多头注意力默认不使用INT8输入和输出，除非特别指定
    # if config.int8 and config.use_int8_multihead and not config.is_calib_mode:
    #     dtype = trt.int8
    return int(dtype)

def init_trt_plugin(severity=None, lib_name=None, logger=None):
    """
    初始化TensorRT插件
    
    参数:
        severity: 日志级别，默认为INFO
        lib_name: 额外的插件库名称
        logger: TensorRT日志记录器
    
    返回:
        初始化后的TensorRT日志记录器
    
    步骤:
    1. 设置日志级别
    2. 创建日志记录器
    3. 加载必要的插件库
    4. 初始化TensorRT插件
    """
    # 设置日志级别
    if severity is None:
        severity = trt.Logger.INFO

    # 创建日志记录器
    if logger is None:
        logger = trt.Logger(severity)

    # 加载必要的插件库
    lib_names = ["libnvinfer_plugin.so"]
    if lib_name is not None:
        lib_names.append(lib_name)
        # lib_name = "libtrt_plugin_plus.so"

    # 使用ctypes加载动态库
    for lib in lib_names:
        handle = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        if not handle:
            raise RuntimeError("Could not load plugin library. Is " + lib + " on your LD_LIBRARY_PATH?")

    # 初始化TensorRT插件
    trt.init_libnvinfer_plugins(logger, "")

    logger.log(logger.INFO, "[TrtHelper LOG] tensorrt plugin init done!")

    return logger

class TrtNetworkHelper():
    """
    TensorRT网络定义辅助类，用于简化TensorRT网络构建过程
    
    该类提供了多种方法来添加常见的神经网络层，如卷积、线性层、激活函数等，
    并处理TensorRT特有的配置和优化。
    """
    def __init__(self, network, plugin_registry, logger, plugin_data_type):
        """
        初始化TrtNetworkHelper
        
        参数:
            network: TensorRT网络对象
            plugin_registry: TensorRT插件注册表
            logger: TensorRT日志记录器
            plugin_data_type: 插件数据类型
        """
        self.network = network
        self.plugin_registry = plugin_registry
        self.logger = logger

        self.input_num = 0

        self.np_data_type = np.array([plugin_data_type], dtype=np.int32)

    def broadcast_matrix(self, mat: np.array, nb_dims: int):
        """
        广播矩阵到指定的维度数
        
        参数:
            mat: 输入矩阵
            nb_dims: 目标维度数
        
        返回:
            广播后的矩阵
        
        说明:
            该方法将低维矩阵广播到高维，通过在前面添加维度为1的轴实现。
            例如，将形状为(10, 20)的矩阵广播到4维，结果为(1, 1, 10, 20)
        """
        mat_nb_dims = len(mat.shape)
        if mat_nb_dims >= nb_dims:
            raise RuntimeError("broadcast_tensor mat_nb_dims >= nb_dims")

        # 创建新形状，前面的维度为1，后面保留原始维度
        new_shape = np.ones([nb_dims], dtype=np.int32)
        new_shape[-mat_nb_dims:] = mat.shape

        # 重塑矩阵
        new_mat = mat.reshape(new_shape)
        self.logger.log(trt.Logger.INFO, "[Network] broadcast_matrix " + \
                                          str(mat.shape) + " to " + str(new_mat.shape))

        return new_mat

    def set_layer_name(self, layer, name):
        """
        设置TensorRT层的名称并打印输出形状
        
        参数:
            layer: TensorRT层对象
            name: 层名称
        """
        if not layer:
            raise RuntimeError("Could not name")

        # 设置层名称，前缀为层编号
        layer.name = str(self.network.num_layers) + "_" + name
        
        # 打印每个输出的形状
        for i in range(0, layer.num_outputs):
            shape = layer.get_output(i).shape
            self.logger.log(trt.Logger.INFO, "[Network] " + layer.name + ", output[" + str(i) + "] shape= " + str(shape))

        return None

    def check_trt_layer(self, trt_layer):
        """
        检查TensorRT层是否创建成功
        
        参数:
            trt_layer: TensorRT层对象
        """
        if not trt_layer:
            raise RuntimeError("add " + str(trt_layer) + " failed!")

        # 检查每个输出的形状
        for i in range(0, trt_layer.num_outputs):
            shape = trt_layer.get_output(i).shape
            # print(trt.volume(shape))

            # 如果形状是1维，可能存在问题（已注释）
            # if len(shape) is 1:
                # raise RuntimeError("add " + layer.name + " failed!")

    def layer_post_process(self, trt_layer, layer_name, precision):
        """
        层后处理：设置精度、设置层名称并检查层
        
        参数:
            trt_layer: TensorRT层对象
            layer_name: 层名称
            precision: 精度类型
        """
        # 设置层精度
        if precision is not None:
            trt_layer.precision = precision

        # 设置层名称
        self.set_layer_name(trt_layer, layer_name)
        
        # 检查层是否创建成功
        self.check_trt_layer(trt_layer)

    def addInput(self, name, dtype, shape):
        """
        添加网络输入
        
        参数:
            name: 输入名称，如果为None则自动生成
            dtype: 数据类型
            shape: 输入形状
        
        返回:
            创建的TensorRT输入张量
        """
        # 如果名称为None，则自动生成
        if name is None:
            name = "input" + str(self.input_num)

        self.input_num = self.input_num + 1

        # 添加网络输入
        trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
        if not trt_input:
            raise RuntimeError("addInput failed!")

        self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))

        return trt_input

    def markOutput(self, x: trt.ITensor):
        """
        标记张量为网络输出
        
        参数:
            x: 要标记为输出的TensorRT张量
        """
        self.network.mark_output(x)
        self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

    def addConv2d(self, x, weight, bias, out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=None,
                  layer_name=None, precision=None):
        """
        添加2D卷积层
        
        参数:
            x: 输入张量
            weight: 卷积权重
            bias: 卷积偏置，可以为None
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 扩张率
            groups: 分组数
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            卷积层的输出张量
        """
        # 设置层名称
        if layer_name is None:
            layer_name = "nn.Conv2d"
        else:
            layer_name = "nn.Conv2d." + layer_name

        # 转换权重和偏置为TensorRT格式
        weight = trt.Weights(weight)
        bias = trt.Weights(bias) if bias is not None else None

        # 添加卷积层
        trt_layer = self.network.add_convolution_nd(
            x, num_output_maps=out_channels,
            kernel_shape=kernel_size,
            kernel=weight, bias=bias)

        # 设置卷积参数
        if stride is not None:
            trt_layer.stride = stride
        if padding is not None:
            trt_layer.padding = padding
        if dilation is not None:
            trt_layer.dilation = dilation
        if groups is not None:
            trt_layer.num_groups = groups

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addDumpTensor(self, x: trt.ITensor, layer_name: str = None):
        """
        添加张量转储插件，用于调试
        
        参数:
            x: 输入张量
            layer_name: 层名称
        
        返回:
            插件的输出张量
        """
        # 获取插件创建器
        plg_creator = self.plugin_registry.get_plugin_creator("DumpTensorPluginDynamic", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find DumpTensorPluginDynamic")

        # 设置层名称
        if layer_name is None:
            layer_name = "DumpTensorPlugin"
        else:
            layer_name = "DumpTensorPlugin." + layer_name

        # 创建插件字段集合
        # data_type = trt.PluginField("data_type", np.array([data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        # pfc = trt.PluginFieldCollection([data_type])
        pfc = trt.PluginFieldCollection([])
        
        # 创建插件
        plugin = plg_creator.create_plugin(layer_name, pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin DumpTensorPluginDynamic")

        # 添加插件层
        layer = self.network.add_plugin_v2([x], plugin)

        # 层后处理
        self.layer_post_process(layer, layer_name, None)

        x = layer.get_output(0)
        return x

    def addEmbedding(self, indices, weight, layer_name=None, precision=None):
        """
        添加嵌入层
        
        参数:
            indices: 索引张量
            weight: 嵌入权重
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            嵌入层的输出张量
        
        说明:
            嵌入层通过常量层和gather操作实现，相当于查表操作
        """
        # 创建常量层存储嵌入表
        constant_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        
        # 使用gather操作查找嵌入向量
        gather_layer = self.network.add_gather(constant_layer.get_output(0),
                                               indices, axis=0)

        # 设置层名称
        if layer_name is None:
            layer_name = "nn.Embedding"
        else:
            layer_name = "nn.Embedding." + layer_name

        # 层后处理
        self.layer_post_process(gather_layer, layer_name, precision)

        return gather_layer.get_output(0)

    def addGELU(self, x, layer_name=None, precision=None):
        """
        添加GELU激活函数层
        
        参数:
            x: 输入张量
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            GELU层的输出张量
        
        说明:
            GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            由于TensorRT没有内置GELU，这里通过基本操作组合实现
        """
        # 创建常量
        POW = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        MULTIPLY = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        
        # 计算x^3
        X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
        X_pow_t = X_pow.get_output(0)
        
        # 计算0.044715 * x^3
        X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        
        # 计算x + 0.044715 * x^3
        X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        
        # 计算sqrt(2/pi) * (x + 0.044715 * x^3)
        X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        
        # 计算tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
        X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        
        # 计算1 + tanh(...)
        X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        
        # 计算0.5 * (1 + tanh(...))
        CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        
        # 计算x * 0.5 * (1 + tanh(...))
        gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

        # 设置层名称
        if layer_name is None:
            layer_name = "nn.GELU"
        else:
            layer_name = "nn.GELU." + layer_name

        # 层后处理
        self.layer_post_process(gelu_layer, layer_name, precision)

        return gelu_layer.get_output(0)

    def addSigmoid(self, x, layer_name=None, precision=None):
        """
        添加Sigmoid激活函数层
        
        参数:
            x: 输入张量
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            Sigmoid层的输出张量
        """
        # 添加Sigmoid激活层
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.SIGMOID)

        # 设置层名称
        if layer_name is None:
            layer_name = "nn.Sigmoid"

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addLayerNorm(self, x, weight, bias, layer_name=None, precision=None):
        """
        添加层归一化
        
        参数:
            x: 输入张量
            weight: 缩放参数
            bias: 偏置参数
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            层归一化的输出张量
        
        说明:
            使用TensorRT的LayerNorm插件实现层归一化
        """
        # 获取插件创建器
        plg_creator = self.plugin_registry.get_plugin_creator("LayerNorm", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find LayerNorm")

        # 设置层归一化参数
        dim = 768
        eps = 0.00001
        gamma = weight
        beta = bias
        
        # 创建插件字段集合
        # data_type = trt.PluginField("data_type", self.np_data_type, trt.PluginFieldType.INT32)
        # dim = trt.PluginField("dim", np.array([dim], dtype=np.int32), trt.PluginFieldType.INT32)
        # eps = trt.PluginField("eps", np.array([eps], dtype=np.float32), trt.PluginFieldType.FLOAT32)
        # gamma_w = trt.PluginField("gamma", gamma.detach().numpy(), trt.PluginFieldType.FLOAT32)
        # beta_w = trt.PluginField("beta", beta.detach().numpy(), trt.PluginFieldType.FLOAT32)
        pfc = trt.PluginFieldCollection([])
        
        # 创建插件
        plugin = plg_creator.create_plugin("LayerNormPluginDynamic", pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin LayerNormPluginDynamic")

        # 添加gamma和beta作为常量输入
        gamma_w = self.addConstant(gamma)
        beta_w = self.addConstant(beta)
        
        # 添加插件层
        trt_layer = self.network.add_plugin_v2([x, gamma_w, beta_w], plugin)

        # 设置层名称
        if layer_name is None:
            layer_name = "nn.LayerNorm"

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addLinear(self, x, weight, bias=None, layer_name=None, precision=None):
        """
        添加线性层
        
        参数:
            x: 输入张量
            weight: 权重矩阵
            bias: 偏置向量，可以为None
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            线性层的输出张量
        
        说明:
            线性层通过矩阵乘法和加法实现: y = x * W + b
        """
        # 广播权重矩阵到输入张量的维度
        weight = self.broadcast_matrix(weight, len(x.shape))

        # 创建权重常量层
        weight_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        weight = weight_layer.get_output(0)
        
        # 添加矩阵乘法层
        # trt_layer = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE)
        trt_layer = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.NONE)
        x = trt_layer.get_output(0)

        # 设置层名称
        if layer_name is None:
            layer_name = "Linear"
        else:
            layer_name = "Linear." + layer_name

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        # 如果有偏置，添加偏置
        if bias is not None:
            # 广播偏置向量到输出张量的维度
            bias = self.broadcast_matrix(bias, len(x.shape))
            
            # 创建偏置常量层
            bias_layer = self.network.add_constant(bias.shape, trt.Weights(bias))
            bias = bias_layer.get_output(0)
            
            # 添加元素级加法层
            trt_layer = self.network.add_elementwise(x, bias, trt.ElementWiseOperation.SUM)
            x = trt_layer.get_output(0)

            # 设置偏置层名称
            if layer_name is None:
                layer_name = "Linear.bias"
            else:
                layer_name = "Linear.bias." + layer_name

        return x

    def addReshape(self, x, reshape_dims, layer_name=None, precision=None):
        """
        添加重塑层
        
        参数:
            x: 输入张量
            reshape_dims: 新的形状维度
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            重塑后的张量
        
        说明:
            在TensorRT中，重塑操作通过Shuffle层实现
        """
        # 创建Shuffle层
        trt_layer = self.network.add_shuffle(x)
        
        # 设置重塑维度
        trt_layer.reshape_dims = reshape_dims

        # 设置层名称
        if layer_name is None:
            layer_name = "torch.reshape"
        else:
            layer_name = "torch.reshape." + layer_name

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

    def addSlice(self, x, start_dim, shape_dim, stride_dim, layer_name=None, precision=None):
        """
        添加切片层
        
        参数:
            x: 输入张量
            start_dim: 起始索引
            shape_dim: 切片形状
            stride_dim: 步长
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            切片后的张量
        """
        # 创建切片层
        trt_layer = self.network.add_slice(x, start_dim, shape_dim, stride_dim)

        # 设置层名称
        if layer_name is None:
            layer_name = "tensor.slice"
        else:
            layer_name = "tensor.slice." + layer_name

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

    def addReLU(self, x, layer_name=None, precision=None):
        """
        添加ReLU激活函数层
        
        参数:
            x: 输入张量
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            ReLU层的输出张量
        """
        # 创建ReLU激活层
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

        # 设置层名称
        if layer_name is None:
            layer_name = "nn.ReLU"

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addTanh(self, x, layer_name=None, precision=None):
        """
        添加Tanh激活函数层
        
        参数:
            x: 输入张量
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            Tanh层的输出张量
        """
        # 创建Tanh激活层
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.TANH)

        # 设置层名称
        if layer_name is None:
            layer_name = "nn.Tanh"

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    ################## elementwise op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
        """
        添加对数运算层
        
        参数:
            x: 输入张量
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            对数运算后的张量
        """
        # 创建一元运算层，计算自然对数
        trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        
        # 设置层名称
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addScale(
            self,
            x: trt.ITensor,
            scale: float,
            layer_name: str = None,
            precision: trt.DataType = None
    ) -> trt.ITensor:
        """
        添加缩放层，将张量乘以标量
        
        参数:
            x: 输入张量
            scale: 缩放因子
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            缩放后的张量
        
        说明:
            TensorRT的Scale层要求输入至少为4维，因此对于3维输入需要先扩展维度，
            计算后再恢复原始维度。
        """
        input_len = len(x.shape)
        if input_len < 3:
            raise RuntimeError("input_len < 3 not support now! ")

        if layer_name is None:
            layer_name = "Scale"

        # TensorRT的Scale层要求输入维度至少为4，对于3维输入需要扩展维度
        if input_len is 3:
            trt_layer = self.network.add_shuffle(x)
            trt_layer.reshape_dims = (0, 0, 0, 1)  # 添加第4维
            self.layer_post_process(trt_layer, layer_name+".3dto4d", precision)
            x = trt_layer.get_output(0)

        # 创建缩放层
        np_scale = trt.Weights(np.array([scale], dtype=np.float32))
        trt_layer = self.network.add_scale(x, mode=trt.ScaleMode.UNIFORM,
                                      shift=None, scale=np_scale, power=None)
        self.layer_post_process(trt_layer, layer_name, precision)
        x = trt_layer.get_output(0)

        # 如果原始输入是3维，需要恢复原始维度
        if input_len is 3:
            trt_layer = self.network.add_shuffle(x)
            trt_layer.reshape_dims = (0, 0, 0)  # 恢复为3维
            self.layer_post_process(trt_layer, layer_name+".4dto3d", precision)
            x = trt_layer.get_output(0)

        return x

    def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor:
        """
        添加Softmax层
        
        参数:
            x: 输入张量
            dim: 应用softmax的维度，默认为-1（最后一个维度）
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            Softmax层的输出张量
        
        说明:
            TensorRT的Softmax层通过位掩码指定维度，需要将dim转换为对应的位掩码
        """
        # 创建Softmax层
        trt_layer = self.network.add_softmax(x)

        # 检查输入维度
        input_len = len(x.shape)
        if input_len < 2:
            raise RuntimeError("softmax input_len must >= 2")

        # 处理负索引
        if dim < 0:
            dim = input_len + dim

        # TensorRT的Softmax层通过位掩码指定维度
        # 例如，对于dim=2，位掩码为1<<2 = 4（二进制100）
        trt_layer.axes = 1 << dim

        # 设置层名称
        layer_name_prefix = "nn.Softmax[dim=" + str(dim) + "]"
        if layer_name is None:
            layer_name = layer_name_prefix
        else:
            layer_name = layer_name_prefix + "." + layer_name

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addCat(self, inputs = [], dim = 0, layer_name=None, precision=None):
        """
        添加连接层，沿指定维度连接多个张量
        
        参数:
            inputs: 输入张量列表
            dim: 连接的维度，默认为0
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            连接后的张量
        """
        # 至少需要两个输入
        assert len(inputs) > 1

        # 创建连接层
        trt_layer = self.network.add_concatenation(inputs)

        # 处理负索引
        if dim == -1:
            dim = len(inputs[0].shape) - 1

        # 设置连接轴
        trt_layer.axis = dim

        # 设置层名称
        if layer_name is None:
            layer_name = "torch.cat"
        else:
            layer_name = "torch.cat." + layer_name

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addMatMul(self, a, b, trans_a=False, trans_b=False, layer_name=None, precision=None):
        """
        添加矩阵乘法层
        
        参数:
            a: 第一个输入张量
            b: 第二个输入张量
            trans_a: 是否转置第一个输入
            trans_b: 是否转置第二个输入
            layer_name: 层名称
            precision: 精度类型
        
        返回:
            矩阵乘法的输出张量
        """
        # 设置层名称
        if layer_name is None:
            layer_name = "matrix_multiply"
        else:
            layer_name = "matrix_multiply." + layer_name

        # 设置矩阵操作
        op_a = trt.MatrixOperation.NONE
        if trans_a is True:
            op_a = trt.MatrixOperation.TRANSPOSE

        op_b = trt.MatrixOperation.NONE
        if trans_b is True:
            op_b = trt.MatrixOperation.TRANSPOSE

        # 创建矩阵乘法层
        trt_layer = self.network.add_matrix_multiply(a, op_a, b, op_b)
        
        # 层后处理
        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addConstant(self, w, layer_name: Optional[str] = None) -> trt.ITensor:
        """
        添加常量层
        
        参数:
            w: 常量值（NumPy数组）
            layer_name: 层名称
        
        返回:
            常量张量
        """
        # 创建常量层
        trt_layer = self.network.add_constant(w.shape, w)

        # 设置层名称
        if layer_name is None:
            layer_name = "trt.Constant"
        else:
            layer_name = "trt.Constant." + layer_name

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, None)
        
        x = trt_layer.get_output(0)
        return x

    def addShuffle(
        self,
        x: trt.ITensor,
        first_transpose: trt.Permutation,
        reshape_dims: trt.Dims,
        second_transpose: trt.Permutation,
        layer_name: Optional[str] = None
    ) -> trt.ITensor:
        """
        添加Shuffle层，可以执行转置、重塑和第二次转置操作
        
        参数:
            x: 输入张量
            first_transpose: 第一次转置的轴顺序，可以为None
            reshape_dims: 重塑的目标维度，可以为None
            second_transpose: 第二次转置的轴顺序，可以为None
            layer_name: 层名称
        
        返回:
            Shuffle层的输出张量
        
        说明:
            TensorRT的Shuffle层可以组合执行转置、重塑和第二次转置操作，
            这三个操作都是可选的。
        """
        # 创建Shuffle层
        trt_layer = self.network.add_shuffle(x)

        # 设置第一次转置
        if first_transpose is not None:
            trt_layer.first_transpose = first_transpose

        # 设置重塑维度
        if reshape_dims is not None:
            trt_layer.reshape_dims = reshape_dims

        # 设置第二次转置
        if second_transpose is not None:
            trt_layer.second_transpose = second_transpose

        # 设置层名称
        if layer_name is None:
            layer_name = "shuffle"
        else:
            layer_name = "shuffle." + layer_name

        # 层后处理
        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

