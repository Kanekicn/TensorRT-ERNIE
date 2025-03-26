#ifndef TRT_HEPLER_
#define TRT_HEPLER_

#include <sys/time.h>
#include <vector>
#include <string>
#include <string.h>
#include <memory>
#include <cassert>
#include <iostream>

#include "NvInfer.h"

#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#  define CUDA_CHECK(status)                                                      \
    if (status != cudaSuccess) {                                                  \
      std::cout << "Cuda failure! Error=" << cudaGetErrorString(status) << std::endl; \
    }
#endif

// 定义一个模板类cuda_shared_ptr, 用于管理cuda内存
template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

// 定义一个模板类InferDeleter, 用于管理trt内存
struct InferDeleter {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

// 定义一个模板函数makeShared, 用于创建trt对象
template <typename T>
std::shared_ptr<T> makeShared(T *obj) {
  if (!obj) {
    throw std::runtime_error("Failed to create object");
  }
  return std::shared_ptr<T>(obj, InferDeleter());
}

// 定义一个模板类CudaDeleter, 用于管理cuda内存
template <typename T>
struct CudaDeleter {
  void operator()(T* buf) {
    if (buf) cudaFree(buf);
  }
};

// 定义一个模板函数make_cuda_shared, 用于创建cuda对象
template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem) {
  ptr.reset(static_cast<T*>(cudaMem), CudaDeleter<T>());
}

// 定义一个模板类TrtDestroyer, 用于管理trt内存
struct TrtDestroyer {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) obj->destroy();
  }
};

// 定义一个模板类TrtUniquePtr, 用于管理trt内存
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer>;

// 定义一个模板函数MakeUnique, 用于创建trt对象
template <typename T>
inline TrtUniquePtr<T> MakeUnique(T *t) {
  return TrtUniquePtr<T>{t};
}

// 定义一个模板函数MakeShared, 用于创建trt对象
template <typename T>
inline std::shared_ptr<T> MakeShared(T *t) {
  return std::shared_ptr<T>(t, TrtDestroyer());
}

// 定义一个结构体sample, 用于存储样本数据, 包含12个字段
struct sample{
    std::string qid;
    std::string label;
    std::vector<int> shape_info_0;
    std::vector<int> i0;
    std::vector<int> shape_info_1;
    std::vector<int> i1;
    std::vector<int> shape_info_2;
    std::vector<int> i2;
    std::vector<int> shape_info_3;
    std::vector<float> i3;
    std::vector<int> shape_info_4;
    std::vector<int> i4;
    std::vector<int> shape_info_5;
    std::vector<int> i5;
    std::vector<int> shape_info_6;
    std::vector<int> i6;
    std::vector<int> shape_info_7;
    std::vector<int> i7;
    std::vector<int> shape_info_8;
    std::vector<int> i8;
    std::vector<int> shape_info_9;
    std::vector<int> i9;
    std::vector<int> shape_info_10;
    std::vector<int> i10;
    std::vector<int> shape_info_11;
    std::vector<int> i11;
    std::vector<float> out_data;
    uint64_t timestamp;
};


// BEGIN_LIB_NAMESPACE {

// Undef levels to support LOG(LEVEL)


/*
 * 定义一个类TrtLogger, 用于管理trt日志
 * 继承自nvinfer1::ILogger
 * 重载log函数, 用于打印日志
 */
class TrtLogger : public nvinfer1::ILogger {
  using Severity = nvinfer1::ILogger::Severity;

 public:
  explicit TrtLogger(Severity level = Severity::kINFO);

  ~TrtLogger() = default;

  nvinfer1::ILogger& getTRTLogger();

  void log(Severity severity, const char* msg) noexcept override;

 private:
  Severity level_;
};

/*
 * 定义一个类TrtHepler, 用于管理trt推理
 * 包含一个构造函数, 一个推理函数, 一个析构函数
 * 构造函数: 初始化trt_helper, 读取模型参数文件, 设置设备ID
 * 推理函数: 推理函数, 输入sample结构体, 返回推理结果
 * 析构函数: 析构函数, 释放trt资源
 * 成员函数: 
 * 1. 构造函数: 初始化trt_helper, 读取模型参数文件, 设置设备ID
 * 2. 推理函数: 推理函数, 输入sample结构体, 返回推理结果
 * 3. 析构函数: 析构函数, 释放trt资源
 * 成员变量: 
 * 1. 设备ID: int _dev_id, 用于设置设备ID
 * 2. 模型参数文件: std::string _model_param, 用于存储模型参数文件路径
 * 3. 引擎: std::shared_ptr<nvinfer1::ICudaEngine> engine_, 用于存储引擎
 * 4. 上下文: std::shared_ptr<nvinfer1::IExecutionContext> context_, 用于存储上下文
 * 5. cuda流: cudaStream_t cuda_stream_, 用于存储cuda流
 * 6. 输入维度: std::vector<nvinfer1::Dims> inputs_dims_, 用于存储输入维度
 * 7. 设备绑定: std::vector<void*> device_bindings_, 用于存储设备绑定
 */
class TrtHepler {
 public:
  TrtHepler(std::string model_param, int dev_id);

  int Forward(sample& s);

  ~TrtHepler();

 private:
  int _dev_id;
  // NS_PROTO::ModelParam *_model_param_ptr;
  std::string _model_param;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t cuda_stream_;

  // The all dims of all inputs.
  std::vector<nvinfer1::Dims> inputs_dims_;
  std::vector<void*> device_bindings_;
};


class TrtEngine{
public:
    TrtEngine(std::string model_param, int dev_id);
    ~TrtEngine(){};

    int dev_id_;
    std::string _model_param;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;

    TrtLogger trt_logger;
};

class TrtContext{
public:
    TrtContext(TrtEngine* trt_engine, int profile_idx);
    int Forward(struct sample& s);
    ~TrtContext();
    int CaptureCudaGraph();

    int dev_id_;
    std::string _model_param;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t cuda_stream_;

    std::vector<nvinfer1::Dims> inputs_dims_;
    std::vector<char*> device_bindings_;
    std::vector<char*> host_bindings_;
    static std::vector<char*> s_device_bindings_;

    char* h_buffer_;
    char* d_buffer_;

    int max_batch_;
    int max_seq_len_;
    int start_binding_idx_;
    int profile_idx_;

    int align_input_bytes_;
    int align_aside_intput_bytes_;
    int whole_bytes_;

    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
    bool graph_created_ = false;
};


// } // BEGIN_LIB_NAMESPACE

#endif // TRT_HEPLER_

