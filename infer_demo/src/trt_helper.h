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
#define CUDA_CHECK(status)                                                      \
    do {                                                                        \
        cudaError_t err = status;                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
        }                                                                       \
    } while (0)
#endif


// 定义一个模板类cuda_shared_ptr, 用于管理cuda内存
template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

/*
 * InferDeleter结构体，用于管理TensorRT对象的资源释放
 * 当智能指针超出作用域时，会自动调用指定的destroy方法
 */
struct InferDeleter {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

/*
 * 创建智能指针管理TensorRT对象
 * @param obj TensorRT对象指针
 * @return 包含该对象的智能指针
 * @throws 如果对象创建失败将抛出运行时异常
 */
template <typename T>
std::shared_ptr<T> makeShared(T *obj) {
  if (!obj) {
    throw std::runtime_error("Failed to create object");
  }
  return std::shared_ptr<T>(obj, InferDeleter());
}

/*
 * CudaDeleter结构体，用于管理CUDA内存的释放
 * 当智能指针超出作用域时，会自动调用cudaFree释放CUDA内存
 */template <typename T>
struct CudaDeleter {
  void operator()(T* buf) {
    if (buf) cudaFree(buf);
  }
};

/*
 * 创建管理CUDA内存的智能指针
 * @param ptr 要初始化的智能指针引用
 * @param cudaMem CUDA内存指针
 */
template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem) {
  ptr.reset(static_cast<T*>(cudaMem), CudaDeleter<T>());
}

/*
 * TrtDestroyer结构体，类似InferDeleter，用于TrtUniquePtr
 */
struct TrtDestroyer {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) obj->destroy();
  }
};

/*
 * 创建TensorRT对象的unique_ptr智能指针
 * @param t TensorRT对象指针
 * @return 包含该对象的unique_ptr智能指针
 */
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer>;

/*
 * sample结构体，用于存储模型输入输出数据
 * 包含：
 * - qid和label：样本标识符和标签
 * - 多个形状信息和输入向量（i0-i11）
 * - 输出数据向量
 * - 时间戳
 */
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
 * TrtLogger类，用于管理TensorRT日志
 * 继承自nvinfer1::ILogger，重写log方法来处理不同级别的日志消息
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
    int _dev_id;                                          // CUDA设备ID
    std::string _model_param;                             // 模型参数文件路径
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;       // TensorRT引擎
    std::shared_ptr<nvinfer1::IExecutionContext> context_; // 执行上下文
    cudaStream_t cuda_stream_;                            // CUDA流

    std::vector<nvinfer1::Dims> inputs_dims_;             // 所有输入的维度信息
    std::vector<void*> device_bindings_;                  // 设备内存绑定
};


class TrtEngine{
public:
    TrtEngine(std::string model_param, int dev_id);
    ~TrtEngine(){};

    int dev_id_;                                      // CUDA设备ID
    std::string _model_param;                         // 模型参数文件路径
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;   // TensorRT引擎
    TrtLogger trt_logger;                             // TensorRT日志记录器
};

/*
 * TrtContext类，负责创建执行上下文并管理推理过程
 * 支持CUDA图优化，可以加速连续的相似推理
 */
class TrtContext{
public:
    TrtContext(TrtEngine* trt_engine, int profile_idx);
    int Forward(struct sample& s);
    ~TrtContext();
    int CaptureCudaGraph();

    int dev_id_;                                          // CUDA设备ID
    std::string _model_param;                             // 模型参数文件路径
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;       // TensorRT引擎
    std::shared_ptr<nvinfer1::IExecutionContext> context_; // 执行上下文
    cudaStream_t cuda_stream_;                            // CUDA流

    std::vector<nvinfer1::Dims> inputs_dims_;             // 所有输入的维度信息
    std::vector<char*> device_bindings_;                  // 设备内存绑定
    std::vector<char*> host_bindings_;                    // 主机内存绑定
    static std::vector<char*> s_device_bindings_;         // 静态设备内存绑定

    char* h_buffer_;                                      // 主机内存缓冲区
    char* d_buffer_;                                      // 设备内存缓冲区

    int max_batch_;                                       // 最大批量大小
    int max_seq_len_;                                     // 最大序列长度
    int start_binding_idx_;                               // 绑定索引起始位置
    int profile_idx_;                                     // 优化配置文件索引

    int align_input_bytes_;                               // 对齐的输入字节数
    int align_aside_intput_bytes_;                        // 对齐的辅助输入字节数
    int whole_bytes_;                                     // 总字节数

    cudaGraph_t graph_;                                   // CUDA图
    cudaGraphExec_t instance_;                            // CUDA图实例
    bool graph_created_ = false;                          // 图是否已创建标志
};


// } // BEGIN_LIB_NAMESPACE

#endif // TRT_HEPLER_

