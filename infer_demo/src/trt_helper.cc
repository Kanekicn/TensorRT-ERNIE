#include "trt_helper.h"

#include <string>
#include <fstream>
#include <sstream>

#include "NvInferPlugin.h"

using namespace std;

// BEGIN_LIB_NAMESPACE {

/*
 * 将cpu数据转换为cuda数据
 * @param shape 输入维度
 * @param data_ptr 输入数据指针
 * @return cuda数据指针
 */
cuda_shared_ptr<void> CpuToDevice(const std::vector<int>& shape, int* data_ptr) {
  void *d_ptr;
  auto cpu_ptr = static_cast<void *>(data_ptr);
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  auto ret = cudaMalloc(&d_ptr, data_size * sizeof(int));
  //printf("int memory\n");
  if (ret) printf("memory error\n");
  ret = cudaMemcpy(d_ptr, cpu_ptr, data_size * sizeof(int), cudaMemcpyHostToDevice);
  if (ret) printf("memory error\n");
  cuda_shared_ptr<void> cuda_ptr;
  make_cuda_shared(cuda_ptr, d_ptr);
  return cuda_ptr;
}

/*
 * 将cpu数据转换为cuda数据
 * @param shape 输入维度
 * @param data_ptr 输入数据指针
 * @return cuda数据指针
 */
cuda_shared_ptr<void> CpuToDevice(const std::vector<int>& shape, float* data_ptr) {
  void *d_ptr;
  auto cpu_ptr = static_cast<void *>(data_ptr);
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  auto ret = cudaMalloc(&d_ptr, data_size * sizeof(float));
  //printf("float memory\n");
  if (ret) printf("memory error\n");
  ret = cudaMemcpy(d_ptr, cpu_ptr, data_size * sizeof(float), cudaMemcpyHostToDevice);
  if (ret) printf("memory error\n");
  cuda_shared_ptr<void> cuda_ptr;
  make_cuda_shared(cuda_ptr, d_ptr);
  return cuda_ptr;
}

/*
 * 将cuda数据转换为cpu数据
 * @param shape 输入维度
 * @param cuda_ptr cuda数据指针
 * @param data_ptr 输出数据指针
 */
void DeviceToCpu(const std::vector<int>& shape, cuda_shared_ptr<void> cuda_ptr, float* data_ptr) {
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  if (data_size == 0) {
    std::cout << "data_size == 0" << std::endl;
    assert(0);
  }
  auto d_ptr = static_cast<void *>(data_ptr);
  auto ret = cudaMemcpy(d_ptr, cuda_ptr.get(), data_size * sizeof(float), cudaMemcpyDeviceToHost);
  printf("copy back\n");
  if (ret) printf("memory error\n");
}

/*
 * 定义一个类TrtLogger, 用于管理trt日志
 * 继承自nvinfer1::ILogger
 * 重载log函数, 用于打印日志
 */ 
TrtLogger::TrtLogger(nvinfer1::ILogger::Severity level) : level_(level) {}

nvinfer1::ILogger& TrtLogger::getTRTLogger() { return *this; }

// trt logger
void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity > level_) {
    return;
  }

  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kERROR:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kWARNING:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kINFO:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kVERBOSE:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
  }
}

/*
 * TrtHepler构造函数：
 * 1. 读取模型参数文件, 设置设备ID
 * 2. 创建cuda流
 * 3. 初始化trt_logger
 * 4. 反序列化模型参数文件, 创建引擎
 * 5. 创建上下文
 * 6. 设置优化配置文件
 * @param model_param 模型参数文件路径
 * @param dev_id 设备ID
 */
TrtHepler::TrtHepler(std::string model_param, int dev_id)
    : _dev_id(dev_id), _model_param(model_param) {
  { // read model, deserializeCudaEngine and createExecutionContext
    // 1. 读取模型参数文件
    std::ifstream t(_model_param);  
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());
    // 2. 设置设备ID
    CUDA_CHECK(cudaSetDevice(_dev_id));
    // 3. 创建cuda流
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    // 4. 初始化trt_logger
    TrtLogger trt_logger;
    // 5. 初始化插件
    initLibNvInferPlugins(&trt_logger.getTRTLogger(), "");
    // 6. 反序列化模型参数文件, 创建引擎
    auto runtime = MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
    auto e = runtime->deserializeCudaEngine((void*)contents.c_str(),
                                            contents.size(), nullptr);
    engine_ = MakeShared(e);
    // 7. 创建上下文
    context_ = MakeShared(engine_->createExecutionContext());
    // 8. 设置优化配置文件
    context_->setOptimizationProfile(0);
  }

}

/*
 * TrtHepler推理函数：
 * 1. 设置设备ID
 * 2. 将cpu数据转换为cuda数据
 * 3. 设置输入维度: 输入维度为13个, 分别为rc_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, tmp6_tensor, tmp7_tensor, tmp8_tensor, tmp9_tensor, tmp10_tensor, tmp11_tensor, tmp12_tensor, tmp13_tensor, cuda_out_ptr
 * 4. 设置设备绑定: 将输入维度设置为设备绑定
 * 5. 推理: 推理
 * 6. 将cuda数据转换为cpu数据: 将cuda数据转换为cpu数据
 * 7. 返回推理结果: 返回推理结果
 * @param s sample结构体
 * @return 推理结果
 */
int TrtHepler::Forward(sample& s) {
  // 1. 设置设备ID
  cudaSetDevice(_dev_id);
  // 2. 将cpu数据转换为cuda数据
  auto rc_ids_tensor = CpuToDevice(s.shape_info_0, s.i0.data());
  auto sent_ids_tensor = CpuToDevice(s.shape_info_1, s.i1.data());
  auto pos_ids_tensor = CpuToDevice(s.shape_info_2, s.i2.data());
  auto input_mask_tensor = CpuToDevice(s.shape_info_3, s.i3.data());
  auto tmp6_tensor = CpuToDevice(s.shape_info_4, s.i4.data());
  auto tmp7_tensor = CpuToDevice(s.shape_info_5, s.i5.data());
  auto tmp8_tensor = CpuToDevice(s.shape_info_6, s.i6.data());
  auto tmp9_tensor = CpuToDevice(s.shape_info_7, s.i7.data());
  auto tmp10_tensor = CpuToDevice(s.shape_info_8, s.i8.data());
  auto tmp11_tensor = CpuToDevice(s.shape_info_9, s.i9.data());
  auto tmp12_tensor = CpuToDevice(s.shape_info_10, s.i10.data());
  auto tmp13_tensor = CpuToDevice(s.shape_info_11, s.i11.data());
  // 3. 设置输入维度: 输入维度为13个, 分别为rc_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, tmp6_tensor, tmp7_tensor, tmp8_tensor, tmp9_tensor, tmp10_tensor, tmp11_tensor, tmp12_tensor, tmp13_tensor, cuda_out_ptr
  void* out_ptr;
  auto ret_ = cudaMalloc(&out_ptr, s.shape_info_0[0] * sizeof(float));  // -1 * 1
  cuda_shared_ptr<void> cuda_out_ptr;
  make_cuda_shared(cuda_out_ptr, out_ptr);

  cudaEvent_t start, stop;
  float elapsed_time = 0.0;

  // 4. 设置设备绑定: 将输入维度设置为设备绑定
  int binding_idx = 0;  // 输入维度索引
  //std::vector<std::vector<int>> input_dims = {s.shape_info_0, s.shape_info_1, s.shape_info_2, s.shape_info_3,
                                              //s.shape_info_4, s.shape_info_5, s.shape_info_6, s.shape_info_7,
                                              //s.shape_info_8, s.shape_info_9, s.shape_info_10, s.shape_info_11};
  std::vector<std::vector<int>> input_dims = {s.shape_info_0, s.shape_info_1, s.shape_info_2, s.shape_info_3,
                                              s.shape_info_4, s.shape_info_5, s.shape_info_6, s.shape_info_7,
                                              s.shape_info_8, s.shape_info_9, s.shape_info_10, s.shape_info_11};
  // 5. 设置设备绑定: 将输入维度设置为设备绑定
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::vector<int> dims_vec = input_dims[i];  // 输入维度向量
    nvinfer1::Dims trt_dims;  // trt维度
    trt_dims.nbDims = static_cast<int>(dims_vec.size());  // 维度数量
    memcpy(trt_dims.d, dims_vec.data(), sizeof(int) * trt_dims.nbDims);  // 复制维度信息
    context_->setBindingDimensions(binding_idx, trt_dims);  // 设置输入维度
    binding_idx ++;  // 输入维度索引加1
  }

  // 6. 检查输入维度是否指定
  if (!context_->allInputDimensionsSpecified()) { 
    //gLogFatal << "context_->allInputDimensionsSpecified() error";
    std::cout << ("context_->allInputDimensionsSpecified() error") << std::endl;
    assert(0);
  }

  // 7. 设置输入维度
  void *device_bindings[13] = {rc_ids_tensor.get(), sent_ids_tensor.get(), pos_ids_tensor.get(),
                               input_mask_tensor.get(),
                               tmp6_tensor.get(), tmp7_tensor.get(),
                               tmp8_tensor.get(), tmp9_tensor.get(), tmp10_tensor.get(),
                               tmp11_tensor.get(), tmp12_tensor.get(), tmp13_tensor.get(),
                               cuda_out_ptr.get()};  // 设备绑定
  // 8. 推理
  bool ret = context_->enqueueV2(device_bindings, cuda_stream_, nullptr); 
  if (!ret) {
    std::cout << ("context_->enqueueV2 failed!") << std::endl;
    return -100;
  }
  // 9. 将cuda数据转换为cpu数据
  cudaMemcpy(s.out_data.data(), cuda_out_ptr.get(), s.shape_info_0[0] * sizeof(float), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(cuda_stream_); 
  struct timeval tv;
  gettimeofday(&tv, NULL);
  s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
  // 10. 返回推理结果
  return 0;
}

TrtHepler::~TrtHepler() {
  CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
}

// } // BEGIN_LIB_NAMESPACE

