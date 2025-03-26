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

/*
 * TrtEngine构造函数：
 * 1. 读取模型参数文件
 * 2. 设置设备ID
 * 3. 初始化插件
 * 4. 反序列化模型参数文件，创建引擎
 * 5. 输出引擎IO张量数量
 * @param model_param 模型参数文件路径
 * @param dev_id 设备ID
 */
TrtEngine::TrtEngine(std::string model_param, int dev_id) : dev_id_(dev_id), _model_param(model_param) {
    std::ifstream t(_model_param);
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());

    CUDA_CHECK(cudaSetDevice(dev_id));

    initLibNvInferPlugins(&trt_logger.getTRTLogger(), "");
    auto runtime = MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
    auto e = runtime->deserializeCudaEngine((void*)contents.c_str(), contents.size(), nullptr);
    engine_ = MakeShared(e);

    cout << "getNbIOTensors: " << engine_->getNbIOTensors() << endl;
}

/*
 * 计算向上取整的除法
 * @param a 被除数
 * @param b 除数
 * @return 向上取整的结果
 */
constexpr size_t kAlignment = 128;
constexpr int ceildiv(int a, int b){
    return (a + b - 1) / b;
}

/*
 * 将数值按指定值对齐
 * @param a 需要对齐的数值
 * @param b 对齐的基准值，默认为kAlignment
 * @return 对齐后的数值
 */
constexpr int AlignTo(int a, int b = kAlignment){
    return ceildiv(a, b) * b;
}

/*
 * TrtContext构造函数：
 * 1. 设置设备ID和profile索引
 * 2. 创建CUDA流
 * 3. 创建执行上下文
 * 4. 设置优化配置文件
 * 5. 计算绑定索引起始位置
 * 6. 获取配置文件的最大维度
 * 7. 分配设备和主机缓冲区
 * 8. 设置绑定指针
 * 9. 设置输入维度
 * @param trt_engine TensorRT引擎对象
 * @param profile_idx 配置文件索引
 */
std::vector<char*> TrtContext::s_device_bindings_;

/*
 * TrtContext推理函数：
 * 1. 设置设备ID
 * 2. 将CPU数据复制到主机缓冲区
 * 3. 将主机缓冲区数据复制到设备缓冲区
 * 4. 执行推理（使用CUDA图或直接执行）
 * 5. 将设备缓冲区中的结果复制回主机
 * 6. 记录时间戳
 * @param s sample结构体
 * @return 推理结果状态码（0表示成功）
 */
TrtContext::TrtContext(TrtEngine *trt_engine, int profile_idx) {
    profile_idx_ = profile_idx;
    engine_ = trt_engine->engine_;
    dev_id_ = trt_engine->dev_id_;
    CUDA_CHECK(cudaSetDevice(dev_id_));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

    context_ = MakeShared(engine_->createExecutionContext());
    context_->setOptimizationProfile(profile_idx);

    start_binding_idx_ = profile_idx * engine_->getNbBindings() / engine_->getNbOptimizationProfiles();
    auto max_profile = engine_->getProfileDimensions(
            start_binding_idx_, profile_idx, nvinfer1::OptProfileSelector::kMAX);

    // 4 input: [B, S], 8 aside inputs and 1 output: [B]
    max_batch_ = max_profile.d[0];
    max_seq_len_ = max_profile.d[1];
    align_input_bytes_ = AlignTo(max_batch_ * max_seq_len_ * sizeof(int));
    align_aside_intput_bytes_ = AlignTo(max_batch_ * sizeof(int));
    whole_bytes_ = align_input_bytes_ * 4 + align_aside_intput_bytes_ * 9;

    CUDA_CHECK(cudaMalloc(&d_buffer_, whole_bytes_));
    CUDA_CHECK(cudaMallocHost(&h_buffer_, whole_bytes_));

    auto d_buffer_ptr = d_buffer_;
    auto h_buffer_ptr = h_buffer_;
    device_bindings_.resize(engine_->getNbBindings());
    for (size_t i = 0; i < device_bindings_.size(); i++){
        device_bindings_[i] = d_buffer_ptr;
    }

    int b_i = 0;
    while (b_i < 4){
        device_bindings_[start_binding_idx_ + b_i] = d_buffer_ptr;
        host_bindings_.push_back(h_buffer_ptr);

        h_buffer_ptr += align_input_bytes_;
        d_buffer_ptr += align_input_bytes_;

        ++b_i;
    }

    while (b_i < 13){
        device_bindings_[start_binding_idx_ + b_i] = d_buffer_ptr;
        host_bindings_.push_back(h_buffer_ptr);

        h_buffer_ptr += align_aside_intput_bytes_;
        d_buffer_ptr += align_aside_intput_bytes_;

        ++b_i;
    }

    vector<int> input_dim = {max_batch_, max_seq_len_, 1};
    vector<int> aside_input_dim = {max_batch_, 1, 1};

    int binding_idx = start_binding_idx_;
    std::vector<std::vector<int>> input_dims = {
            input_dim, input_dim, input_dim, input_dim,
            aside_input_dim, aside_input_dim, aside_input_dim, aside_input_dim,
            aside_input_dim, aside_input_dim, aside_input_dim, aside_input_dim
    };

    for (size_t i = 0; i < input_dims.size(); i++){
        std::vector<int> dims_vec = input_dims[i];
        nvinfer1::Dims trt_dims;
        trt_dims.nbDims = static_cast<int>(dims_vec.size());
        memcpy(trt_dims.d, dims_vec.data(), sizeof(int) * trt_dims.nbDims);
        context_->setBindingDimensions(binding_idx, trt_dims);
        ++ binding_idx;
    }

    if (!context_->allInputDimensionsSpecified()){
        std::cout << (" context_->allInputDimensionsSpecified() error") << std::endl;
        assert(0);
    }

    for (size_t i = 0; i < device_bindings_.size(); i++){
        s_device_bindings_.push_back(device_bindings_[i]);
    }

    CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_intput_bytes_,
                               cudaMemcpyHostToDevice, cuda_stream_));
    cudaStreamSynchronize(cuda_stream_);
}

/*
 * TrtContext推理函数：
 * 1. 设置设备ID
 * 2. 将CPU数据复制到主机缓冲区
 * 3. 将主机缓冲区数据复制到设备缓冲区
 * 4. 执行推理（使用CUDA图或直接执行）
 * 5. 将设备缓冲区中的结果复制回主机
 * 6. 记录时间戳
 * @param s sample结构体
 * @return 推理结果状态码（0表示成功）
 */
int TrtContext::Forward(struct sample &s) {
    cudaSetDevice(dev_id_);

    int idx = 0;

    auto batch = s.shape_info_0[0];
    auto seq_len = s.shape_info_0[1];
    auto input_bytes = batch * seq_len * sizeof(int);
    auto aside_input_bytes = batch * sizeof(int);

    memcpy(host_bindings_[0], s.i0.data(), input_bytes);
    memcpy(host_bindings_[1], s.i1.data(), input_bytes);
    memcpy(host_bindings_[2], s.i2.data(), input_bytes);
    memcpy(host_bindings_[3], s.i3.data(), input_bytes);

    memcpy(host_bindings_[4], s.i4.data(), aside_input_bytes);
    memcpy(host_bindings_[5], s.i5.data(), aside_input_bytes);
    memcpy(host_bindings_[6], s.i6.data(), aside_input_bytes);
    memcpy(host_bindings_[7], s.i7.data(), aside_input_bytes);
    memcpy(host_bindings_[8], s.i8.data(), aside_input_bytes);
    memcpy(host_bindings_[9], s.i9.data(), aside_input_bytes);
    memcpy(host_bindings_[10], s.i10.data(), aside_input_bytes);
    memcpy(host_bindings_[11], s.i11.data(), aside_input_bytes);

    cudaEvent_t start, stop;
    float elapsed_time = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_intput_bytes_,
                               cudaMemcpyHostToDevice, cuda_stream_));
    vector<int> v_data(128);
    CUDA_CHECK(cudaMemcpyAsync(v_data.data(), device_bindings_[13], 128 * sizeof(int),
                               cudaMemcpyDeviceToHost, cuda_stream_));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

    if (graph_created_){
        CUDA_CHECK(cudaGraphLaunch(instance_, cuda_stream_));
    }else{
        auto status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
        if (!status){
            cerr << "Enqueue failed\n";
            exit(-1);
        }
    }

    s.out_data.resize(batch);
    CUDA_CHECK(cudaMemcpyAsync(s.out_data.data(), device_bindings_[start_binding_idx_ + 12], batch * sizeof(float ),
                               cudaMemcpyDeviceToHost, cuda_stream_));
    cudaStreamSynchronize(cuda_stream_);

    struct timeval tv;
    gettimeofday(&tv, NULL);
    s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
    return 0;
}

/*
 * TrtContext析构函数：
 * 1. 销毁CUDA流
 * 2. 释放设备和主机缓冲区
 */
TrtContext::~TrtContext() {
    CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    cudaFree(d_buffer_);
    cudaFreeHost(h_buffer_);
}

/*
 * 用指定值填充数组
 * @param ptr 数组指针
 * @param size 数组大小
 * @param v 填充值
 */
template <class T>
void _fill(T* ptr, int size, T v){
    for (int i = 0; i < size; i++)  ptr[i] = v;
}

/*
 * 捕获CUDA图以加速后续推理
 * 1. 检查图是否已创建
 * 2. 使用虚拟数据填充所有输入
 * 3. 执行一次初始推理
 * 4. 开始捕获CUDA图
 * 5. 执行推理操作
 * 6. 结束捕获并实例化图
 * @return 状态码（0表示成功，1表示图已创建）
 */
int TrtContext::CaptureCudaGraph() {
    if (graph_created_) return 1;

    auto input_size = max_batch_ * max_seq_len_;
    _fill((int*)host_bindings_[0], input_size, 1);
    _fill((int*)host_bindings_[1], input_size, 1);
    _fill((int*)host_bindings_[2], input_size, 1);
    _fill((float*)host_bindings_[3], input_size, 1.0f);

    _fill((int*)host_bindings_[4], max_batch_, 1);
    _fill((int*)host_bindings_[5], max_batch_, 1);
    _fill((int*)host_bindings_[6], max_batch_, 1);
    _fill((int*)host_bindings_[7], max_batch_, 1);
    _fill((int*)host_bindings_[8], max_batch_, 1);
    _fill((int*)host_bindings_[9], max_batch_, 1);
    _fill((int*)host_bindings_[10], max_batch_, 1);
    _fill((int*)host_bindings_[11], max_batch_, 1);

    CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_intput_bytes_,
                               cudaMemcpyHostToDevice, cuda_stream_));
    auto status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
    if (!status){
        cerr << "Enqueue failed\n";
        exit(1);
    }

    CUDA_CHECK(cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeRelaxed));
    status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
    if (!status){
        cerr << "Enqueue failed\n";
        exit(1);
    }

    CUDA_CHECK(cudaStreamEndCapture(cuda_stream_, &graph_));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
    CUDA_CHECK(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));

    CUDA_CHECK(cudaMemcpyAsync(host_bindings_[12], device_bindings_[12], align_aside_intput_bytes_,
                               cudaMemcpyDeviceToHost, cuda_stream_));
    graph_created_ = true;
    cout << "profile_idx = " << profile_idx_ << " , CaptureCudaGraphDone !" << endl;
    return 0;
}

// } // BEGIN_LIB_NAMESPACE

