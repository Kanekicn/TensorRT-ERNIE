#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
// #include "paddle_inference_api.h"
#include "trt_helper.h"

// using paddle_infer::Config;
// using paddle_infer::Predictor;
// using paddle_infer::CreatePredictor;

// 最大序列长度
static const int MAX_SEQ = 128;

/*
 * 将输入字符串按delimiter分割成多个字符串, 并返回分割后的字符串向量
 * @param str 输入字符串
 * @param delimiter 分割符
 * @param fields 分割后的字符串向量
 */
void split_string(const std::string& str,
                  const std::string& delimiter,
                  std::vector<std::string>& fields) {
    size_t pos = 0;
    size_t start = 0;
    size_t length = str.length();
    std::string token;
    while ((pos = str.find(delimiter, start)) != std::string::npos && start < length) {
        token = str.substr(start, pos - start);
        fields.push_back(token);
        start += delimiter.length() + token.length();
    }
    if (start <= length - 1) {
        token = str.substr(start);
        fields.push_back(token);
    }
}

/**
 * 将输入字符串转换为向量, 并根据padding参数决定是否填充, 并返回形状信息。
 * @param input_str 输入字符串
 * @param padding 是否填充
 * @param shape_info 形状信息
 * @param i64_vec 64位整数向量
 * @param f_vec 浮点数向量
 */
void field2vec(const std::string& input_str,
               bool padding,
               std::vector<int>* shape_info,
               std::vector<int>* i64_vec,
               std::vector<float>* f_vec = nullptr) {
    std::vector<std::string> i_f;
    split_string(input_str, ":", i_f);
    std::vector<std::string> i_v;
    split_string(i_f[1], " ", i_v);
    std::vector<std::string> s_f;
    split_string(i_f[0], " ", s_f);
    for (auto& f : s_f) {
        shape_info->push_back(std::stoi(f));
    }
    int batch_size = shape_info->at(0);
    int seq_len = shape_info->at(1);
    if (i64_vec) {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                i64_vec->push_back(std::stoll(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++j) {
                i64_vec->push_back(0);
            }
        }
    } else {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                f_vec->push_back(std::stof(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++ j) {
                f_vec->push_back(0);
            }
        }
    }
    if (padding) {
        (*shape_info)[1] = MAX_SEQ;
    }
}

/**
 * 将输入字符串转换为sample结构体, 实现方式：   
 * 1. 将输入字符串按;分割成14个字段
 * 2. 将第0个字段按:分割成qid和qid_len
 * 3. 将第1个字段按:分割成label和label_len
 * 4. 将第2个字段按:分割成shape_info_0和i0
 * 5. 将第3个字段按:分割成shape_info_1和i1
 * 6. 将第4个字段按:分割成shape_info_2和i2
 * 7. 将第5个字段按:分割成shape_info_3和i3
 * 8. 将第6个字段按:分割成shape_info_4和i4
 * 9. 将第7个字段按:分割成shape_info_5和i5
 * 10. 将第8个字段按:分割成shape_info_6和i6
 * 11. 将第9个字段按:分割成shape_info_7和i7
 * 12. 将第10个字段按:分割成shape_info_8和i8
 * 13. 将第11个字段按:分割成shape_info_9和i9
 * 14. 将第12个字段按:分割成shape_info_10和i10
 * 15. 将第13个字段按:分割成shape_info_11和i11
 * 16. 将shape_info_11[0]赋值给out_data的大小
 * 17. 返回
 * @param line 输入字符串
 * @param sout 输出sample结构体
 */
void line2sample(const std::string& line, sample* sout) {
    std::vector<std::string> fields;
    split_string(line, ";", fields);
    assert(fields.size() == 14);
    // parse qid
    std::vector<std::string> qid_f;
    split_string(fields[0], ":", qid_f);
    sout->qid = qid_f[1];
    // Parse label
    std::vector<std::string> label_f;
    split_string(fields[1], ":", label_f);
    sout->label = label_f[1];
    // Parse input field
    field2vec(fields[2], true, &(sout->shape_info_0), &(sout->i0));
    field2vec(fields[3], true, &(sout->shape_info_1), &(sout->i1));
    field2vec(fields[4], true, &(sout->shape_info_2), &(sout->i2));
    field2vec(fields[5], true, &(sout->shape_info_3), nullptr, &(sout->i3));
    field2vec(fields[6], false, &(sout->shape_info_4), &(sout->i4));
    field2vec(fields[7], false, &(sout->shape_info_5), &(sout->i5));
    field2vec(fields[8], false, &(sout->shape_info_6), &(sout->i6));
    field2vec(fields[9], false, &(sout->shape_info_7), &(sout->i7));
    field2vec(fields[10], false, &(sout->shape_info_8), &(sout->i8));
    field2vec(fields[11], false, &(sout->shape_info_9), &(sout->i9));
    field2vec(fields[12], false, &(sout->shape_info_10), &(sout->i10));
    field2vec(fields[13], false, &(sout->shape_info_11), &(sout->i11));

    sout->out_data.resize(sout->shape_info_11[0]);
    return;
}

/*
 * 主函数, 实现方式：
 * 1. 初始化trt_helper: 初始化trt_helper, 读取模型参数文件, 设置设备ID
 * 2. 读取输入文件: 打开输入文件, 读取每一行, 调用line2sample(aline, &s)将样本转换为sample结构体, 并加入sample_vec
 * 3. 读取输出文件: 打开输出文件, 准备写入推理结果
 * 4. 读取样本: 遍历输入文件, 调用line2sample(aline, &s)将样本转换为sample结构体, 并加入sample_vec
 * 5. 推理: 遍历样本, 调用trt_helper->Forward(s)
 * 6. 后处理: 遍历样本, 将推理结果写入输出文件
 * 7. 关闭文件
 * 8. 返回
 */
int main(int argc, char *argv[]) {
  // 1. 初始化trt_helper: 初始化trt_helper, 读取模型参数文件, 设置设备ID    
  std::string model_para_file = argv[1];
  std::cout << model_para_file << std::endl;
  auto trt_helper = new TrtHepler(model_para_file, 0);
  // 2. 读取输入文件: 打开输入文件, 读取每一行, 
  // 调用line2sample(aline, &s)将样本转换为sample结构体, 并加入sample_vec
  std::string aline;
  std::ifstream ifs;
  ifs.open(argv[2], std::ios::in);
  std::ofstream ofs;
  ofs.open(argv[3], std::ios::out);
  std::vector<sample> sample_vec;
  while (std::getline(ifs, aline)) {
      sample s;
      line2sample(aline, &s);
      sample_vec.push_back(s);
  }

  // 3. 推理: 遍历样本, 调用trt_helper->Forward(s)
  for (auto& s : sample_vec) {
      // //run(predictor.get(), s);
      trt_helper->Forward(s);
  }

  // 4. 后处理: 遍历样本, 将推理结果写入输出文件
  for (auto& s : sample_vec) {
      std::ostringstream oss;
      oss << s.qid << "\t";
      oss << s.label << "\t";
      for (int i = 0; i < s.out_data.size(); ++i) {
          oss << s.out_data[i];
          if (i == s.out_data.size() - 1) {
              oss << "\t";
          } else {
              oss << ",";
          }
      }
      oss << s.timestamp << "\n";
      ofs.write(oss.str().c_str(), oss.str().length());
  }
  // 5. 关闭文件
  ofs.close();
  ifs.close();
  return 0;
}
