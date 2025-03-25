#!/usr/bin/env python
# encoding=utf8
# ras-sat ernie模型评估脚本
# 评价方式：Metric损失率=abs(原始Metric-优化后Metric)/原始Metric

# 1个输入：优化后模型打分文件
#       打分文件命名格式：参赛label.res.txt和perf.res.txt
#       打分文件每行格式：query_id、label_id、score_list、微秒时间戳共4个输出，字符串格式且以\t分割
#                         多batch的score_list以逗号分割

# 2个输出：metric: 数值越大越好
#    推理平均时延：数值越小越好

import sys
import os
import math


def metric(qid, label, pred):
    """
    计算排序质量指标
    
    参数:
        qid: 查询ID列表，用于分组计算
        label: 标签列表(真实相关性得分)
        pred: 预测分数列表
    
    返回:
        m: 排序质量指标，即正确排序的对数与错误排序的对数之比
    
    步骤:
    1. 按查询ID对样本进行分组
    2. 对于每个查询下的样本对，计算是否正确排序
    3. 统计正确排序和错误排序的对数
    4. 返回正确排序对数与错误排序对数的比值
    """
    saver = {}
    assert len(qid) == len(label) == len(pred)
    
    for q, l, p in zip(qid, label, pred):
        if q not in saver:
            saver[q] = []
        saver[q].append((l, p))
    
    p = 0
    n = 0
    
    for qid, outputs in saver.items():
        for i in range(0, len(outputs)):
            l1, p1 = outputs[i]
            for j in range(i + 1, len(outputs)):
                l2, p2 = outputs[j]
                
                if l1 > l2:
                    if p1 > p2:
                        p += 1
                    elif p1 < p2:
                        n += 1
                elif l1 < l2:
                    if p1 < p2:
                        p += 1
                    elif p1 > p2:
                        n += 1
    
    m = 1. * p / n if n > 0 else 0.0
    return m


def avg_inf(opt_tms):
    """
    计算平均推理时间
    
    参数:
        opt_tms: 时间戳列表，每个元素代表一次推理的完成时间
    
    返回:
        平均推理时间(微秒)
    
    步骤:
    1. 计算相邻时间戳的差值，即每次推理所需时间
    2. 计算这些时间差的平均值
    """
    opt_nums = len(opt_tms)
    opt_all_times = 0.0
    
    for i in range(1, opt_nums):
        opt_all_times += float(opt_tms[i]) - float(opt_tms[i-1])


    return 1.* (opt_all_times / (opt_nums-1))


def evalute(opt_list):
    """
    评估模型性能
    
    参数:
        opt_list: 模型输出列表，每行包含查询ID、标签、预测分数和时间戳
    
    返回:
        result: 包含评估指标和平均推理时间的字典
    
    步骤:
    1. 解析每一行数据，提取查询ID、标签、预测分数和时间戳
    2. 调用metric函数计算排序质量指标
    3. 调用avg_inf函数计算平均推理时间
    4. 返回包含两个评估结果的字典
    """
    opt_qids = []
    opt_labels = []
    opt_scores = []
    opt_tms = []
    
    for line in opt_list:
        value = line.strip().split("\t")
        opt_qids.append(int(value[0]))
        
        if value[1] != "-":
            opt_labels.append(int(float(value[1])))
            opt_scores.append(float(value[2]))
        else:
            opt_scores.append(value[2])
            
        opt_tms.append(value[3])
    
    opt_metric = "-"
    if len(opt_labels):
        opt_metric = metric(opt_qids, opt_labels, opt_scores)
    
    result = {}
    result["metric"] = opt_metric
    result["inf_time(us)"] = avg_inf(opt_tms)

    return result


if __name__ == "__main__":
    """
    主函数：读取输入文件并评估模型性能
    
    使用方法:
    python local_evaluate.py label.res.txt
    
    步骤:
    1. 从命令行参数读取输入文件路径
    2. 读取文件内容
    3. 调用evalute函数进行评估
    4. 打印评估结果
    """
    opt_list = []
    with open(sys.argv[1], 'r') as f:
        for line in f.readlines():
            opt_list.append(line.strip())

    print(evalute(opt_list))
