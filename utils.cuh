// utils.cuh
#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <string>
#include <cudnn.h>
#include <cublas_v2.h>

// 檢查 CUDNN 的呼叫是否成功
#define CHECK_CUDNN(call)                                                    \
    {                                                                        \
        cudnnStatus_t status = (call);                                       \
        if (status != CUDNN_STATUS_SUCCESS) {                                \
            std::cerr << "Error on line " << __LINE__ << ": "                \
                      << cudnnGetErrorString(status) << std::endl;           \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    }

// 檢查 CUBLAS 的呼叫是否成功
#define CHECK_CUBLAS(call)                                                   \
    {                                                                        \
        cublasStatus_t status = (call);                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "CUBLAS Error on line " << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    }

// Kernel 函式宣告

// 用於快速初始化陣列值
__global__ void fill_array(float* arr, int size, float value);

// 使用 SGD 進行參數更新
__global__ void sgd_update(float* params, float* grads, float learning_rate, int size);

// Host 函式宣告

// 載入 MNIST 的二進制格式資料集
void loadMNIST(std::vector<float> &images, std::vector<int> &labels, const std::string &imageFile, const std::string &labelFile);

// 正規化圖片
void normalizeImages(std::vector<float> &images);

// Cross Entropy 損失函數
float crossEntropyLoss(const std::vector<int> &labels, const std::vector<float> &predictions, int batch_size, int num_classes);

// 計算梯度
void computeGradients(const float* output_data, const std::vector<int>& labels, float* loss_gradients, int batch_size, int num_classes);

// 保存模型参数
void saveModel(const std::string& filename, const std::vector<float>& data);

// 隨機打亂資料
void shuffleData(std::vector<float> &images, std::vector<int> &labels);

// 分割資料集為訓練集和驗證集
void splitData(const std::vector<float> &images, const std::vector<int> &labels,
               std::vector<float> &train_images, std::vector<int> &train_labels,
               std::vector<float> &val_images, std::vector<int> &val_labels,
               float train_ratio);

#endif // UTILS_CUH
