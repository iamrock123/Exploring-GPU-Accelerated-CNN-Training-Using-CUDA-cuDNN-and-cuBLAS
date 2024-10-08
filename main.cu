// main.cu

#include <iostream>
#include <fstream>
#include <vector>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>

#include "utils.cuh"

int main() {
    
    // Step 1. 初始化和資源分配

    // 用於管理 cuDNN 庫的狀態和資源，需要在使用 cuDNN 任何函式之前建立。
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 用於管理 cuBLAS 庫的狀態和資源，需要在使用 cuBLAS 任何函式之前建立。
    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    // Step 2. 載入訓練資料集和預處理

    // 載入 MNIST 訓練資料 (shape: [28, 28, 1])
    std::vector<float> images; 
    std::vector<int> labels;
    loadMNIST(images, labels, "./dataset/train-images.idx3-ubyte", "./dataset/train-labels.idx1-ubyte");

    // 隨機打亂訓練資料
    shuffleData(images, labels);
    
    // 將訓練圖片進行標準化 (從 [0,255] 的範圍縮放到 [0,1] 之間)
    normalizeImages(images);

    // 將資料集分割成訓練集跟驗證集
    std::vector<float> train_images, val_images;
    std::vector<int> train_labels, val_labels;
    splitData(images, labels, train_images, train_labels, val_images, val_labels, 0.8f);

    // Step 3. 模型參數初始化

    // 定義 CNN 神經網路參數
    const int batch_size = 64;          //訓練批次中的數量
    const int channels = 1;             //輸入圖片的通道數 (彩色為3, 灰階為1)
    const int height = 28;              //圖片高度
    const int width = 28;               //圖片寬度
    const int num_classes = 10;         //分類類別         
    const float learning_rate = 0.001f; //學習率

    // Step 4. 建立卷積結構

    // 設定模型輸入的 Tensor Descriptor

    // input_desc 模型輸入
    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        batch_size, channels, 
        height, width
    ));

    // 卷積層 的 Descriptor

    // 第一層 convolutional layer 的參數
    // filter_desc1 用於設定第一層 convolutional layer 的 filter
    cudnnFilterDescriptor_t filter_desc1;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc1));

    // 定義 32 個 5 * 5 的 filters，每個 filter 的 channel = 1
    const int filter_count1 = 32, filter_height1 = 5, filter_width1 = 5;
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc1, 
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 
        filter_count1, channels, 
        filter_height1, filter_width1
    ));

    // conv_desc1  用於設定第一層 convolutional layer 的卷積操作屬性
    // 包含填充（Padding）、步幅（Stride）、擴張（Dilation）、卷積模式（Convolution Mode）和資料類型
    cudnnConvolutionDescriptor_t conv_desc1;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc1));
    const int pad_h1 = 2, pad_w1 = 2, stride_h1 = 1, stride_w1 = 1;
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc1, 
        pad_h1, pad_w1, 
        stride_h1, stride_w1, 
        1, 1, 
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
    ));

    // 使用 cudnnGetConvolution2dForwardOutputDim 計算卷積操作後輸出的 tensor shape
    // 存儲在 output_n1（批次大小）、output_c1（輸出通道數）、output_h1（輸出高度）、output_w1（輸出寬度）中。
    int output_n1, output_c1, output_h1, output_w1;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc1, 
        input_desc, 
        filter_desc1, 
        &output_n1, 
        &output_c1, 
        &output_h1, 
        &output_w1
    ));

    // conv1_output_desc 用於計算卷積後的操作輸出 Tensor
    cudnnTensorDescriptor_t conv1_output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv1_output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        conv1_output_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        output_n1, output_c1, 
        output_h1, output_w1
    ));

    // 建立 conv1 Bias 的 Tensor Descriptor
    cudnnTensorDescriptor_t conv1_bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv1_bias_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        conv1_bias_desc, 
        CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, 
        1,                  // 批次大小為 1，通常在每個通道上共享
        filter_count1,      // 通道數，等於卷積的過濾器數量
        1,                  // 高度和寬度設置為 1，因為不涉及空間維度
        1
    ));

    // 第二層 convolutional layer 的參數
    // filter_desc2 用於設定第二層 convolutional layer 的 filter
    cudnnFilterDescriptor_t filter_desc2;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc2));

    // 定義 64 個 5 * 5 的 filters，每個 filter 的 channel = 1
    const int filter_count2 = 64;
    const int filter_height2 = 5;
    const int filter_width2 = 5;

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc2, 
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 
        filter_count2, filter_count1, 
        filter_height2, filter_width2
    ));

    // conv_desc2  用於設定第二層 convolutional layer 的卷積操作屬性
    cudnnConvolutionDescriptor_t conv_desc2;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc2));
    const int pad_h2 = 2, pad_w2 = 2, stride_h2 = 1, stride_w2 = 1;
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc2, 
        pad_h2, 
        pad_w2, 
        stride_h2, 
        stride_w2, 
        1, 
        1, 
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_FLOAT
    ));


    // 使用 cudnnGetConvolution2dForwardOutputDim 計算卷積操作後輸出的 tensor shape
    // 存儲在 output_n2（批次大小）、output_c2（輸出通道數）、output_h2（輸出高度）、output_w2（輸出寬度）中。
    int output_n2, output_c2, output_h2, output_w2;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc2, 
        conv1_output_desc, filter_desc2, 
        &output_n2, &output_c2, 
        &output_h2, &output_w2
    ));

    // conv2_output_desc 用於計算卷積後的操作輸出 Tensor
    cudnnTensorDescriptor_t conv2_output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv2_output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        conv2_output_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        output_n2, output_c2, 
        output_h2, output_w2
    ));

    // 建立 conv2 Bias 的 Tensor Descriptor
    cudnnTensorDescriptor_t conv2_bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv2_bias_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        conv2_bias_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        1, filter_count2, 
        1, 1
    ));

    // activation_desc 用於設定 Activation Function
    cudnnActivationDescriptor_t activation_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        activation_desc,            
        CUDNN_ACTIVATION_RELU,      // 指定要使用的 Activation Function
        CUDNN_PROPAGATE_NAN,        // 如何處理 NaN 值 (忽略 NaN)
        0.0                         // Activation Function 數值上限 (ReLU不需要)
    ));


    // Step 5. 分配 GPU 記憶體

    // 全連接層的輸入和輸出大小
    const int fc_input_size = output_c2 * output_h2 * output_w2;
    const int fc_output_size = num_classes;

    // 模型参数和損失梯度
    float *conv1_weights, *conv1_bias; // 第一層卷積的 Weights 和 Bias
    float *conv2_weights, *conv2_bias; // 第二層卷積的 Weights 和 Bias
    float *fc_weights, *fc_bias;       // 全連接層的 Weights 和 Bias
    float *loss_gradients;             // 損失函數的梯度，用於反向傳播

    // 為参数和損失梯度分配記憶體
    cudaMalloc(&conv1_weights, filter_count1 * channels * filter_height1 * filter_width1 * sizeof(float));          // 儲存所有 conv1 Filters Weights
    cudaMalloc(&conv1_bias, filter_count1 * sizeof(float));                                                         // 儲存所有 conv1 Bias
    cudaMalloc(&conv2_weights, filter_count2 * filter_count1 * filter_height2 * filter_width2 * sizeof(float));     // 儲存所有 conv2 Filters Weights
    cudaMalloc(&conv2_bias, filter_count2 * sizeof(float));                                                         // 儲存所有 conv2 Bias
    cudaMalloc(&fc_weights, fc_input_size * fc_output_size * sizeof(float));                                        // 儲存所有 fc Weights
    cudaMalloc(&fc_bias, fc_output_size * sizeof(float));                                                           // 儲存所有 fc Bias
    cudaMalloc(&loss_gradients, batch_size * num_classes * sizeof(float));                                          // 儲存所有 fc Weights

    // 為模型參數的梯度分配記憶體
    // 在反向傳播過程中，需要存儲每個參數的梯度，以便更新模型的 Weights 和 Bias
    float *conv1_weight_gradients, *conv1_bias_gradients;
    float *conv2_weight_gradients, *conv2_bias_gradients;
    float *fc_weight_gradients, *fc_bias_gradients;

    // 為参数的梯度分配記憶體
    cudaMalloc(&conv1_weight_gradients, filter_count1 * channels * filter_height1 * filter_width1 * sizeof(float));
    cudaMalloc(&conv1_bias_gradients, filter_count1 * sizeof(float));
    cudaMalloc(&conv2_weight_gradients, filter_count2 * filter_count1 * filter_height2 * filter_width2 * sizeof(float));
    cudaMalloc(&conv2_bias_gradients, filter_count2 * sizeof(float));
    cudaMalloc(&fc_weight_gradients, fc_input_size * fc_output_size * sizeof(float));
    cudaMalloc(&fc_bias_gradients, fc_output_size * sizeof(float));

    // 分配記憶體給在向前和反向傳播的過程中，儲存各層中間的輸出，以便後續的計算

    // 計算每個層數輸入輸出所需要的記憶體大小
    size_t input_bytes = batch_size * channels * height * width * sizeof(float);                    // 每個批次的輸入圖片所需的記憶體
    size_t conv1_output_bytes = batch_size * output_c1 * output_h1 * output_w1 * sizeof(float);     // 第一層卷積輸出所需的記憶體
    size_t conv2_output_bytes = batch_size * output_c2 * output_h2 * output_w2 * sizeof(float);     // 第二層卷積輸出所需的記憶體
    size_t fc_input_bytes = batch_size * fc_input_size * sizeof(float);                             // 全連接層輸入所需的記憶體。
    size_t fc_output_bytes = batch_size * fc_output_size * sizeof(float);                           // 全連接層輸出所需的記憶體。

    // 在 GPU 上分配記憶體來儲存輸入、卷積層輸出和全連接層輸出等
    float *d_input, *d_conv1_output, *d_conv2_output, *d_fc_input, *d_fc_output;
    float *d_relu1_output, *d_relu2_output;

    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_conv1_output, conv1_output_bytes);
    cudaMalloc(&d_conv2_output, conv2_output_bytes);
    cudaMalloc(&d_fc_input, fc_input_bytes);
    cudaMalloc(&d_fc_output, fc_output_bytes);

    // 分配 ReLU 的輸出，用於反向傳播
    cudaMalloc(&d_relu1_output, conv1_output_bytes); // 儲存第一層卷積的 Activation 輸出
    cudaMalloc(&d_relu2_output, conv2_output_bytes); // 儲存第二層卷積的 Activation 輸出

    // 用於計算全連接層的 Bias 梯度
    float *ones;
    cudaMalloc(&ones, batch_size * sizeof(float));

    // 使用 Kernel Function 透過 CUDA 填充 ones 陣列
    int threads_per_block = 256;
    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
    fill_array<<<blocks_per_grid, threads_per_block>>>(ones, batch_size, 1.0f);
    cudaDeviceSynchronize();  // 等待 Kernel 執行完畢 (因為 cpu 跟 gpu是異構架構)


    // Step 6. 初始化和傳輸權重到 GPU

    // 隨機初始化權重
    // 如果所有權重都初始化為相同的值，神經網路中的各個神經元將學習相同的特徵，導致模型性能受限。
    // 隨機初始化權重可以打破對稱性，讓不同的神經元學習不同的特徵。
    // 適當的初始化可以幫助控制梯度的流動，防止梯度消失或爆炸，從而加速模型的收斂。
    // Bias 通常初始化為零或小的常數值，這有助於模型在訓練初期更快地收斂。
    std::mt19937 rng(std::random_device{}());                                                               // 使用 Mersenne Twister 生成隨機數
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);                                                // 用於隨機初始化權重，使其在指定範圍內均勻分布
    std::vector<float> host_conv1_weights(filter_count1 * channels * filter_height1 * filter_width1);
    std::vector<float> host_conv1_bias(filter_count1); 
    std::vector<float> host_conv2_weights(filter_count2 * filter_count1 * filter_height2 * filter_width2);
    std::vector<float> host_conv2_bias(filter_count2);
    std::vector<float> host_fc_weights(fc_input_size * fc_output_size);
    std::vector<float> host_fc_bias(fc_output_size);

    for (auto &w : host_conv1_weights) w = dist(rng);
    for (auto &b : host_conv1_bias) b = dist(rng);
    for (auto &w : host_conv2_weights) w = dist(rng);
    for (auto &b : host_conv2_bias) b = dist(rng);
    for (auto &w : host_fc_weights) w = dist(rng);
    for (auto &b : host_fc_bias) b = dist(rng);

    // 將每層的 Weights 和 Bias 從 Host memory 傳輸到 Device memory
    cudaMemcpy(conv1_weights, host_conv1_weights.data(), host_conv1_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv1_bias, host_conv1_bias.data(), host_conv1_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv2_weights, host_conv2_weights.data(), host_conv2_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv2_bias, host_conv2_bias.data(), host_conv2_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_weights, host_fc_weights.data(), host_fc_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_bias, host_fc_bias.data(), host_fc_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Step 7. 實現向前傳播

    // 分配 workspace 的記憶體
    // 在 Device 中，一個計算所需的空間必須事前聲明
    // 除了輸入輸出之外，進行這個計算所需的額外 “工作空間”，也可以簡單地理解為空間複雜度。
    void *d_workspace = nullptr;

    // 7-1. 實現向前傳播(Forward Pass)

    // 獲取向前傳播所需空間的大小
    size_t fwd_workspace_size1 = 0, fwd_workspace_size2 = 0;

    // 選擇第一層向前卷積的算法
    cudnnConvolutionFwdAlgo_t conv1_fwd_algo_type;      // 用於表示卷積向前算法的類型
    cudnnConvolutionFwdAlgoPerf_t conv1_fwd_algo_perf;  // 儲存算法的具體資訊(名稱，性能，執行時間)
    int conv1_fwd_returned_algo_count;                      // 儲存返回的算法數量

    // 選擇最佳的向前卷積算法 
    // 使用 cudnnGetConvolutionForwardAlgorithm_v7 會根據 input、filter、Conv、output Descriptor 的特性，評估並選擇最佳的卷積算法。
    // 包括直接卷積（Direct Convolution）、基於 FFT 的卷積（FFT-based Convolution）、Winograd 算法等。
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,                                          // 管理 cudnn 的狀態和資源
        input_desc,                                     // 輸入的 Descriptor (輸入資料的維度及格式)
        filter_desc1,                                   // 第一層卷積 filter 的 Descriptor (包含 filter 數量，尺寸)
        conv_desc1,                                     // 第一層卷積 conv 的 Descriptor (包含填充，步長)
        conv1_output_desc,                              // 第一層卷積輸出 Tensor 的 Descriptor (包含進行卷積操作輸出後的資料維度跟格式)
        1,                                              // 只須返回一個最佳算法
        &conv1_fwd_returned_algo_count,                 // 儲存返回算法數量
        &conv1_fwd_algo_perf                            // 儲存選擇算法的性能結果
    ));

    // 使用 conv1_fwd_algo_type 提取選擇的算法
    conv1_fwd_algo_type = conv1_fwd_algo_perf.algo;

    // 獲取選定算法所需的工作空間大小
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,                                          // 管理 cudnn 的狀態和資源
        input_desc,                                     // input Descriptor
        filter_desc1,                                   // conv1 filter Descriptor
        conv_desc1,                                     // conv1 Descriptor
        conv1_output_desc,                              // conv1 output Descriptor
        conv1_fwd_algo_type,                            // cudnnGetConvolutionForwardAlgorithm_v7 所選擇的算法
        &fwd_workspace_size1                            // 儲存 conv1 所需工作空間的大小
    ));

    // 選擇第二層向前卷積的算法
    cudnnConvolutionFwdAlgo_t conv2_fwd_algo_type;      // 用於表示卷積向前算法的類型
    cudnnConvolutionFwdAlgoPerf_t conv2_fwd_algo_perf;  // 儲存算法的具體資訊(名稱，性能，執行時間)
    int conv2_fwd_returned_algo_count;                      // 儲存返回的算法數量

    // 選擇最佳的向前卷積算法
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        conv1_output_desc,                              // 接著第一層 conv1 的輸出
        filter_desc2,
        conv_desc2,
        conv2_output_desc,
        1,
        &conv2_fwd_returned_algo_count,
        &conv2_fwd_algo_perf
    ));

    // 使用 conv2_fwd_algo_type 提取選擇的算法
    conv2_fwd_algo_type = conv2_fwd_algo_perf.algo;

    // 獲取選定算法所需的工作空間大小
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        conv1_output_desc,                              // 接著第一層 conv1 的輸出
        filter_desc2,
        conv_desc2,
        conv2_output_desc,
        conv2_fwd_algo_type,
        &fwd_workspace_size2                                // 儲存 conv2 所需工作空間的大小
    ));

    // 7-2. 實現反向傳播(Backward Pass)

    // 選擇第一層反向卷積的算法
    // 反向傳播與向前傳播不同的是，需要計算 filter 以及 data 的梯度
    cudnnConvolutionBwdFilterAlgo_t conv1_bwd_filter_algo_type;
    cudnnConvolutionBwdDataAlgo_t conv1_bwd_data_algo_type;
    cudnnConvolutionBwdFilterAlgoPerf_t conv1_bwd_filter_algo_perf;
    cudnnConvolutionBwdDataAlgoPerf_t conv1_bwd_data_algo_perf;
    int conv1_bwd_filter_algo_count, conv1_bwd_data_algo_count;
    size_t bwd_filter_workspace_size1 = 0, bwd_data_workspace_size1 = 0;

    // 選擇最佳的反向卷積算法(filter) 
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cudnn,
        input_desc,
        conv1_output_desc,
        conv_desc1,
        filter_desc1,
        1,
        &conv1_bwd_filter_algo_count,
        &conv1_bwd_filter_algo_perf
    ));

    // 使用 conv1_bwd_filter_algo_type 提取選擇的算法
    conv1_bwd_filter_algo_type = conv1_bwd_filter_algo_perf.algo;

    // 第一層反向卷積 filter 算法工作空間大小
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn,
        input_desc,
        conv1_output_desc,
        conv_desc1,
        filter_desc1,
        conv1_bwd_filter_algo_type,
        &bwd_filter_workspace_size1
    ));

    // 選擇最佳的反向卷積算法(data) 
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cudnn,
        filter_desc1,
        conv1_output_desc,
        conv_desc1,
        input_desc,
        1,
        &conv1_bwd_data_algo_count,
        &conv1_bwd_data_algo_perf
    ));

    // 使用 conv1_bwd_data_algo_type 提取選擇的算法
    conv1_bwd_data_algo_type = conv1_bwd_data_algo_perf.algo;

    // 第一層反向卷積 data 算法工作空間大小
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        filter_desc1,
        conv1_output_desc,
        conv_desc1,
        input_desc,
        conv1_bwd_data_algo_type,
        &bwd_data_workspace_size1
    ));

    // 選擇第二層反向卷積的算法
    cudnnConvolutionBwdFilterAlgo_t conv2_bwd_filter_algo_type;
    cudnnConvolutionBwdDataAlgo_t conv2_bwd_data_algo_type;
    cudnnConvolutionBwdFilterAlgoPerf_t conv2_bwd_filter_algo_perf;
    cudnnConvolutionBwdDataAlgoPerf_t conv2_bwd_data_algo_perf;
    int conv2_bwd_filter_algo_count, conv2_bwd_data_algo_count;
    size_t bwd_filter_workspace_size2 = 0, bwd_data_workspace_size2 = 0;

    // 選擇最佳的反向卷積算法(filter) 
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cudnn,
        conv1_output_desc,
        conv2_output_desc,
        conv_desc2,
        filter_desc2,
        1,
        &conv2_bwd_filter_algo_count,
        &conv2_bwd_filter_algo_perf
    ));

    // 使用 conv2_bwd_filter_algo_type 提取選擇的算法
    conv2_bwd_filter_algo_type = conv2_bwd_filter_algo_perf.algo;

    // 第二層反向卷積 filter 算法工作空間大小
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn,
        conv1_output_desc,
        conv2_output_desc,
        conv_desc2,
        filter_desc2,
        conv2_bwd_filter_algo_type,
        &bwd_filter_workspace_size2
    ));

    // 選擇最佳的反向卷積算法(data) 
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cudnn,
        filter_desc2,
        conv2_output_desc,
        conv_desc2,
        conv1_output_desc,
        1,
        &conv2_bwd_data_algo_count,
        &conv2_bwd_data_algo_perf
    ));

    // 使用 conv2_bwd_data_algo_type 提取選擇的算法
    conv2_bwd_data_algo_type = conv2_bwd_data_algo_perf.algo;

    // 第二層反向卷積 data 算法工作空間大小
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        filter_desc2,
        conv2_output_desc,
        conv_desc2,
        conv1_output_desc,
        conv2_bwd_data_algo_type,
        &bwd_data_workspace_size2
    ));

    // 分配工作空間
    size_t workspace_sizes[] = {
        fwd_workspace_size1,            // 第一層向前卷積
        fwd_workspace_size2,            // 第二層向前卷積
        bwd_filter_workspace_size1,     // 第一層反前卷積(filter)
        bwd_data_workspace_size1,       // 第一層反前卷積(data)
        bwd_filter_workspace_size2,     // 第二層反前卷積(filter)
        bwd_data_workspace_size2        // 第二層反前卷積(data)
    };

    // 找出需要最大的 workspace
    size_t workspace_bytes = *std::max_element(std::begin(workspace_sizes), std::end(workspace_sizes));

    // 分配 GPU 記憶體 (workspace_bytes 不可超過 GPU 最大記憶體)
    cudaMalloc(&d_workspace, workspace_bytes);

    // Step 8. 開始進行模型訓練

    // 設定訓練參數
    const int num_epochs = 10;                                      // 訓練 epoch 數量
    const int num_samples = train_images.size() / (height * width); // 訓練的總樣本數(需從一維陣列回推)
    const int num_batches = num_samples / batch_size;               // 每個 epoch 的批次數量

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // 每個 epoch 開始時進行初始化，用於累加 loss 和 Accuracy
        float epoch_train_loss = 0.0f;
        int total_train_correct = 0;

        for (int batch = 0; batch < num_batches; ++batch) {
            // 取出當前批次的圖片和標籤
            std::vector<float> batch_images(batch_size * channels * height * width);
            std::vector<int> batch_labels(batch_size);

            // 循環處理批次中的每個樣本
            for (int i = 0; i < batch_size; ++i) {
                int idx = batch * batch_size + i;                           // idx 用於計算當前樣本在整個訓練集的索引
                std::copy(
                    train_images.begin() + idx * height * width,            // 第 idx 張圖片的起始位置
                    train_images.begin() + (idx + 1) * height * width,      // 第 idx 張圖片的結束位置，也就是 idx * height * width + height * width
                    batch_images.begin() + i * height * width               // 儲存位置
                );
                batch_labels[i] = train_labels[idx];                        // 複製標籤
            }

            // 將批次的訓練資料複製到 GPU 上
            cudaMemcpy(d_input, batch_images.data(), input_bytes, cudaMemcpyHostToDevice);

            // 向前傳播
            // Output = alpha * Operation(Input) + beta * Output
            const float alpha = 1.0f; // 縮放操作的輸出
            const float beta = 0.0f;  // 縮放現有的輸出

            // 第一層向前卷積
            CHECK_CUDNN(cudnnConvolutionForward(
                cudnn,
                &alpha,                             // 縮放卷積操作結果
                input_desc, d_input,                // input 的 descriptor 跟 指向 gpu 的 pointer
                filter_desc1, conv1_weights,        // 第一層卷積 filter 跟 weightd 的 descriptor
                conv_desc1, conv1_fwd_algo_type,    // 第一層卷積的操作和算法
                d_workspace, workspace_bytes,       // workspace 的 pointer 和大小
                &beta,                              // 縮放現有輸出
                conv1_output_desc, d_conv1_output   // 卷積輸出的 descriptor
            ));

            // 添加 Bias
            CHECK_CUDNN(cudnnAddTensor(
                cudnn,
                &alpha,                             // 第一個 alpha 用於縮放 conv1_bias
                conv1_bias_desc, conv1_bias,
                &alpha,                             // 這邊應該是 beta(0)，但需要累加操作，所以 reuse alpha(1.0f)
                conv1_output_desc, d_conv1_output
            ));

            // 第一層向前卷積的 Activation Function
            CHECK_CUDNN(cudnnActivationForward(
                cudnn,
                activation_desc, 
                &alpha,
                conv1_output_desc, d_conv1_output,
                &beta,
                conv1_output_desc, d_relu1_output
            ));

            // 第二層向前卷積
            CHECK_CUDNN(cudnnConvolutionForward(
                cudnn,
                &alpha,
                conv1_output_desc, d_relu1_output,
                filter_desc2, conv2_weights,
                conv_desc2, conv2_fwd_algo_type,
                d_workspace, workspace_bytes,
                &beta,
                conv2_output_desc, d_conv2_output
            ));

            // 添加 Bias
            CHECK_CUDNN(cudnnAddTensor(
                cudnn,
                &alpha,
                conv2_bias_desc, conv2_bias,
                &alpha,
                conv2_output_desc, d_conv2_output
            ));

            // 第二層向前卷積的 Activation Function
            CHECK_CUDNN(cudnnActivationForward(
                cudnn,
                activation_desc,
                &alpha,
                conv2_output_desc, d_conv2_output,
                &beta,
                conv2_output_desc, d_relu2_output
            ));

            // 將卷基層的輸出展開作為全連接層的輸入，d_relu2_output 複製到 d_fc_input
            cudaMemcpy(
                d_fc_input,                 // 全連接層輸入的 GPU pointer
                d_relu2_output,             // 第二層卷積 Activation Function 的 GPU Pointer
                conv2_output_bytes,         // 卷積輸出大小
                cudaMemcpyDeviceToDevice
            );

            // 全連接層
            // cublasSgemm : 執行全連接層的矩陣操作
            CHECK_CUBLAS(cublasSgemm(
                cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,   // CUBLAS_OP_T : 矩陣 A 要轉置，CUBLAS_OP_N : 矩陣 B 不變
                fc_output_size, batch_size, fc_input_size,
                &alpha,
                fc_weights, fc_input_size,
                d_fc_input, fc_input_size,
                &beta,
                d_fc_output, fc_output_size
            ));

            // 添加全連接層的 Bias
            CHECK_CUBLAS(cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                fc_output_size, batch_size, 1,
                &alpha,
                fc_bias, fc_output_size,
                ones, 1,
                &alpha,
                d_fc_output, fc_output_size
            ));

            // Softmax Activation Function
            cudnnTensorDescriptor_t fc_output_desc;     // 建立全連接層輸出的形狀和格式
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&fc_output_desc));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(
                fc_output_desc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batch_size, num_classes, 1, 1           // N, C, H, W
            ));

            // Softmax 向前傳播
            CHECK_CUDNN(cudnnSoftmaxForward(
                cudnn,
                CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, // 準確和操作模式
                &alpha,
                fc_output_desc, d_fc_output,                        // 經過 softmax 輸入
                &beta,
                fc_output_desc, d_fc_output                         // 經過 softmax 輸出
            ));

            // 計算損失
            std::vector<float> output(batch_size * num_classes);

            // 將最後 Softmax 輸出從 GPU 的記憶體（d_fc_output）複製到 Host 的記憶體（output）
            cudaMemcpy(output.data(), d_fc_output, fc_output_bytes, cudaMemcpyDeviceToHost);

            // 計算準確度
            int batch_correct = 0;
            for (int i = 0; i < batch_size; ++i) {
                // 找出批次樣本中預測概率最大的類別
                int predicted_label = std::distance(            //使用距離函式，與最大元素間的距離即為預測的類別標籤
                    output.begin() + i * num_classes,
                    std::max_element(
                        output.begin() + i * num_classes,       //樣本起始位置
                        output.begin() + (i + 1) * num_classes  //樣本結束位置
                    )
                );
                if (predicted_label == batch_labels[i]) {
                    batch_correct++;
                }
            }

            total_train_correct += batch_correct;

            // 使用 Cross Entropy 計算 train loss
            float batch_loss = crossEntropyLoss(batch_labels, output, batch_size, num_classes);
            epoch_train_loss += batch_loss * batch_size;

            // 計算梯度
            std::vector<float> host_loss_gradients(batch_size * num_classes);   // 存儲每個樣本對每個類別的梯度
            computeGradients(output.data(), batch_labels, host_loss_gradients.data(), batch_size, num_classes);
            cudaMemcpy(loss_gradients, host_loss_gradients.data(), batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);

            // 反向傳播
            // 全連接層的反向傳播
            float *d_fc_input_gradients;    //分配GPU記憶體，用於儲存全連接層輸入的梯度
            cudaMalloc(&d_fc_input_gradients, batch_size * fc_input_size * sizeof(float));

            // 計算全連接層的 weight 梯度
            CHECK_CUBLAS(cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_T,
                fc_input_size, fc_output_size, batch_size,
                &alpha,
                d_fc_input, fc_input_size,
                loss_gradients, fc_output_size,
                &beta,
                fc_weight_gradients, fc_input_size
            ));

            // 計算全連接層的 bias 梯度
            CHECK_CUBLAS(cublasSgemv(
                cublas,
                CUBLAS_OP_N, fc_output_size, batch_size,
                &alpha,
                loss_gradients, fc_output_size,
                ones, 1,
                &beta,
                fc_bias_gradients, 1
            ));

            // 計算全連接層的輸入梯度
            CHECK_CUBLAS(cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                fc_input_size, batch_size, fc_output_size,
                &alpha,
                fc_weights, fc_input_size,
                loss_gradients, fc_output_size,
                &beta,
                d_fc_input_gradients, fc_input_size
            ));

            // 卷積層的反向傳播
            // 第二層卷積的反向傳播
            float *d_relu2_gradients = d_fc_input_gradients;    // Shape 一致

            // 反向傳播 ReLU 梯度計算
            CHECK_CUDNN(cudnnActivationBackward(
                cudnn,
                activation_desc,
                &alpha,
                conv2_output_desc, d_relu2_output,
                conv2_output_desc, d_relu2_gradients,
                conv2_output_desc, d_relu2_output,
                &beta,
                conv2_output_desc, d_conv2_output
            ));

            // 第二層卷積的 weight 梯度計算
            CHECK_CUDNN(cudnnConvolutionBackwardFilter(
                cudnn,
                &alpha,
                conv1_output_desc, d_relu1_output,
                conv2_output_desc, d_conv2_output,
                conv_desc2, conv2_bwd_filter_algo_type,
                d_workspace, workspace_bytes,
                &beta,
                filter_desc2, conv2_weight_gradients
            ));

            // 第二層卷積的 bias 梯度計算
            CHECK_CUDNN(cudnnConvolutionBackwardBias(
                cudnn,
                &alpha,
                conv2_output_desc, d_conv2_output,
                &beta,
                conv2_bias_desc, conv2_bias_gradients
            ));

            // 計算第二層卷積的輸入梯度，以便將梯度反向傳遞給第一卷積
            float *d_relu1_gradients;
            cudaMalloc(&d_relu1_gradients, conv1_output_bytes);

            CHECK_CUDNN(cudnnConvolutionBackwardData(
                cudnn,
                &alpha,
                filter_desc2, conv2_weights,
                conv2_output_desc, d_conv2_output,
                conv_desc2, conv2_bwd_data_algo_type,
                d_workspace, workspace_bytes,
                &beta,
                conv1_output_desc, d_relu1_gradients
            ));

            // 第一層卷積的反向傳播
            // 反向傳播 ReLU 梯度計算
            CHECK_CUDNN(cudnnActivationBackward(
                cudnn,
                activation_desc,
                &alpha,
                conv1_output_desc, d_relu1_output,
                conv1_output_desc, d_relu1_gradients,
                conv1_output_desc, d_relu1_output,
                &beta,
                conv1_output_desc, d_conv1_output
            ));

            // 第一層卷積的 weight 梯度計算
            CHECK_CUDNN(cudnnConvolutionBackwardFilter(
                cudnn,
                &alpha,
                input_desc, d_input,
                conv1_output_desc, d_conv1_output,
                conv_desc1, conv1_bwd_filter_algo_type,
                d_workspace, workspace_bytes,
                &beta,
                filter_desc1, conv1_weight_gradients
            ));

            // 第一層卷積的 bias 梯度計算
            CHECK_CUDNN(cudnnConvolutionBackwardBias(
                cudnn,
                &alpha,
                conv1_output_desc, d_conv1_output,
                &beta,
                conv1_bias_desc, conv1_bias_gradients
            ));

            // 使用 SGD 更新模型參數
            // 定義線程數 thread 跟 block 數量
            int threads_per_block_update = 256;
            int blocks_per_grid_update;

            // 更新全連接層的 weights
            int fc_weights_size = fc_input_size * fc_output_size;
            blocks_per_grid_update = (fc_weights_size + threads_per_block_update - 1) / threads_per_block_update;
            sgd_update<<<blocks_per_grid_update, threads_per_block_update>>>(fc_weights, fc_weight_gradients, learning_rate, fc_weights_size);

            // 更新全連接層的 bias
            blocks_per_grid_update = (fc_output_size + threads_per_block_update - 1) / threads_per_block_update;
            sgd_update<<<blocks_per_grid_update, threads_per_block_update>>>(fc_bias, fc_bias_gradients, learning_rate, fc_output_size);

            // 更新第二卷積層的 weights
            int conv2_weights_size = filter_count2 * filter_count1 * filter_height2 * filter_width2;
            blocks_per_grid_update = (conv2_weights_size + threads_per_block_update - 1) / threads_per_block_update;
            sgd_update<<<blocks_per_grid_update, threads_per_block_update>>>(conv2_weights, conv2_weight_gradients, learning_rate, conv2_weights_size);

            // 更新第二卷積層的 bias
            blocks_per_grid_update = (filter_count2 + threads_per_block_update - 1) / threads_per_block_update;
            sgd_update<<<blocks_per_grid_update, threads_per_block_update>>>(conv2_bias, conv2_bias_gradients, learning_rate, filter_count2);

            // 更新第一卷積層的 weights
            int conv1_weights_size = filter_count1 * channels * filter_height1 * filter_width1;
            blocks_per_grid_update = (conv1_weights_size + threads_per_block_update - 1) / threads_per_block_update;
            sgd_update<<<blocks_per_grid_update, threads_per_block_update>>>(conv1_weights, conv1_weight_gradients, learning_rate, conv1_weights_size);

            // 更新第一卷積層的 bias
            blocks_per_grid_update = (filter_count1 + threads_per_block_update - 1) / threads_per_block_update;
            sgd_update<<<blocks_per_grid_update, threads_per_block_update>>>(conv1_bias, conv1_bias_gradients, learning_rate, filter_count1);

            // 釋放不需要用到的記憶體空間
            cudaFree(d_fc_input_gradients);
            cudaFree(d_relu1_gradients);
            cudnnDestroyTensorDescriptor(fc_output_desc);
        }

        // 進行驗證
        float epoch_val_loss = 0.0f;
        int total_val_correct = 0;

        int val_num_samples = val_images.size() / (height * width);
        int val_num_batches = val_num_samples / batch_size;

        for (int batch = 0; batch < val_num_batches; ++batch) {
            // 批次取出驗證資料集的資料
            std::vector<float> batch_images(batch_size * channels * height * width);
            std::vector<int> batch_labels(batch_size);

            for (int i = 0; i < batch_size; ++i) {
                int idx = batch * batch_size + i;
                std::copy(
                    val_images.begin() + idx * height * width,
                    val_images.begin() + (idx + 1) * height * width,
                    batch_images.begin() + i * height * width
                );
                batch_labels[i] = val_labels[idx];
            }

            // 將批次的驗證資料集複製到 GPU 上
            cudaMemcpy(d_input, batch_images.data(), input_bytes, cudaMemcpyHostToDevice);

            // 向前傳播(驗證集不需要更新模型權重，因此不需要反向傳播)
            const float alpha = 1.0f;
            const float beta = 0.0f;

            // 第一層向前卷積
            CHECK_CUDNN(cudnnConvolutionForward(
                cudnn,
                &alpha,
                input_desc, d_input,
                filter_desc1, conv1_weights,
                conv_desc1, conv1_fwd_algo_type,
                d_workspace, workspace_bytes,
                &beta,
                conv1_output_desc, d_conv1_output
            ));

            // 添加 Bias
            CHECK_CUDNN(cudnnAddTensor(
                cudnn,
                &alpha,
                conv1_bias_desc, conv1_bias,
                &alpha,
                conv1_output_desc, d_conv1_output
            ));

            // 第一層向前卷積的 Activation Function
            CHECK_CUDNN(cudnnActivationForward(
                cudnn,
                activation_desc,
                &alpha,
                conv1_output_desc, d_conv1_output,
                &beta,
                conv1_output_desc, d_relu1_output
            ));

            // 第二層向前卷積
            CHECK_CUDNN(cudnnConvolutionForward(
                cudnn,
                &alpha,
                conv1_output_desc, d_relu1_output,
                filter_desc2, conv2_weights,
                conv_desc2, conv2_fwd_algo_type,
                d_workspace, workspace_bytes,
                &beta,
                conv2_output_desc, d_conv2_output
            ));

            // 添加 Bias
            CHECK_CUDNN(cudnnAddTensor(
                cudnn,
                &alpha,
                conv2_bias_desc, conv2_bias,
                &alpha,
                conv2_output_desc, d_conv2_output
            ));

            // 第二層向前卷積的 Activation Function
            CHECK_CUDNN(cudnnActivationForward(
                cudnn,
                activation_desc,
                &alpha,
                conv2_output_desc, d_conv2_output,
                &beta,
                conv2_output_desc, d_relu2_output
            ));

            // 將卷基層的輸出展開作為全連接層的輸入，d_relu2_output 複製到 d_fc_input
            cudaMemcpy(
                d_fc_input,
                d_relu2_output,
                conv2_output_bytes,
                cudaMemcpyDeviceToDevice
            );

            // 全連接層
            // cublasSgemm : 執行全連接層的矩陣操作
            CHECK_CUBLAS(cublasSgemm(
                cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                fc_output_size, batch_size, fc_input_size,
                &alpha,
                fc_weights, fc_input_size,
                d_fc_input, fc_input_size,
                &beta,
                d_fc_output, fc_output_size
            ));

            // 添加全連接層的 Bias
            CHECK_CUBLAS(cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                fc_output_size, batch_size, 1,
                &alpha,
                fc_bias, fc_output_size,
                ones, 1,
                &alpha,
                d_fc_output, fc_output_size
            ));

            // Softmax Activation Function
            cudnnTensorDescriptor_t fc_output_desc_val;                     // 建立全連接層輸出的形狀和格式
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&fc_output_desc_val));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(
                fc_output_desc_val,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batch_size, num_classes, 1, 1
            ));

            // Softmax 向前傳播
            CHECK_CUDNN(cudnnSoftmaxForward(
                cudnn,
                CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, // 準確和操作模式
                &alpha,
                fc_output_desc_val, d_fc_output,                    // 經過 softmax 輸入
                &beta,
                fc_output_desc_val, d_fc_output                     // 經過 softmax 輸出
            ));

            // 計算損失
            std::vector<float> output(batch_size * num_classes);

            // 將最後 Softmax 輸出從 GPU 的記憶體（d_fc_output）複製到 Host 的記憶體（output）
            cudaMemcpy(output.data(), d_fc_output, fc_output_bytes, cudaMemcpyDeviceToHost);

            // 計算準確度
            int batch_correct = 0;
            for (int i = 0; i < batch_size; ++i) {
                // 找出批次樣本中預測概率最大的類別
                int predicted_label = std::distance(            //使用距離函式，與最大元素間的距離即為預測的類別標籤
                    output.begin() + i * num_classes,
                    std::max_element(
                        output.begin() + i * num_classes,       //樣本起始位置
                        output.begin() + (i + 1) * num_classes  //樣本結束位置
                    )
                );
                if (predicted_label == batch_labels[i]) {
                    batch_correct++;
                }
            }

            total_val_correct += batch_correct;

            // 使用 Cross Entropy 計算 valid loss
            float batch_loss = crossEntropyLoss(batch_labels, output, batch_size, num_classes);
            epoch_val_loss += batch_loss * batch_size;

            // 釋放記憶體
            cudnnDestroyTensorDescriptor(fc_output_desc_val);
        }
        
        // 計算整個 epoch 訓練的平均 loss 和 accuracy
        float train_accuracy = static_cast<float>(total_train_correct) / (num_batches * batch_size);
        float average_train_loss = epoch_train_loss / (num_batches * batch_size);

        // 計算整個 epoch 驗證的平均 loss 和 accuracy
        float val_accuracy = static_cast<float>(total_val_correct) / (val_num_batches * batch_size);
        float average_val_loss = epoch_val_loss / (val_num_batches * batch_size);
        
        std::cout << "\nEpoch " << epoch + 1 << " completed. "
                << "Training Loss: " << average_train_loss << ", "
                << "Training Accuracy: " << train_accuracy * 100 << "%, " 
                << "Validation Loss: " << average_val_loss << ", "
                << "Validation Accuracy: " << val_accuracy * 100 << "%" << std::endl;

    }

    // Step 9. 儲存模型的權重

    // 將訓練好的模型各層數參數從 GPU 複製回 Host
    cudaMemcpy(host_conv1_weights.data(), conv1_weights, host_conv1_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_conv1_bias.data(), conv1_bias, host_conv1_bias.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_conv2_weights.data(), conv2_weights, host_conv2_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_conv2_bias.data(), conv2_bias, host_conv2_bias.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_fc_weights.data(), fc_weights, host_fc_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_fc_bias.data(), fc_bias, host_fc_bias.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 將各層數參數儲存成二進制檔案
    saveModel("./weights/conv1_weights.bin", host_conv1_weights);
    saveModel("./weights/conv1_bias.bin", host_conv1_bias);
    saveModel("./weights/conv2_weights.bin", host_conv2_weights);
    saveModel("./weights/conv2_bias.bin", host_conv2_bias);
    saveModel("./weights/fc_weights.bin", host_fc_weights);
    saveModel("./weights/fc_bias.bin", host_fc_bias);

    // Step 10. 使用未參與訓練的資料集測試模型

    // 載入測試資料集
    std::vector<float> test_images;
    std::vector<int> test_labels;
    loadMNIST(test_images, test_labels, "./dataset/t10k-images.idx3-ubyte", "./dataset/t10k-labels.idx1-ubyte");
    normalizeImages(test_images);

    // 設定測試參數
    int test_batch_size = 100;                                      // 每個測試 batch 大小
    int num_test_samples = test_images.size() / (height * width);   // 測試的總樣本數
    int num_test_batches = num_test_samples / test_batch_size;      // 測試 batch 數量
    int total_correct = 0;                                          // 測試正確數量

    // 重新分配輸入和輸出記憶體
    // 釋放之前分配給的 GPU 進行訓練的記憶體，重新分配給測試使用。
    cudaFree(d_input);
    cudaFree(d_fc_output);
    input_bytes = test_batch_size * channels * height * width * sizeof(float);
    fc_output_bytes = test_batch_size * fc_output_size * sizeof(float);
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_fc_output, fc_output_bytes);

    // 更新輸入和輸出 Descriptor，確認 Descriptor 的 shape 與測試參數一致
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        test_batch_size, channels, 
        height, width
    ));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        conv1_output_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        test_batch_size, output_c1, 
        output_h1, output_w1
    ));
    
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        conv2_output_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        test_batch_size, output_c2, 
        output_h2, output_w2
    ));

    // 重新分配用於訓練的 GPU 記憶體給測試使用
    cudaFree(d_conv1_output);
    cudaFree(d_conv2_output);
    cudaFree(d_relu1_output);
    cudaFree(d_relu2_output);
    conv1_output_bytes = test_batch_size * output_c1 * output_h1 * output_w1 * sizeof(float);
    conv2_output_bytes = test_batch_size * output_c2 * output_h2 * output_w2 * sizeof(float);
    cudaMalloc(&d_conv1_output, conv1_output_bytes);
    cudaMalloc(&d_conv2_output, conv2_output_bytes);
    cudaMalloc(&d_relu1_output, conv1_output_bytes);
    cudaMalloc(&d_relu2_output, conv2_output_bytes);

    // 重新分配全連接層輸入的 GPU 記憶體
    cudaFree(d_fc_input);
    fc_input_bytes = test_batch_size * fc_input_size * sizeof(float);
    cudaMalloc(&d_fc_input, fc_input_bytes);

    for (int batch = 0; batch < num_test_batches; ++batch) {
        // 取得當前測試集的批次
        std::vector<float> batch_images(test_batch_size * channels * height * width);
        std::vector<int> batch_labels(test_batch_size);
        for (int i = 0; i < test_batch_size; ++i) {
            // idx 用於計算當前樣本在整個測試集的索引
            int idx = batch * test_batch_size + i;
            std::copy(
                test_images.begin() + idx * height * width,         // 第 idx 張圖片的起始位置
                test_images.begin() + (idx + 1) * height * width,   // 第 idx 張圖片的結束位置
                batch_images.begin() + i * height * width           // 儲存位置
            );
            batch_labels[i] = test_labels[idx];                     // 複製標籤
        }

        // 將批次的測試資料複製到 GPU 上
        cudaMemcpy(d_input, batch_images.data(), input_bytes, cudaMemcpyHostToDevice);

        // 向前傳播(測試不需要更新模型參數，因此只需要一次向前傳播得到最終 Softmax 結果就可)
        const float alpha = 1.0f;
        const float beta = 0.0f;
    
        // 第一層向前卷積
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn, 
            &alpha, 
            input_desc, d_input, 
            filter_desc1, conv1_weights, 
            conv_desc1, conv1_fwd_algo_type, 
            d_workspace, workspace_bytes, 
            &beta, 
            conv1_output_desc, d_conv1_output
        ));

        // 添加 Bias
        CHECK_CUDNN(cudnnAddTensor(
            cudnn, 
            &alpha, 
            conv1_bias_desc, conv1_bias, 
            &alpha, 
            conv1_output_desc, d_conv1_output
        ));

        // 第一層向前卷積的 Activation Function
        CHECK_CUDNN(cudnnActivationForward(
            cudnn, 
            activation_desc, 
            &alpha, 
            conv1_output_desc, d_conv1_output, 
            &beta, 
            conv1_output_desc, d_relu1_output
        ));
 
        // 第二層向前卷積
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn, 
            &alpha, 
            conv1_output_desc, d_relu1_output, 
            filter_desc2, conv2_weights, 
            conv_desc2, conv2_fwd_algo_type, 
            d_workspace, workspace_bytes, 
            &beta, 
            conv2_output_desc, d_conv2_output
        ));

        // 添加 Bias
        CHECK_CUDNN(cudnnAddTensor(
            cudnn, 
            &alpha, 
            conv2_bias_desc, conv2_bias, 
            &alpha, 
            conv2_output_desc, d_conv2_output
        ));

        // 第二層向前卷積的 Activation Function
        CHECK_CUDNN(cudnnActivationForward(
            cudnn, 
            activation_desc, 
            &alpha, 
            conv2_output_desc, d_conv2_output, 
            &beta, 
            conv2_output_desc, d_relu2_output
        ));

        // 將卷積層的輸出展開作為全連接層的輸入，d_relu2_output 複製到 d_fc_input
        cudaMemcpy(d_fc_input, d_relu2_output, conv2_output_bytes, cudaMemcpyDeviceToDevice);

        // 全連接層
        // cublasSgemm : 執行全連接層的矩陣操作
        CHECK_CUBLAS(cublasSgemm(
            cublas, 
            CUBLAS_OP_T, CUBLAS_OP_N, 
            fc_output_size, test_batch_size, fc_input_size, 
            &alpha, 
            fc_weights, fc_input_size, 
            d_fc_input, fc_input_size, 
            &beta, 
            d_fc_output, fc_output_size
        ));

        // 添加全連接層的 Bias
        CHECK_CUBLAS(cublasSgemm(
            cublas, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            fc_output_size, test_batch_size, 1, 
            &alpha, 
            fc_bias, fc_output_size, 
            ones, 1, 
            &alpha, 
            d_fc_output, fc_output_size
        ));

        // Softmax Activation Function
        cudnnTensorDescriptor_t fc_output_desc_test;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&fc_output_desc_test));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            fc_output_desc_test, 
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
            test_batch_size, num_classes, 
            1, 1
        ));

        // Softmax 向前傳播
        CHECK_CUDNN(cudnnSoftmaxForward(
            cudnn, 
            CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, 
            &alpha, 
            fc_output_desc_test, d_fc_output, 
            &beta, 
            fc_output_desc_test, d_fc_output
        ));

        // 計算準確度
        std::vector<float> output(test_batch_size * num_classes);
        cudaMemcpy(output.data(), d_fc_output, fc_output_bytes, cudaMemcpyDeviceToHost);

        int batch_correct = 0;
        for (int i = 0; i < test_batch_size; ++i) {
            // 找出批次樣本中預測概率最大的類別
            int predicted_label = std::distance(            // 使用距離函式，與最大元素間的距離即為預測的類別標籤
                output.begin() + i * num_classes,
                std::max_element(
                    output.begin() + i * num_classes,       // 樣本起始位置
                    output.begin() + (i + 1) * num_classes  // 樣本結束位置
                )
            );
            if (predicted_label == batch_labels[i]) {
                batch_correct++;
            }
        }
        total_correct += batch_correct;

        // 釋放記憶體
        cudnnDestroyTensorDescriptor(fc_output_desc_test);
    }

    // 計算測試的準確度
    float test_accuracy = static_cast<float>(total_correct) / num_test_samples;
    std::cout << "\nTest Accuracy: " << test_accuracy * 100 << "%" << std::endl;

    // 釋放記憶體
    cudaFree(conv1_weights);
    cudaFree(conv1_bias);
    cudaFree(conv2_weights);
    cudaFree(conv2_bias);
    cudaFree(fc_weights);
    cudaFree(fc_bias);
    cudaFree(loss_gradients);

    cudaFree(conv1_weight_gradients);
    cudaFree(conv1_bias_gradients);
    cudaFree(conv2_weight_gradients);
    cudaFree(conv2_bias_gradients);
    cudaFree(fc_weight_gradients);
    cudaFree(fc_bias_gradients);

    cudaFree(d_input);
    cudaFree(d_conv1_output);
    cudaFree(d_conv2_output);
    cudaFree(d_fc_input);
    cudaFree(d_fc_output);
    cudaFree(d_relu1_output);
    cudaFree(d_relu2_output);
    cudaFree(d_workspace);
    cudaFree(ones);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(conv1_output_desc);
    cudnnDestroyTensorDescriptor(conv2_output_desc);
    cudnnDestroyTensorDescriptor(conv1_bias_desc);
    cudnnDestroyTensorDescriptor(conv2_bias_desc);
    cudnnDestroyActivationDescriptor(activation_desc);
    cudnnDestroyFilterDescriptor(filter_desc1);
    cudnnDestroyFilterDescriptor(filter_desc2);
    cudnnDestroyConvolutionDescriptor(conv_desc1);
    cudnnDestroyConvolutionDescriptor(conv_desc2);

    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}
