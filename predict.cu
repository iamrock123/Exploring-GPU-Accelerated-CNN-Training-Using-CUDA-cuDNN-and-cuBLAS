#include <iostream>
#include <fstream>
#include <vector>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>  
#include <numeric>    
#include <opencv2/opencv.hpp>  // 用於讀取和處理圖片

#define CHECK_CUDNN(call)                                                    \
    {                                                                        \
        cudnnStatus_t status = (call);                                       \
        if (status != CUDNN_STATUS_SUCCESS) {                                \
            std::cerr << "Error on line " << __LINE__ << ": "                \
                      << cudnnGetErrorString(status) << std::endl;           \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    }

#define CHECK_CUBLAS(call)                                                   \
    {                                                                        \
        cublasStatus_t status = (call);                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "CUBLAS Error on line " << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    }

// 載入訓練好的模型參數
void loadModel(const std::string& filename, std::vector<float>& data) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Failed to open file for loading model: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    inFile.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    inFile.close();
}

int main(int argc, char** argv) {
    
    // 檢查輸入參數
    if (argc < 2) {
        std::cerr << "Correct Use: " << argv[0] << " <Image File Path> " << std::endl;
        return 1;
    }
    std::string image_file = argv[1];

    // Step 1. 初始化和資源分配

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    // Step 2. 模型參數初始化

    // 定義 CNN 神經網路參數
    const int batch_size = 1;  // 使用單張圖片進行預測
    const int channels = 1;
    const int height = 28;
    const int width = 28;
    const int num_classes = 10;

    // Step 3. 建立模型結構
    // 輸入、輸出、卷積、全連接層和 Activation Function 的 Descriptor

    // 設定模型輸入及輸出的 Tensor Descriptor
    // input_desc 模型輸入
    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        batch_size, channels, 
        height, width
    ));

    // 第一層 convolutional layer 的參數
    // filter_desc1 用於設定第一層 convolutional layer 的 filter
    cudnnFilterDescriptor_t filter_desc1;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc1));

    // 定義 32 個 5 * 5 的 filters，每個 filter 的 channel = 1
    const int filter_count1 = 32;
    const int filter_height1 = 5;
    const int filter_width1 = 5;
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
        input_desc, filter_desc1, 
        &output_n1, &output_c1, 
        &output_h1, &output_w1
    ));

    // conv1_output_desc 用於計算卷積後的操作輸出 Tensor
    cudnnTensorDescriptor_t conv1_output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv1_output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        conv1_output_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        batch_size, output_c1, 
        output_h1, output_w1
    ));

    // 建立 conv1 Bias 的 Tensor Descriptor
    cudnnTensorDescriptor_t conv1_bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv1_bias_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        conv1_bias_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        1, filter_count1, 
        1, 1
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
        pad_h2, pad_w2, 
        stride_h2, stride_w2, 
        1, 1, 
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
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
        batch_size, output_c2, 
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
        CUDNN_ACTIVATION_RELU, // 指定要使用的 Activation Function
        CUDNN_PROPAGATE_NAN,   // 如何處理 NaN 值 (忽略 NaN)
        0.0                    // Activation Function 數值上限 (ReLU不需要)
    ));

    // Step 4. 分配 GPU 記憶體

    // 全連接層的輸入和輸出大小
    const int fc_input_size = output_c2 * output_h2 * output_w2;
    const int fc_output_size = num_classes;

    // 模型参数
    float *conv1_weights, *conv1_bias, *conv2_weights, *conv2_bias; // 第一、第二層卷積的 Weights 和 Bias
    float *fc_weights, *fc_bias;                                    // 全連接層的 Weights 和 Bias

    // 為参数分配記憶體
    cudaMalloc(&conv1_weights, filter_count1 * channels * filter_height1 * filter_width1 * sizeof(float));
    cudaMalloc(&conv1_bias, filter_count1 * sizeof(float));
    cudaMalloc(&conv2_weights, filter_count2 * filter_count1 * filter_height2 * filter_width2 * sizeof(float));
    cudaMalloc(&conv2_bias, filter_count2 * sizeof(float));
    cudaMalloc(&fc_weights, fc_input_size * fc_output_size * sizeof(float));
    cudaMalloc(&fc_bias, fc_output_size * sizeof(float));

    // 載入訓練好的模型權重
    std::vector<float> host_conv1_weights(filter_count1 * channels * filter_height1 * filter_width1);
    std::vector<float> host_conv1_bias(filter_count1);
    std::vector<float> host_conv2_weights(filter_count2 * filter_count1 * filter_height2 * filter_width2);
    std::vector<float> host_conv2_bias(filter_count2);
    std::vector<float> host_fc_weights(fc_input_size * fc_output_size);
    std::vector<float> host_fc_bias(fc_output_size);

    loadModel("./weights/conv1_weights.bin", host_conv1_weights);
    loadModel("./weights/conv1_bias.bin", host_conv1_bias);
    loadModel("./weights/conv2_weights.bin", host_conv2_weights);
    loadModel("./weights/conv2_bias.bin", host_conv2_bias);
    loadModel("./weights/fc_weights.bin", host_fc_weights);
    loadModel("./weights/fc_bias.bin", host_fc_bias);

    // 將模型參數複製到 GPU
    cudaMemcpy(conv1_weights, host_conv1_weights.data(), host_conv1_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv1_bias, host_conv1_bias.data(), host_conv1_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv2_weights, host_conv2_weights.data(), host_conv2_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv2_bias, host_conv2_bias.data(), host_conv2_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_weights, host_fc_weights.data(), host_fc_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_bias, host_fc_bias.data(), host_fc_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 分配記憶體給在向前和反向傳播的過程中，儲存各層中間的輸出，以便後續的計算

    // 計算每個層數輸入輸出所需要的記憶體大小
    size_t input_bytes = batch_size * channels * height * width * sizeof(float);
    size_t conv1_output_bytes = batch_size * output_c1 * output_h1 * output_w1 * sizeof(float);
    size_t conv2_output_bytes = batch_size * output_c2 * output_h2 * output_w2 * sizeof(float);
    size_t fc_input_bytes = batch_size * fc_input_size * sizeof(float);
    size_t fc_output_bytes = batch_size * fc_output_size * sizeof(float);

    // 在 GPU 上分配記憶體來儲存輸入、卷積層輸出和全連接層輸出等
    float *d_input, *d_conv1_output, *d_conv2_output, *d_fc_input, *d_fc_output;
    float *d_relu1_output, *d_relu2_output;

    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_conv1_output, conv1_output_bytes);
    cudaMalloc(&d_conv2_output, conv2_output_bytes);
    cudaMalloc(&d_fc_input, fc_input_bytes);
    cudaMalloc(&d_fc_output, fc_output_bytes);

    // 分配 ReLU 的輸出，用於反向傳播
    cudaMalloc(&d_relu1_output, conv1_output_bytes);
    cudaMalloc(&d_relu2_output, conv2_output_bytes);

    // Step 5. 實現向前傳播

    // 分配 workspace 的記憶體
    // 在 Device 中，一個計算所需的空間必須事前聲明
    // 除了輸入輸出之外，進行這個計算所需的額外 “工作空間”，也可以簡單地理解為空間複雜度。

    size_t workspace_bytes = 0;
    void *d_workspace = nullptr;

    // 獲取向前傳播所需空間的大小
    size_t fwd_workspace_size1 = 0, fwd_workspace_size2 = 0;

    // 選擇第一層向前卷積的算法
    cudnnConvolutionFwdAlgo_t conv1_fwd_algo_type;
    cudnnConvolutionFwdAlgoPerf_t conv1_fwd_algo_perf;
    int conv1_fwd_returned_algo_count;
    
    // 選擇最佳的向前卷積算法 
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        input_desc,filter_desc1,
        conv_desc1,conv1_output_desc,
        1,
        &conv1_fwd_returned_algo_count,
        &conv1_fwd_algo_perf
    ));

    // 使用 conv1_fwd_algo_type 提取選擇的算法
    conv1_fwd_algo_type = conv1_fwd_algo_perf.algo;

    // 獲取選定算法所需的工作空間大小
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, 
        input_desc, filter_desc1, 
        conv_desc1, conv1_output_desc, 
        conv1_fwd_algo_type, &fwd_workspace_size1
    ));

    // 選擇第二層向前卷積的算法
    cudnnConvolutionFwdAlgo_t conv2_fwd_algo_type;
    cudnnConvolutionFwdAlgoPerf_t conv2_fwd_algo_perf;
    int conv2_fwd_returned_algo_count;
    
    // 選擇最佳的向前卷積算法 
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        conv1_output_desc, filter_desc2,
        conv_desc2, conv2_output_desc,
        1,
        &conv2_fwd_returned_algo_count,
        &conv2_fwd_algo_perf));
    
    // 使用 conv2_fwd_algo_type 提取選擇的算法
    conv2_fwd_algo_type = conv2_fwd_algo_perf.algo;

    // 獲取選定算法所需的工作空間大小
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, 
        conv1_output_desc, filter_desc2, 
        conv_desc2, conv2_output_desc, 
        conv2_fwd_algo_type, &fwd_workspace_size2
    ));

    // 找出需要最大的 workspace
    workspace_bytes = std::max(fwd_workspace_size1, fwd_workspace_size2);

    // 分配 GPU 記憶體 (workspace_bytes 不可超過 GPU 最大記憶體)
    cudaMalloc(&d_workspace, workspace_bytes);

    // Step 6. 使用圖片進行預測

    // 使用 OpenCV 讀取圖片
    cv::Mat img = cv::imread(image_file, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error! Can not Load the Image Correctly : " << image_file << std::endl;
        return -1;
    }
    // 將輸入的圖片調整為28*28的大小
    cv::resize(img, img, cv::Size(28, 28));
    
    // 將圖片轉為浮點數並進行正規化
    img.convertTo(img, CV_32FC1, 1.0 / 255.0);

    // 將正規化後的資料轉為 NCHW 格式
    std::vector<float> input_image(channels * height * width);
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                input_image[c * height * width + h * width + w] = img.at<float>(h, w);
            }
        }
    }

    // 將預測圖片複製到 GPU
    cudaMemcpy(d_input, input_image.data(), input_bytes, cudaMemcpyHostToDevice);

    // 向前傳播(預測不需要反向傳播)
    // Output = alpha * Operation(Input) + beta * Output
    const float alpha = 1.0f; // 縮放操作的輸出
    const float beta = 0.0f;  // 縮放現有的輸出

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
    cudaMemcpy(d_fc_input, d_relu2_output, conv2_output_bytes, cudaMemcpyDeviceToDevice);

    // 全連接層
    // cublasSgemm : 執行全連接層的矩陣操作
    CHECK_CUBLAS(cublasSgemv(
        cublas, 
        CUBLAS_OP_T, 
        fc_input_size, fc_output_size, 
        &alpha, 
        fc_weights, fc_input_size, 
        d_fc_input, 1, 
        &beta, 
        d_fc_output, 1
    ));

    // 添加全連接層的 Bias
    CHECK_CUBLAS(cublasSaxpy(
        cublas, 
        fc_output_size, 
        &alpha, 
        fc_bias, 1, 
        d_fc_output, 1
    ));

    // Softmax Activation Function
    cudnnTensorDescriptor_t fc_output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&fc_output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        fc_output_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        batch_size, num_classes, 
        1, 1
    ));

    // Softmax 向前傳播
    CHECK_CUDNN(cudnnSoftmaxForward(
        cudnn, 
        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, 
        &alpha, 
        fc_output_desc, d_fc_output, 
        &beta, 
        fc_output_desc, d_fc_output
    ));

    // 取得輸出
    std::vector<float> output(num_classes);
    cudaMemcpy(output.data(), d_fc_output, fc_output_bytes, cudaMemcpyDeviceToHost);

    // 輸出 Softmax 的概率分布
    std::cout << "Predict Result: \n" << std::endl;
    std::cout << "Softmax Output:" << std::endl;
    for (int i = 0; i < num_classes; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // 輸出預測最大值
    int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    std::cout << "\nPredicted Label: " << predicted_label << std::endl;

    // 釋放記憶體
    cudaFree(conv1_weights);
    cudaFree(conv1_bias);
    cudaFree(conv2_weights);
    cudaFree(conv2_bias);
    cudaFree(fc_weights);
    cudaFree(fc_bias);

    cudaFree(d_input);
    cudaFree(d_conv1_output);
    cudaFree(d_conv2_output);
    cudaFree(d_fc_input);
    cudaFree(d_fc_output);
    cudaFree(d_relu1_output);
    cudaFree(d_relu2_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(conv1_output_desc);
    cudnnDestroyTensorDescriptor(conv2_output_desc);
    cudnnDestroyTensorDescriptor(conv1_bias_desc);
    cudnnDestroyTensorDescriptor(conv2_bias_desc);
    cudnnDestroyTensorDescriptor(fc_output_desc);
    cudnnDestroyActivationDescriptor(activation_desc);
    cudnnDestroyFilterDescriptor(filter_desc1);
    cudnnDestroyFilterDescriptor(filter_desc2);
    cudnnDestroyConvolutionDescriptor(conv_desc1);
    cudnnDestroyConvolutionDescriptor(conv_desc2);

    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}
