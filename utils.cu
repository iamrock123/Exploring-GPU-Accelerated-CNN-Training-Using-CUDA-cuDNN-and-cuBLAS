// utils.cu
#include "utils.cuh"

// Kernel 函式定義

// 用於快速初始化陣列值
__global__ void fill_array(float* arr, int size, float value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // 計算當前線程的索引
    if (idx < size) {
        arr[idx] = value;
    }
}

// 使用 SGD 進行參數更新
__global__ void sgd_update(float* params, float* grads, float learning_rate, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // 計算當前線程的索引
    if (idx < size) {
        params[idx] -= learning_rate * grads[idx];
    }
}

// Host 函式定義

// 載入 MNIST 的二進制格式資料集
void loadMNIST(std::vector<float> &images, std::vector<int> &labels, const std::string &imageFile, const std::string &labelFile) {
    // 使用 ifstream 以二進制的方式打開圖片以及標籤
    std::ifstream imgFile(imageFile, std::ios::binary);
    std::ifstream lblFile(labelFile, std::ios::binary);

    if (!imgFile.is_open() || !lblFile.is_open()) {
        std::cerr << "Failed to open files!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int magic_number_img = 0; // 用於辨認文件類型(圖片)
    int number_of_images = 0; // 文件中儲存的圖片總數
    int n_rows = 0; // 圖片高度
    int n_cols = 0; // 圖片寬度

    // 載入圖片文件的 Header 訊息
    imgFile.read(reinterpret_cast<char*>(&magic_number_img), sizeof(magic_number_img));
    magic_number_img = __builtin_bswap32(magic_number_img); // 字節順序轉換

    imgFile.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    number_of_images = __builtin_bswap32(number_of_images);

    imgFile.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    n_rows = __builtin_bswap32(n_rows);

    imgFile.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
    n_cols = __builtin_bswap32(n_cols);

    int magic_number_lbl = 0; // 用於辨認文件類型(標籤)
    int number_of_labels = 0; // 文件中儲存的標籤總數

    // 載入標籤文件的 Header 訊息
    lblFile.read(reinterpret_cast<char*>(&magic_number_lbl), sizeof(magic_number_lbl));
    magic_number_lbl = __builtin_bswap32(magic_number_lbl);

    lblFile.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
    number_of_labels = __builtin_bswap32(number_of_labels);

    // 檢查圖片與標籤的總數是否相同
    if (number_of_images != number_of_labels) {
        std::cerr << "Number of images does not match number of labels!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 使用 vector 讀取圖片與標籤
    images.resize(number_of_images * n_rows * n_cols); // 一維陣列
    labels.resize(number_of_labels);

    for (int i = 0; i < number_of_images; ++i) {
        for (int j = 0; j < n_rows * n_cols; ++j) {
            unsigned char temp = 0;
            imgFile.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            images[i * n_rows * n_cols + j] = static_cast<float>(temp);
        }
        unsigned char temp_label = 0;
        lblFile.read(reinterpret_cast<char*>(&temp_label), sizeof(temp_label));
        labels[i] = static_cast<int>(temp_label);
    }
}

// 正規化圖片
void normalizeImages(std::vector<float> &images) {
    for (auto &pixel : images) {
        pixel /= 255.0f;  // 將像素值從 [0~255] 正規化成 [0~1]
    }
}

// Cross Entropy 損失函數
float crossEntropyLoss(const std::vector<int> &labels, const std::vector<float> &predictions, int batch_size, int num_classes) {
    float loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        // 使用對數損失計算 Loss
        loss -= std::log(std::max(predictions[i * num_classes + labels[i]], 1e-7f)); // 避免 log(0)
    }
    return loss / batch_size;
}

// 計算梯度
void computeGradients(const float* output_data, const std::vector<int>& labels, float* loss_gradients, int batch_size, int num_classes) {
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            loss_gradients[i * num_classes + j] = output_data[i * num_classes + j];
            if (j == labels[i]) {
                loss_gradients[i * num_classes + j] -= 1.0f;  // 正確類別的梯度
            }
        }
    }
}

// 保存模型参数
void saveModel(const std::string& filename, const std::vector<float>& data) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for saving model: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    outFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    outFile.close();
}

// 隨機打亂資料
void shuffleData(std::vector<float> &images, std::vector<int> &labels) {
    // 確保 images 和 labels 的大小匹配
    if (images.size() / (28 * 28) != labels.size()) {
        std::cerr << "Mismatch between images and labels sizes!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int num_samples = labels.size();
    std::vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }

    // 使用隨機數引擎打亂索引
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // 根據打亂的索引重新排列 images 和 labels
    std::vector<float> shuffled_images(num_samples * 28 * 28);
    std::vector<int> shuffled_labels(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        int idx = indices[i];
        std::copy(
            images.begin() + idx * 28 * 28,
            images.begin() + (idx + 1) * 28 * 28,
            shuffled_images.begin() + i * 28 * 28
        );
        shuffled_labels[i] = labels[idx];
    }

    images = std::move(shuffled_images);
    labels = std::move(shuffled_labels);
}

// 分割資料集為訓練集和驗證集
void splitData(const std::vector<float> &images, const std::vector<int> &labels,
               std::vector<float> &train_images, std::vector<int> &train_labels,
               std::vector<float> &val_images, std::vector<int> &val_labels,
               float train_ratio) {
    int num_samples = labels.size();
    int train_samples = static_cast<int>(num_samples * train_ratio);

    train_images.assign(images.begin(), images.begin() + train_samples * 28 * 28);
    train_labels.assign(labels.begin(), labels.begin() + train_samples);

    val_images.assign(images.begin() + train_samples * 28 * 28, images.end());
    val_labels.assign(labels.begin() + train_samples, labels.end());
}
