# Makefile

# 編譯器
NVCC = nvcc

# 編譯選項
CFLAGS = -O2 -std=c++11

# 庫
LIBS = -lcublas -lcudnn

# OpenCV的編譯和鏈接選項
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# 目標可執行文件
TARGET1 = train
TARGET2 = predict

# 源文件
SRC1 = main.cu utils.cu
SRC2 = predict.cu

# 物件文件
OBJ1 = $(SRC1:.cu=.o)
OBJ2 = $(SRC2:.cu=.o)

# 預設目標
all: $(TARGET1) $(TARGET2)

# 編譯 train
$(TARGET1): $(SRC1)
	$(NVCC) $(CFLAGS) $(SRC1) -o $(TARGET1) $(LIBS)

# 編譯 predict
$(TARGET2): $(SRC2)
	$(NVCC) $(CFLAGS) $(SRC2) -o $(TARGET2) $(OPENCV_CFLAGS) $(OPENCV_LIBS) -lcublas -lcudnn

# 清理編譯生成的文件
clean:
	rm -f $(TARGET1) $(TARGET2) *.o
