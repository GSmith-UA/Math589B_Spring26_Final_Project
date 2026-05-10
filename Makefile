TARGET   = solver
NVCC     = nvcc

# V100 = sm_70. Override with: make ARCH=sm_80
ARCH     ?= sm_70

# Eigen: system default, override with: make EIGEN=/path/to/eigen3
EIGEN    ?= /usr/include/eigen3

# gencode sm_70: native V100 binary
# gencode compute_60/code=compute_60: PTX fallback JIT-compiled for any sm_60+ GPU
NVCCFLAGS = -O3 -std=c++17 -DUSE_GPU \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_60,code=compute_60 \
            -I$(EIGEN) -Isrc

SRC = src/main.cu          \
      src/solver.cu        \
      src/shoot_gpu.cu     \
      src/dynamics.cpp     \
      src/rk4.cpp          \
      src/lqr.cpp          \
      src/utils.cpp        \
      src/manifold.cpp     \
      src/continuation.cpp \
      src/query.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

# CPU-only build for local testing (no nvcc required)
cpu: NVCCFLAGS = -O3 -std=c++17 -I$(EIGEN) -Isrc
cpu: NVCC = g++
cpu: SRC = src/main.cu src/solver.cu src/dynamics.cpp src/rk4.cpp \
           src/lqr.cpp src/utils.cpp src/manifold.cpp \
           src/continuation.cpp src/query.cpp
cpu:
	$(NVCC) $(NVCCFLAGS) -x c++ $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
