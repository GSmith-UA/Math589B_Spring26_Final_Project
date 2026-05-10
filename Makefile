TARGET   = solver
NVCC     = nvcc

# V100 = sm_70. Override with: make ARCH=sm_80
ARCH     ?= sm_70

# Eigen: system default, override with: make EIGEN=/path/to/eigen3
EIGEN    ?= /usr/include/eigen3

NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -I$(EIGEN) -Isrc

SRC = src/main.cu      \
      src/solver.cu    \
      src/dynamics.cpp \
      src/rk4.cpp      \
      src/lqr.cpp      \
      src/utils.cpp    \
      src/manifold.cpp \
      src/continuation.cpp \
      src/query.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
