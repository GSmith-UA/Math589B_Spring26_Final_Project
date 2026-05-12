TARGET = solver
NVCC   = nvcc

# Auto-detect Eigen3.
# Checks (in order): module env vars (EasyBuild/common HPC), system paths, pkg-config.
# Override: make EIGEN=/your/path
ifndef EIGEN
EIGEN := $(shell \
  for d in \
      "$${EBROOTEIGETN}" \
      "$${EIGEN_ROOT}" \
      "$${EIGEN_DIR}" \
      "$${EIGENDIR}" \
      /usr/include/eigen3 \
      /usr/local/include/eigen3 \
      /opt/local/include/eigen3; \
  do \
      [ -n "$$d" ] && [ -f "$$d/Eigen/Dense" ] && echo "$$d" && exit 0; \
  done; \
  pkg-config --variable=includedir eigen3 2>/dev/null)
endif
# Final fallback: bundled headers in third_party/eigen3
ifeq ($(EIGEN),)
EIGEN := third_party/eigen3
endif

# gencode sm_70: native V100 binary
# gencode compute_60/code=compute_60: PTX fallback JIT-compiled for any sm_60+ GPU
NVCCFLAGS = -O3 -std=c++17 -DUSE_GPU \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_60,code=compute_60 \
            -I$(EIGEN) -Isrc

SRC = src/main.cu \
      src/solver.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
