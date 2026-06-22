# CPU build of gemm-optim (naive + OpenMP-tiled GEMM).
# Builds the project, runs the correctness test, and ships the benchmark as the
# entrypoint.
#   docker build -t gemm-cpu .
#   docker run --rm gemm-cpu 2048
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# GEMM_NATIVE=OFF -> portable image (no -march=native): the binary runs on any
# x86-64 CPU, not only the one that built it.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGEMM_NATIVE=OFF \
    && cmake --build build -j \
    && ctest --test-dir build --output-on-failure

ENTRYPOINT ["/app/build/bench"]
CMD ["1024"]
