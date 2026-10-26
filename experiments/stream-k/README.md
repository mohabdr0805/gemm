# Stream-K

v7 with the tiles of its last wave split in K across one block per SM. It stays
out of the library because it needs a workspace and a second kernel on a second
stream.

|              | geometric mean, 25 sizes | above cuBLAS | n=4096     |
|--------------|--------------------------|--------------|------------|
| v7           | 102.1%                   | 14 of 25     | 98.5%      |
| **Stream-K** | **106.7%**               | **21 of 25** | **103.0%** |

Default clocks, mean of two runs, 12 alternated rounds per size, both kernels in
the same run. Each size is first checked against cuBLAS three times, on an
output filled with NaN.

## Run

From the repo root, on an `sm_80` or newer card:

```
nvcc -std=c++17 -O3 -arch=sm_86 -Iinclude -Isrc experiments/stream-k/streamk.cu -lcublas -o streamk
./streamk      # ./streamk 0 checks without timing
```
