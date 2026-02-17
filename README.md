# TRUCO: Threaded Raster Unrestricted Contour Ownership

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research-orange.svg?style=flat-square)

**A lock-free, high-performance parallel algorithm for extracting geometric contours from binary images.**

> **TRUCO** outperforms the industry-standard Suzuki-Abe algorithm (OpenCV `findContours`) by **over 10x** on multi-core CPUs.

## 🚀 Why TRUCO?

Standard contour extraction algorithms (like Suzuki-Abe) are sequential and rely on expensive 32-bit labeling to build topological hierarchies. In modern computer vision—especially for 4K/high-res inputs—you often just need the *geometry* (the points) fast, without the hierarchy overhead.

**TRUCO** solves this by:
* **Going Parallel:** Leverages multi-core architectures effectively using a row-based domain decomposition.
* **Lock-Free Execution:** Uses "benign race conditions" where threads can safely write to the same pixels without mutexes or atomic instructions.
* **Start-Point Ownership:** A unique logic where a contour belongs strictly to the thread that finds its starting pixel, preventing double-counting without complex stitching.
* **Memory Efficiency:** Operates on an **8-bit state space** (vs. standard 32-bit), reducing memory bandwidth pressure and increasing SIMD throughput.

---

## 📊 Performance

TRUCO has been benchmarked against OpenCV's `findContours` (Suzuki-Abe) and Run-Length Encoding (RLE) methods.

### Key Benchmarks
* **Single-Threaded:** Even without parallelism, TRUCO is **~1.8x faster** than OpenCV due to 8-bit optimizations.
* **Multi-Threaded (Consumer CPU):** On an Intel i7 (14 cores), TRUCO achieves up to **10x speedup** on 16MP real-world images.
* **Multi-Threaded (Server CPU):** On an Intel Xeon, it also maintains robust scaling, approximately **10x speedup**.

### Scalability Analysis
TRUCO scales strongly with thread count on complex images.
| Speedup vs. Resolution | Scalability by Complexity |
| :---: | :---: |
| <img src="images/real-combined_speedup_analysis.png" width="100%"> | <img src="images/real-scalability_analysis.png" width="100%"> |
| *Figure 1: Speedup factors relative to Suzuki-Abe.* | *Figure 2: Scaling stratified by image complexity.* |
---

## 🛠 Installation & Usage

### TRUCO is designed to be easily integrated into existing OpenCV-based C++ projects.

### Dependencies
* OpenCV 4.12+
* C++17 compliant compiler

### Basic Usage

Simply include the header and use `findTRUContours` just like you would use standard OpenCV functions.

```cpp
#include "findtrucontour.h"
#include <opencv2/opencv.hpp>

int main() {
    // 1. Load your binary image (Must be CV_8UC1)
    cv::Mat binaryImage = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
    
    // 2. Prepare output vector
    std::vector<std::vector<cv::Point>> contours;
    
    // 3. Run TRUCO
    // Arguments: Input Image, Output Contours, Min Points (optional), Threads (optional)
    cv::findTRUContours(binaryImage, contours, 0, 0);
    
    // 4. Draw or process results
    cv::Mat debug;
    cv::cvtColor(binaryImage, debug, cv::COLOR_GRAY2BGR);
    cv::drawContours(debug, contours, -1, cv::Scalar(0, 0, 255), 2);
    
    return 0;
}
```

##  🧠 Algorithm Details
The "Speculative Tracing" Model
Traditional parallel contour extraction requires complex "stitching" of image tiles. TRUCO avoids this by allowing threads to speculatively trace contours downwards into neighboring regions.

Trace Down: A thread can trace a contour anywhere below its starting row.

Abort Up: If a trace moves upwards into a previous thread's region, it aborts immediately. This ensures the contour is only processed by the "owner" thread above.

Benign Races: Threads mark pixels as VISITED. If two threads mark the same pixel, the result is identical, so no locks are needed.

## 📄 Citation
If you use TRUCO in your research, please cite our paper:
```
@article{truco2026,
  title={TRUCO: Threaded Raster Unrestricted Contour Ownership},
  author={Muñoz-Salinas, Rafael and Romero-Ramirez, Francisco J. and Marín-Jiménez, Manuel J.},
  journal={To be published},
  year={2026}
}
```
## 🧠 Reproducing Benchmarks

This repository includes a performance testing tool (test_trucontour.cpp).

### 🏗 Build
```
bash
mkdir build && cd build
cmake ..
make
```

### 🏃 Run
To strictly reproduce the paper results (preventing thermal throttling), run:

```
bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
sudo cpupower frequency-set -u 3500MHz
./test_trucontour <path_to_image_directory> [-show] [-scale 0.5]
```
[Download the Image Dataset on Zenodo](https://zenodo.org/records/18667188?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY0NWJmOWQ4LWQxZmQtNDY0OC1iMjEwLWZkMzFhYzViMGJkYyIsImRhdGEiOnt9LCJyYW5kb20iOiIxYjI5YWFlOTRlOGNhYTFmMzIzN2ZhZjdiNTc3OWRiYSJ9.bjgkyDaWsBMeTqrotIlBdyQ_65cKygzA5uOqlEpsATtuXnxtFDLXKZ7zEG_a2mxcghlplnMh2c0B8n_To7i38w)


## 👥 Authors
- Rafael Muñoz-Salinas - University of Córdoba
- Francisco J. Romero-Ramirez - University of Córdoba
- Manuel J. Marín-Jiménez - University of Córdoba
