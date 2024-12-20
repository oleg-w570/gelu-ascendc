#include "acl/acl.h"
#include <cstdint>
#include <cmath>
#include <random>
#include <iostream>
#include <chrono>
#include <string>

using namespace std;

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

extern void gelu_custom_do(const int block_dim, void *stream, uint8_t *input, uint8_t *output, const int vector_size, const int tile_num);

void GenerateRandomVector(float *const vector, const int size) {
    static mt19937 rng(random_device{}());
    uniform_real_distribution<float> dis(-10.0f, 10.0f);

    for (int i = 0; i < size; ++i) {
        vector[i] = dis(rng);
    }
}

float AbsoluteMaxDifference(const float *lhs, const float *rhs, const int size) {
    auto max_diff = 0.0f;

    for (int i = 0; i < size; ++i) {
        const auto diff = abs(lhs[i] - rhs[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}

void GeluSequential(const float *input, float *const output, const int size) {
    for (int i = 0; i < size; ++i) {
        const auto x = input[i];
        output[i] = 0.5f * x * (1.0f + erf(x / sqrt(2.0f)));
    }
}

void GeluAscend(const float *input, float *const output, const int size, const int block_dim, const int tile_num) {
    const auto byte_size = size * sizeof(float);
    CHECK_ACL(aclInit(nullptr));
    auto device_id = 0;
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *input_device, *output_device;
    CHECK_ACL(aclrtMalloc((void**)(&input_device), byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)(&output_device), byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(input_device, byte_size, input, byte_size, ACL_MEMCPY_HOST_TO_DEVICE));

    gelu_custom_do(block_dim, stream, input_device, output_device, size, tile_num);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(output, byte_size, output_device, byte_size, ACL_MEMCPY_DEVICE_TO_HOST));
    
    CHECK_ACL(aclrtFree(output_device));
    CHECK_ACL(aclrtFree(input_device));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device_id));
    CHECK_ACL(aclFinalize());
}


int main(int argc, char *argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <vector_size> <block_dim> <tile_num>" << endl;
        return 1;
    }
    const auto vector_size = stoi(argv[1]);
    const auto block_dim = stoi(argv[2]);
    const auto tile_num = stoi(argv[3]);

    auto input = new float[vector_size];
    auto output_seq = new float[vector_size];
    auto output_ascend = new float[vector_size];

    GenerateRandomVector(input, vector_size);

    const auto start_seq = chrono::high_resolution_clock::now();
    GeluSequential(input, output_seq, vector_size);
    const auto end_seq = chrono::high_resolution_clock::now();
    const chrono::duration<double> time_seq = end_seq - start_seq;

    const auto start_ascend = chrono::high_resolution_clock::now();
    GeluAscend(input, output_ascend, vector_size, block_dim, tile_num);
    const auto end_ascend = chrono::high_resolution_clock::now();
    const chrono::duration<double> time_ascend = end_ascend - start_ascend;

    cout << "===================================================================" << endl;
    cout << "GELU Sequential cpu execution time: " << time_seq.count() << " sec." << endl;
    cout << "GELU Ascend execution time: " << time_ascend.count() << " sec." << endl;
    cout << "Max difference: " << AbsoluteMaxDifference(output_seq, output_ascend, vector_size) << endl;
    cout << "===================================================================" << endl;
    
    delete[] output_ascend;
    delete[] output_seq;
    delete[] input;
}