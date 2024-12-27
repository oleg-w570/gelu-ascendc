#include "kernel_operator.h"
#include <cstdint>

constexpr int BUFFER_NUM = 2;
constexpr int TILE_SIZE = 256 / sizeof(float);

class KernelGelu {
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> output_queue;
    AscendC::GlobalTensor<float> input_global;
    AscendC::GlobalTensor<float> output_global;
    int block_size;

    __aicore__ inline void CopyIn(const int progress)
    {
        AscendC::LocalTensor<float> input_local = input_queue.AllocTensor<float>();
        AscendC::DataCopy(input_local, input_global[progress * TILE_SIZE], TILE_SIZE);
        input_queue.EnQue(input_local);
    }
    __aicore__ inline void Compute(const int progress)
    {
        AscendC::LocalTensor<float> input_local = input_queue.DeQue<float>();
        AscendC::LocalTensor<float> output_local = output_queue.AllocTensor<float>();
       
        AscendC::Muls(output_local, input_local, 1.0f / sqrt(2.0f), TILE_SIZE);
        AscendC::Erf(output_local, output_local, TILE_SIZE);
        AscendC::Adds(output_local, output_local, 1.0f, TILE_SIZE);
        AscendC::Muls(output_local, output_local, 0.5f, TILE_SIZE);
        AscendC::Mul(output_local, output_local, input_local, TILE_SIZE);

        output_queue.EnQue<float>(output_local);
        input_queue.FreeTensor(input_local);
    }
    __aicore__ inline void CopyOut(const int progress)
    {
        AscendC::LocalTensor<float> output_local = output_queue.DeQue<float>();
        AscendC::DataCopy(output_global[progress * TILE_SIZE], output_local, TILE_SIZE);
        output_queue.FreeTensor(output_local);
    }

public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const int size)
    {
        this->block_size = size / AscendC::GetBlockNum();

        input_global.SetGlobalBuffer((__gm__ float *)input + block_size * AscendC::GetBlockIdx(), block_size);
        output_global.SetGlobalBuffer((__gm__ float *)output + block_size * AscendC::GetBlockIdx(), block_size);

        pipe.InitBuffer(input_queue, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(output_queue, BUFFER_NUM, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int loop_count = block_size / TILE_SIZE;
        for (int i = 0; i < loop_count; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

};

extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR input, GM_ADDR output, const int size)
{
    KernelGelu op;
    op.Init(input, output, size);
    op.Process();
}

void gelu_custom_do(const int block_dim, void *stream, uint8_t *input, uint8_t *output, const int size) {
    gelu_custom<<< block_dim, nullptr, stream >>>(input, output, size);
}