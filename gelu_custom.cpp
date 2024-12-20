#include "kernel_operator.h"
#include <cstdint>

constexpr auto BUFFER_NUM = 2;

class KernelGelu {
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> output_queue;
    AscendC::GlobalTensor<float> input_global;
    AscendC::GlobalTensor<float> output_global;

    int block_size;

    int total_tiles;
    int base_tile_size;
    int tile_remainder;
    int curr_tile_size;
    int curr_tile_offset;

    __aicore__ inline void UpdateProgress(const int progress) {
        curr_tile_size = progress < tile_remainder ? base_tile_size + 1 : base_tile_size;
        curr_tile_offset = progress < tile_remainder ? progress * (base_tile_size + 1) : tile_remainder * (base_tile_size + 1) + (progress - tile_remainder) * base_tile_size; 
    }

    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> input_local = input_queue.AllocTensor<float>();
        AscendC::DataCopy(input_local, input_global[curr_tile_offset], curr_tile_size);
        input_queue.EnQue(input_local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> input_local = input_queue.DeQue<float>();
        AscendC::LocalTensor<float> output_local = output_queue.AllocTensor<float>();
       
        AscendC::Muls(output_local, input_local, 1.0f / sqrt(2.0f), curr_tile_size);
        AscendC::Erf(output_local, output_local, curr_tile_size);
        AscendC::Adds(output_local, output_local, 1.0f, curr_tile_size);
        AscendC::Muls(output_local, output_local, 0.5f, curr_tile_size);
        AscendC::Mul(output_local, output_local, input_local, curr_tile_size);

        output_queue.EnQue<float>(output_local);
        input_queue.FreeTensor(input_local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> output_local = output_queue.DeQue<float>();
        AscendC::DataCopy(output_global[curr_tile_offset], output_local, curr_tile_size);
        output_queue.FreeTensor(output_local);
    }

public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const int vector_size, const int tile_num)
    {
        const int block_n = AscendC::GetBlockNum();
        const int block_i = AscendC::GetBlockIdx();

        const int base_block_size = vector_size / block_n;
        const int remainder = vector_size % block_n;
        block_size = block_i < remainder ? base_block_size + 1 : base_block_size;
        const int offset = block_i < remainder ? block_i * (base_block_size + 1) : remainder * (base_block_size + 1) + (block_i - remainder) * base_block_size;

        // this->block_size = vector_size / block_n;
        // const int block_offset = block_i * block_size;

        input_global.SetGlobalBuffer((__gm__ float *)input + block_offset, block_size);
        output_global.SetGlobalBuffer((__gm__ float *)output + block_offset, block_size);

        this->total_tiles = tile_num * BUFFER_NUM;
        this->base_tile_size = block_size / total_tiles;
        this->tile_remainder = block_size % total_tiles;
        const int max_tile_size = tile_remainder > 0 ? base_tile_size + 1 : base_tile_size;
        const int aligned_tile_size = ((max_tile_size + 31) / 32) * 32;

        pipe.InitBuffer(input_queue, BUFFER_NUM, aligned_tile_size * sizeof(float));
        pipe.InitBuffer(output_queue, BUFFER_NUM, aligned_tile_size * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (int i = 0; i < total_tiles; ++i) {
            UpdateProgress(i);
            CopyIn();
            Compute();
            CopyOut();
        }
    }

};

extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR input, GM_ADDR output, const int vector_size, const int tile_num)
{
    KernelGelu op;
    op.Init(input, output, vector_size, tile_num);
    op.Process();
}

void gelu_custom_do(const int block_dim, void *stream, uint8_t *input, uint8_t *output, const int vector_size, const int tile_num) {
    gelu_custom<<< block_dim, nullptr, stream >>>(input, output, vector_size, tile_num);
}