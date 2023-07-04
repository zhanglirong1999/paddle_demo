/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example matmul.cpp
/// > Annotated version: @ref matmul_example_cpp
///
/// @page matmul_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page matmul_example_cpp Matmul Primitive Example
/// @copydetails matmul_example_cpp_short
///
/// @include matmul.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

#include <chrono>
using namespace std::chrono;

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void mlp_example(dnnl::engine::kind engine_kind) {
    static int warm_up = 10;
    static int iter_num = 10;

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions. This is too big, cause munmap_chunk()
    // const memory::dim MB = 3, // batch size
    //         M = 128, K = 128, HIDDEN_SIZE = 256, OUTPUT_SIZE = 512;

    const memory::dim MB = 1, // batch size
            M = 12800, K = 128, HIDDEN_SIZE = 64, OUTPUT_SIZE = 32;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {MB, M, K};
    memory::dims weights1_dims = {MB, K, HIDDEN_SIZE}; // M*K and K*HIDDEN_SIZE
    memory::dims bias1_dims = {1, 1, HIDDEN_SIZE};
    memory::dims hidden_dims = {MB, M, HIDDEN_SIZE};

    memory::dims weights2_dims = {MB, HIDDEN_SIZE, OUTPUT_SIZE};
    memory::dims bias2_dims = {1, 1, OUTPUT_SIZE};
    memory::dims dst_dims = {MB, M, OUTPUT_SIZE};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights1_data(product(weights1_dims));
    std::vector<float> bias1_data(product(bias1_dims));
    std::vector<float> hidden_data(product(hidden_dims));
    
    std::vector<float> weights2_data(product(weights2_dims));
    std::vector<float> bias2_data(product(bias2_dims));
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights1, bias1, weights2, bias2
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights1_data.begin(), weights1_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias1_data.begin(), bias1_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    std::generate(weights2_data.begin(), weights2_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias2_data.begin(), bias2_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory descriptors and memory objects for src, weights, bias, and
    // hidden.
    auto src_md = memory::desc(src_dims, dt::f32, tag::abc);
    auto weights1_md = memory::desc(weights1_dims, dt::f32, tag::abc);
    auto bias1_md = memory::desc(bias1_dims, dt::f32, tag::abc);
    auto hidden_md = memory::desc(hidden_dims, dt::f32, tag::abc);
    auto weights2_md = memory::desc(weights2_dims, dt::f32, tag::abc);
    auto bias2_md = memory::desc(bias2_dims, dt::f32, tag::abc);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::abc);

    auto src_mem = memory(src_md, engine);
    auto weights1_mem = memory(weights1_md, engine);
    auto bias1_mem = memory(bias1_md, engine);
    auto hidden_mem = memory(hidden_md, engine);
    auto weights2_mem = memory(weights2_md, engine);
    auto bias2_mem = memory(bias2_md, engine);
    auto dst_mem = memory(dst_md, engine);

    // Write data to memory object's handles.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights1_data.data(), weights1_mem);
    write_to_dnnl_memory(bias1_data.data(), bias1_mem);
    write_to_dnnl_memory(weights2_data.data(), weights2_mem);
    write_to_dnnl_memory(bias2_data.data(), bias2_mem);

    // Create primitive non-post-ops (ReLU).
    const float alpha = 1.2;
    const float beta = 0.f;
    // post_ops matmul_ops;
    // matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    // primitive_attr matmul_attr;
    // matmul_attr.set_post_ops(matmul_ops);

    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(
            engine, src_md, weights1_md, bias1_md, hidden_md);

    // Create the primitive.
    auto matmul_fc1 = matmul(matmul_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights1_mem});
    matmul_args.insert({DNNL_ARG_BIAS, bias1_mem});
    matmul_args.insert({DNNL_ARG_DST, hidden_mem});


    // ReLU: max(alpha * x, x) + beta
    auto relu_pd = eltwise_forward::primitive_desc(engine,
                prop_kind::forward_inference, algorithm::eltwise_relu, hidden_md, hidden_md,
                alpha, beta);
    auto relu = eltwise_forward(relu_pd);

    // second layer mutmul
    auto matmul_pd2 = matmul::primitive_desc(
            engine, hidden_md, weights2_md, bias2_md, dst_md);

    auto matmul_fc2 = matmul(matmul_pd2);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args2;
    matmul_args2.insert({DNNL_ARG_SRC, hidden_mem});
    matmul_args2.insert({DNNL_ARG_WEIGHTS, weights2_mem});
    matmul_args2.insert({DNNL_ARG_BIAS, bias2_mem});
    matmul_args2.insert({DNNL_ARG_DST, dst_mem});

    // second layer relu
    // ReLU: max(alpha * x, x) + beta
    const float alpha2 = 0.7;
    const float beta2 = 0.f;
    auto relu_pd2 = eltwise_forward::primitive_desc(engine,
                prop_kind::forward_inference, algorithm::eltwise_relu, dst_md, dst_md,
                alpha2, beta2);
    auto relu2 = eltwise_forward(relu_pd2);

    // warm up
    for (int i = 0; i < warm_up; i++) {
        // Primitive execution: first layer mutmul.
        matmul_fc1.execute(engine_stream, matmul_args);
        relu.execute(engine_stream, {{DNNL_ARG_SRC, hidden_mem}, {DNNL_ARG_DST, hidden_mem}});

        // Primitive execution
        matmul_fc2.execute(engine_stream, matmul_args2);
        relu2.execute(engine_stream, {{DNNL_ARG_SRC, dst_mem}, {DNNL_ARG_DST, dst_mem}});
    }

    auto start_time = high_resolution_clock::now();
    for (int i = 0; i < iter_num; i++) {
        // Primitive execution: first layer mutmul.
        matmul_fc1.execute(engine_stream, matmul_args);
        relu.execute(engine_stream, {{DNNL_ARG_SRC, hidden_mem}, {DNNL_ARG_DST, hidden_mem}});

        // Primitive execution
        matmul_fc2.execute(engine_stream, matmul_args2);
        relu2.execute(engine_stream, {{DNNL_ARG_SRC, dst_mem}, {DNNL_ARG_DST, dst_mem}});
    }
    auto end_time = high_resolution_clock::now();

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    // read_from_dnnl_memory(hidden_data.data(), hidden_mem);
    read_from_dnnl_memory(dst_data.data(), dst_mem);

    auto elapsed_time = duration_cast<milliseconds>(end_time - start_time).count();

    // Print the elapsed time.
    std::cout << "Elapsed time on double mlp without post-ops: " << elapsed_time << " ms" << std::endl;
    // std::cout << "Output: ";
    // for (int i = 0; i < MB * M * HIDDEN_SIZE; i++) {
    //     std::cout << hidden_data[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Output: ";
    // for (int i = 0; i < MB * M * OUTPUT_SIZE; i++) {
    //     std::cout << dst_data[i] << " ";
    // }
    // std::cout << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(mlp_example, parse_engine_kind(argc, argv));
}
