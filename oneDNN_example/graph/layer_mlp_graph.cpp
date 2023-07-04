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

/// @example cpu_simple_pattern_f32.cpp
/// @copybrief cpu_simple_pattern_f32_cpp
/// Annotated version: @ref cpu_simple_pattern_f32_cpp

/// @page cpu_simple_pattern_f32_cpp CPU example for a simple f32 pattern
///
/// > Example code: @ref cpu_simple_pattern_f32.cpp

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"
// #include "oneapi/dnnl/dnnl_graph_sycl.hpp"

#include "common/example_utils.hpp"
#include "common/helpers_any_layout.hpp"
#include <chrono>
using namespace std::chrono;

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
void mlp_graph(dnnl::engine::kind ekind){
    /// create a graph
    static int warm_up = 10;
    static int iter_num = 10;

    graph g(ekind);

    std::vector<int64_t> input_dims {1, 1, 128};
    std::vector<int64_t> weights1_dims {1, 128, 64};
    std::vector<int64_t> bias1_dims {1, 1, 64};

    std::vector<int64_t> weights2_dims {1, 64, 2};
    std::vector<int64_t> bias2_dims {1, 1, 2};

    std::vector<int64_t> hidden_dims {1, 1, 64};
    std::vector<int64_t> output_dims {1, 1, 2};

    std::vector<int64_t> hidden_tmp_dims {1, 1, 64};
    std::vector<int64_t> output_tmp_dims {1, 1, 2};

    // Create logical tensors for the input, hidden, and output layers
    logical_tensor input_desc {0, data_type::f32, input_dims, layout_type::undef};
    logical_tensor hidden_desc {1, data_type::f32, hidden_dims, layout_type::undef};
    /// ndims = 4, let the library to calculate the output shape.
    logical_tensor output_desc {2, data_type::f32, output_dims, layout_type::undef};
    // logical_tensor output_desc {2, data_type::f32, 3, layout_type::undef};


    logical_tensor weights1_desc {
        3, data_type::f32, weights1_dims, layout_type::undef};

    logical_tensor bias1_desc {
        4, data_type::f32, bias1_dims, layout_type::undef};

    logical_tensor weights2_desc {
        5, data_type::f32, weights2_dims, layout_type::undef};

    logical_tensor bias2_desc {
        6, data_type::f32, bias2_dims, layout_type::undef};

    logical_tensor hidden_tmp_desc {7, data_type::f32, hidden_tmp_dims, layout_type::undef};
    /// ndims = 4, let the library to calculate the output shape.
    logical_tensor output_tmp_desc {8, data_type::f32, output_tmp_dims, layout_type::undef};


    // Create the operations for the MLP
    op hidden_matmul {0, op::kind::MatMul, {input_desc, weights1_desc, bias1_desc},
        {hidden_tmp_desc}, "hidden"};
    op relu {1, op::kind::ReLU, {hidden_tmp_desc}, {hidden_desc}, "relu"};

    op output_matmul {2, op::kind::MatMul, {hidden_desc, weights2_desc, bias2_desc},
        {output_tmp_desc}, "matmul"};

    op relu_2 {3, op::kind::ReLU, {output_tmp_desc}, {output_desc}, "relu"};

    // Set the attributes for the operations
    // hidden_matmul.set_attr<std::string>(op::attr::data_format, "NXC");
    // hidden_matmul.set_attr<std::string>(op::attr::weights_format, "OIX");

    g.add_op(hidden_matmul);
    g.add_op(relu);
    g.add_op(output_matmul);
    g.add_op(relu_2);

    // Finalize the graph
    g.finalize();
    auto partitions = g.get_partitions();

    std::cout << "size of partition: " <<partitions.size()  << std::endl;

    std::unordered_set<size_t> ids_with_any_layout;
    // / This is a helper function which helps decide which logical tensor is
    // / needed to be set with `dnnl::graph::logical_tensor::layout_type::any`
    // / layout. Typically, users need implement the similar logic in their code
    // / for best performance.
    set_any_layout(partitions, ids_with_any_layout);

    // Set the layout types for the input and output tensors
    // input_desc.set_layout_type(layout_type::undef);
    // output_desc.set_layout_type(layout_type::undef);
    allocator alloc {};
    dnnl::engine eng =make_engine_with_allocator(ekind, 0, alloc);
    dnnl::stream strm {eng};

    
    // mapping from logical tensor id to output tensors
    // used to the connection relationship between partitions (e.g partition 0's
    // output tensor is fed into partition 1)
    std::unordered_map<size_t, tensor> global_outputs_ts_map;
    // manage the lifetime of memory buffers binded to those input/output tensors
    std::vector<std::shared_ptr<void>> data_buffers;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;
    

    for (const auto &partition : partitions) {
        if (partition.is_supported()) {
            // The required inputs and outputs of a partition are also called ports
            std::vector<logical_tensor> inputs = partition.get_input_ports();
            std::vector<logical_tensor> outputs = partition.get_output_ports();

            //对于每个逻辑张量，如果其是另一个分区的输出，则将其更新为已查询的逻辑张量。
            //否则，将其布局转换为实际的内存布局
            // update input logical tensors with concrete layout
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                size_t id = inputs[idx].get_id();
                // the tensor is an output of another partition
                if (id_to_queried_logical_tensors.find(id)
                        != id_to_queried_logical_tensors.end())
                    inputs[idx] = id_to_queried_logical_tensors[id];
                else {
                    auto ori_lt = inputs[idx];
                    // create logical tensor with strided layout
                    inputs[idx] = logical_tensor {ori_lt.get_id(),
                            ori_lt.get_data_type(), ori_lt.get_dims(),
                            layout_type::strided};
                }
            }

            // update output logical tensors with concrete layout
            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                size_t id = outputs[idx].get_id();
                layout_type ltype = layout_type::strided;
                if (ids_with_any_layout.count(id)) ltype = layout_type::any;
                auto ori_lt = outputs[idx];
                // create logical tensor with strided/any layout
                outputs[idx] = logical_tensor {ori_lt.get_id(),
                        ori_lt.get_data_type(), ori_lt.get_dims(), ltype};
            }

            /// Compile the partition to generate compiled partition with the
            /// input and output logical tensors.
            /// @snippet cpu_get_started.cpp Compile partition
            //[Compile partition]
            // 编译分区
            compiled_partition cp = partition.compile(inputs, outputs, eng);
            //[Compile partition]

            // update output logical tensors with queried one
            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                size_t id = outputs[idx].get_id();
                outputs[idx] = cp.query_logical_tensor(id);
                id_to_queried_logical_tensors[id] = outputs[idx];
            }

            // Binding data buffers with input and output logical tensors
            std::vector<tensor> inputs_ts, outputs_ts;
            inputs_ts.reserve(inputs.size());
            outputs_ts.reserve(outputs.size());
            for (const auto &in : inputs) {
                size_t id = in.get_id();
                size_t mem_size = in.get_mem_size();
                // check if the input is an output of another partition
                auto pos = global_outputs_ts_map.find(id);
                if (pos != global_outputs_ts_map.end()) {
                    inputs_ts.push_back(pos->second);
                    continue;
                }
                // memory allocation
                data_buffers.push_back({});
                data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                inputs_ts.push_back(
                        tensor {in, eng, data_buffers.back().get()});
            }

            for (const auto &out : outputs) {
                size_t mem_size = out.get_mem_size();
                // memory allocation
                data_buffers.push_back({});
                data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                outputs_ts.push_back(
                        tensor {out, eng, data_buffers.back().get()});
                std::cout << mem_size  << std::endl;
                global_outputs_ts_map[out.get_id()] = outputs_ts.back();
            }

            /// Execute the compiled partition 1 on the specified stream.
            /// @snippet cpu_get_started.cpp Execute compiled partition 1
            //[Execute compiled partition]
            // warm up
            for (int i = 0; i < warm_up; i++) {
                cp.execute(strm, inputs_ts, outputs_ts);
            }

            auto start_time = high_resolution_clock::now();
            for (int i = 0; i < iter_num; i++) {
                cp.execute(strm, inputs_ts, outputs_ts);
            }
            auto end_time = high_resolution_clock::now();
            auto elapsed_time = duration_cast<milliseconds>(end_time - start_time).count();
            //[Execute compiled partition]
            std::cout << "Elapsed time on mlp in Graph API: " << elapsed_time << " ms" << std::endl;

        } else {
            std::cout << "program: got unsupported partition, users need "
                "handle the operators by themselves." << std::endl;
        }
    }


    // auto end_time = high_resolution_clock::now();
    // auto elapsed_time = duration_cast<milliseconds>(end_time - start_time).count();

    // Print the elapsed time.
    std::cout << "End of graph computation."  << std::endl;

}
int main(int argc, char **argv) {
    // clang-format off
    std::cout << "This is an example."  << std::endl;
    const dnnl::engine::kind ekind = dnnl::engine::kind::cpu;
    mlp_graph(ekind);


}