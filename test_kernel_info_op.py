import os
import unittest

import numpy as np
from eager_op_test import OpTest
from paddle import fluid
import collections
import difflib
import re
import sys

SEED = 2023
class TestDnnlFusedMatMulOp(OpTest):
    def setUp(self):
        self.op_type = "fusion_gru"
        self.__class__.op_type = "fusion_gru"
        self.lod = [[2, 4, 3]]
        self.M = 3
        self.D = 5
        self.is_reverse = False
        self.with_h0 = False
        self.use_mkldnn = True
        self._cpu_only = True
        self.with_bias = True
        self.act_state = 'tanh'
        self.act_gate = 'sigmoid'
        self.origin_mode = False
        self.use_mkldnn = True
        self.mkldnn_data_type = "bfloat16"
        self.force_fp32_output = False
        self.weights_dtype = 'fp32'
        
    def test_registered_op(self):

        all_kernels_info = fluid.core._get_all_register_op_kernels("phi")
        # all_kernels_info = fluid.core._get_all_register_op_kernels("fluid")
        
        # [u'data_type[double]:data_layout[ANY_LAYOUT]:place[CPUPlace]:library_type[PLAIN]'
        op_kernel_types = collections.defaultdict(list)
        if "fused_matmul" in all_kernels_info:
            print('fused_matmul in key')
            print(all_kernels_info['fused_matmul'])
            print(all_kernels_info['fusion_gru'])
        else:
            print('fused_matmul not in key')
        for op_type, op_infos in all_kernels_info.items():

            is_grad_op = op_type.endswith("_grad")
            if is_grad_op:
                continue

            for op_info in op_infos:
                op_kernel_types[op_type].append(op_info)

        for op_type, op_kernels in sorted(
            op_kernel_types.items(), key=lambda x: x[0]
        ):
            print(op_type)
    
    def test_registered_phi_kernels(self):
        print('---------now nernel name ----------')
        phi_function_kernel_infos = fluid.core._get_registered_phi_kernels("function")
        registered_kernel_list = list(phi_function_kernel_infos.keys())
        # for key, value in kernel_infos.items():
        #     print(key, " ".join(value))
        for kernel_name in registered_kernel_list:
            print(kernel_name)
            
    # def test_all_kernel_api(self):
    #     kernel_res = fluid.core._get_kernels_test()
    #     print('pass')
        

if __name__ == "__main__":
    unittest.main()