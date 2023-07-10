# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np
from eager_op_test import OpTest


class TestDnnlFusedMatMulOp(OpTest):
    def generate_data(self):
        self.x = np.random.random((25, 2, 2)).astype("float32")
        self.beta = 1.0
        self.out = np.log(1 + np.exp(self.x))

    def set_attributes(self):
        self.beta = self.beta if hasattr(self, 'beta') else 1.0
        self.attrs = {'beta': self.beta}

    def setUp(self):
        # Set max isa, otherwise fails on SKX and earlier
        # os.environ["DNNL_MAX_CPU_ISA"] = "AVX"
        self.op_type = "fused_softplus"
        self._cpu_only = True
        self.use_mkldnn = True
        self.generate_data()
        self.set_attributes()
        self.attrs['use_mkldnn'] = True

        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_dygraph=False)



if __name__ == "__main__":
    from paddle import enable_static

    enable_static()
    unittest.main()