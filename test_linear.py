import paddle
from paddle.nn import Linear
from paddle import fluid
from paddle.fluid.framework import _global_flags
import os
import unittest

SEED = 2023

class TestLiner(unittest.TestCase):
    def test_linear(self):
        print('--------no dygraph ------------')
        fluid.set_flags({'FLAGS_use_mkldnn': True})
        input_data = paddle.randn([3, 2], 'float32')  

        linear = paddle.nn.Linear(in_features=2, out_features=4)  
        print(
                        "check: _global_flags()['FLAGS_use_mkldnn']=",
                        _global_flags()["FLAGS_use_mkldnn"],
                    )
        print("check: DNNL_VERBOSE=", os.environ['DNNL_VERBOSE'])
        output = linear(input_data)
        print(output.shape)  


    def test_dygraph_linear(self):
            print('---------dygraph-----------')
            fluid.set_flags({'FLAGS_use_mkldnn': True})
            place = fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                    fluid.default_main_program().random_seed = SEED
                    fluid.default_startup_program().random_seed = SEED
                    weight_attr = paddle.ParamAttr(
                        name="weight",
                        initializer=paddle.nn.initializer.Constant(value=0.5))
                    bias_attr = paddle.ParamAttr(
                        name="bias",
                        initializer=paddle.nn.initializer.Constant(value=1.0))
                    linear = paddle.fluid.dygraph.nn.Linear(2, 4, param_attr=weight_attr, bias_attr =bias_attr)

                    x = paddle.randn((3, 2), dtype="float32")
                    output = linear(x)
                    print(output.shape)  

if __name__ == "__main__":
    unittest.main()