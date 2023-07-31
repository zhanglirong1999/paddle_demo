import paddle
import numpy as np
from paddle import fluid

paddle.enable_static()
fluid.set_flags({'FLAGS_use_mkldnn': True})

data = paddle.static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')

conv2d = paddle.static.nn.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
print(conv2d.shape) # [-1, 2, 30, 30]

place = paddle.CPUPlace()  
exe = paddle.static.Executor(place) 

# 开启compile program，ir pass优化：
startup_program = paddle.static.default_startup_program()
main_program = paddle.static.default_main_program()

build_strategy = paddle.static.BuildStrategy()
# build_strategy.enable_addto = True
build_strategy.fuse_elewise_add_act_ops = True

main_program = paddle.static.CompiledProgram(
        main_program, build_strategy=build_strategy
    )
exe.run(startup_program) 
data_np = np.random.rand(1, 3, 32, 32).astype('float32')
output = exe.run(main_program, feed={'data': data_np}, fetch_list=[conv2d])
print(main_program)


# 不开启compile program，没有ir pass优化：
# startup_program = paddle.static.default_startup_program()
# main_program = paddle.static.default_main_program()
# exe.run(startup_program)
# data_np = np.random.rand(1, 3, 32, 32).astype('float32')
# output = exe.run(main_program, feed={'data': data_np}, fetch_list=[conv2d])

