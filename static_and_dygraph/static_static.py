import paddle
import numpy as np
from paddle import fluid

paddle.enable_static()
fluid.set_flags({'FLAGS_use_mkldnn': True})

data = paddle.static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
conv2d = paddle.static.nn.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
print(conv2d.shape) # [-1, 2, 30, 30]

place = paddle.CPUPlace()  
exe = paddle.static.Executor(place)  # 用Executor对象

startup_program = paddle.static.default_startup_program()
main_program = paddle.static.default_main_program()
exe.run(startup_program) 
data_np = np.random.rand(1, 3, 32, 32).astype('float32')
output = exe.run(main_program, feed={'data': data_np}, fetch_list=[conv2d])
print(output)

