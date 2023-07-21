import paddle
import numpy as np
from paddle import fluid

data = paddle.to_tensor(np.random.rand(1, 3, 32, 32).astype('float32'))

fluid.set_flags({'FLAGS_use_mkldnn': True})

conv2d = paddle.nn.Conv2D(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
conv2d = paddle.jit.to_static(conv2d)

output = conv2d(data)
print(output)
print(output.shape) 
