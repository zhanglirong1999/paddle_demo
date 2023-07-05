import paddle.fluid as fluid
import numpy as np
from paddle import fluid

class ResNet50(fluid.dygraph.Layer):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        self.conv = fluid.dygraph.Conv2D(num_channels=3, num_filters=64, filter_size=7, stride=2, padding=3, bias_attr=False)
        self.bn = fluid.dygraph.BatchNorm(num_channels=64, act='relu')
        self.pool = fluid.dygraph.Pool2D(pool_size=3, pool_stride=2, pool_type='max', ceil_mode=True)
        self.fc = fluid.dygraph.Linear(input_dim=2048, output_dim=num_classes)

    def forward(self, x, label=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = fluid.layers.reshape(x, [-1, 2048])
        x = self.fc(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            loss = fluid.layers.cross_entropy(input=x, label=label)
            avg_loss = fluid.layers.mean(loss)
            return x, acc, avg_loss
        else:
            return x


fluid.set_flags({'FLAGS_use_mkldnn': True})
model = ResNet50(num_classes=10)

input_data = np.random.randn(1, 3, 224, 224).astype('float32')

with fluid.dygraph.guard():
    input_var = fluid.dygraph.to_variable(input_data)
    output_var = model(input_var)
    print(output_var.shape)