import os
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec
from paddle.vision.models import resnet50
import time
from paddle import fluid
from paddle.fluid.framework import _global_flags

# use_dnnl = True
# # # 设置PaddlePaddle环境变量，启用OneDNN加速
# if use_dnnl:
#     os.environ['FLAGS_use_mkldnn'] = '1'

# paddle.fluid.set_flags({'FLAGS_use_mkldnn': False})
# paddle.fluid.set_flags({'FLAGS_use_mkldnn': True})
SEED = 2023

input_spec = InputSpec([None, 3, 224, 224], 'float32', 'image')
label_spec = InputSpec([None, 1], 'int64', 'label')

class CustomDataset(paddle.io.Dataset):
    def __init__(self):
        self.data = np.random.rand(10, 3, 224, 224).astype('float32')
        self.label = np.random.randint(0, 2, size=(10, 1)).astype('int64')

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

train_dataset = CustomDataset()
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
def train(to_static=False):
    print(
            "check: _global_flags()['FLAGS_use_mkldnn']=",
            _global_flags()["FLAGS_use_mkldnn"],
        )
    print("check: DNNL_VERBOSE=", os.environ['DNNL_VERBOSE'])

    loss_data = []
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED
        model = resnet50(num_classes=10)
        loss_fn = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

        if to_static:
            model = paddle.jit.to_static(model)  

        start_time = time.time()
        for epoch in range(1):

            for batch_id, data in enumerate(train_loader()):
                x_data = data[0]
                y_data = data[1]
                logits = model(x_data)
                loss = loss_fn(logits, y_data)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                loss_data.append(float(loss))
                print('Epoch {}, batch {}, loss {}'.format(epoch, batch_id, loss.numpy()))
            
        end_time = time.time()
        elapsed_time = end_time - start_time

        print('cost time: {:.2f}'.format(elapsed_time))
    return loss_data

print('-----------now dygraph ----------------')

dygraph_loss_cpu = train()
fluid.set_flags({'FLAGS_use_mkldnn': True})
try:
    dygraph_loss_mkldnn = train()
finally:
    fluid.set_flags({'FLAGS_use_mkldnn': False})

np.testing.assert_allclose(
    dygraph_loss_cpu,
    dygraph_loss_mkldnn,
    rtol=1e-05,
    err_msg='cpu dygraph is {}\n mkldnn dygraph is \n{}'.format(
        dygraph_loss_cpu, dygraph_loss_mkldnn
            ),
)

print('-----------now static ----------------')

static_loss_cpu = train(True)
fluid.set_flags({'FLAGS_use_mkldnn': True})
try:
    static_loss_mkldnn = train(True)
finally:
    fluid.set_flags({'FLAGS_use_mkldnn': False})

np.testing.assert_allclose(
    static_loss_cpu,
    static_loss_mkldnn,
    rtol=1e-05,
    err_msg='cpu static is {}\n mkldnn static is \n{}'.format(
        static_loss_cpu, static_loss_mkldnn
            ),
)
