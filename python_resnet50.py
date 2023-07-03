import argparse
import numpy as np
import time

# 引用 paddle inference 推理库
import paddle.inference as paddle_infer

def main():
    args = parse_args()

    config = paddle_infer.Config(args.model_file, args.params_file)

    # 启用 oneDNN
    config.enable_mkldnn()

    print(config.mkldnn_enabled())

    # create predictor
    predictor = paddle_infer.create_predictor(config)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
    input_handle.reshape([args.batch_size, 3, 318, 318])
    input_handle.copy_from_cpu(fake_input)

    start_time = time.time()
    # 运行predictor
    predictor.run()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('one epoch cast time: {:.2f}'.format(elapsed_time))

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
    print("Output data size is {}".format(output_data.size))
    print("Output data shape is {}".format(output_data.shape))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

if __name__ == "__main__":
    main()

