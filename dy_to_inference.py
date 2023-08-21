import paddle
from paddlenlp.transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
from paddle.static import InputSpec
#paddle.enable_static()
#paddle.disable_static()
paddle.set_device("cpu")
config = LlamaConfig.from_pretrained("facebook/llama-7b")
config.use_fused_rms_norm = False
config.use_flash_attention = False
config.rope_fusion_level = "full"
# model = LlamaForCausalLM.from_pretrained(
#     "facebook/llama-7b",
#     dtype="float32",
#     tensor_parallel_degree=1,
#     tensor_parallel_rank=1,
# )

model = LlamaForCausalLM._from_config(config, dtype="float32")
print(config)
# model = paddle.jit.to_static(model, input_spec=[paddle.static.InputSpec(shape=[1, 8, 32, 128], dtype='int64')])

paddle.jit.save(
layer=model,
path='example_model',
input_spec=[paddle.static.InputSpec(shape=[1, 8, 32, 128], dtype='int64')])

#   去除：
#         #with paddle.amp.auto_cast(False):
#        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
 python predictor.py \
    --model_name_or_path "facebook/llama-7b" \
    --batch_size 1 \
    --type static

python predictor.py     --model_name_or_path facebook/llama-7b     --batch_size 1     --dtype "float16"     --type dygraph

