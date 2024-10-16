base_model=YOUR_MODEL_PATH
path=YOUR_EVAL_PATH
result_path=YOUR_RESULT_PATH
embed_path=YOUR_EMBEDDING_PATH
cp YOUR_TOKENIZER_PATH/*token*.json ${base_model}/
CUDA_VISIBLE_DEVICES=6 python fft_infer.py \
    --base_model ${base_model} \
    --test_data_path YOUR_TEST_JSON_PATH \
    --result_json_data ${path} \
    --length_penalty 1.0 \

CUDA_VISIBLE_DEVICES=6 python embedding.py \
    --base_model ${base_model} \
    --embed_path ${embed_path} \

CUDA_VISIBLE_DEVICES=6 python eval_acc_2.py \
    --path ${path} \
    --result_path ${result_path} \
    --base_model ${base_model} \
    --embed_path ${embed_path} \ 