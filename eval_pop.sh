path=YOUR_EVAL_PATH
result_path=YOUR_RESULT_PATH
base_model=YOUR_MODEL_PATH
embed_path=YOUR_EMBED_PATH
info_path=YOUR_INFO_PATH
test_csv_path=TEST_CSV_PATH
pop_count_path=POP_COUNT_PATH

CUDA_VISIBLE_DEVICES=0 python evaluate_all_pop.py \
    --path ${path} \
    --result_path ${result_path} \
    --base_model ${base_model}\
    --embed_path ${embed_path}\
    --info_path ${info_path}\
    --test_csv_path ${test_csv_path}\
    --pop_count_path ${pop_count_path}\
