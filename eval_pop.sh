path=Your_Result_Path
result_path=/NAS/yjt/CFT/model/books/Song1_0_-1_qwen_64_16_lr_1e-4_0.01_0.2_weighting=True__add_loss_diff=True/pop_count_without_his.json
base_model=/NAS/yjt/CFT/model/books/Song1_0_-1_qwen_64_16_lr_1e-4_0.01_0.2_weighting=True__add_loss_diff=True/checkpoint-10672
embed_path=/NAS/yjt/CFT/model/books/Song1_0_-1_qwen_64_16_lr_1e-4_0.01_0.2_weighting=True__add_loss_diff=True/embedding4.pt
info_path=/NAS/yjt/CFT/data/books/info.txt
test_csv_path=/NAS/yjt/CFT/data/books/test.csv
pop_count_path=/NAS/yjt/CFT/data/books/books_pop_count_5.csv

CUDA_VISIBLE_DEVICES=0 python evaluate_all_pop.py \
    --path ${path} \
    --result_path ${result_path} \
    --base_model ${base_model}\
    --embed_path ${embed_path}\
    --info_path ${info_path}\
    --test_csv_path ${test_csv_path}\
    --pop_count_path ${pop_count_path}\
