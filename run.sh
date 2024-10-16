# accelerate config
for seed in 0
do
    for lr in 1e-4
    do
        for dropout in 0.05    
        do
            for sample in -1
            do
                echo "lr: $lr, dropout: $dropout , seed: $seed,"
                CUDA_VISIBLE_DEVICES=7 accelerate launch train.py \
                    --base_model "YOUR_MODEL_NAME" \
                    --train_data_path "[\"YOUR_TRAIN_DATA_PATH\"]"  \
                    --val_data_path "[\"YOUR_VAL_DATA_PATH\"]"  \
                    --output_dir YOUR_SAVE_PATH\
                    --batch_size 64 \
                    --micro_batch_size 16\
                    --num_epochs 3 \
                    --learning_rate $lr \
                    --cutoff_len 1024\
                    --train_on_inputs False\
                    --group_by_length False\
                    --resume_from_checkpoint 'XXX' \
                    --seed $seed \
                    --sample $sample 
            done    
        done
    done
done
