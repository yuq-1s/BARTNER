for lb in latest best; do
    for dm in train eval; do
        CUDA_VISIBLE_DEVICES=6 python test_t5.py --bart_name=t5-base --use_latest_or_best=$lb --dataset_mode=$dm --batch_size=256
    done
done