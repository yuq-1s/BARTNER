for lb in latest best; do
    for dm in train eval; do
        CUDA_VISIBLE_DEVICES=4 python test_t5.py --bart_name=t5-3b --use_latest_or_best=$lb --dataset_mode=$dm --batch_size=32
    done
done