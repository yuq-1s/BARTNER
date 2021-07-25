for lb in latest best; do
    for dm in train eval; do
        CUDA_VISIBLE_DEVICES=5 python test_t5.py --bart_name=t5-large --use_latest_or_best=$lb --dataset_mode=$dm --batch_size=128
    done
done