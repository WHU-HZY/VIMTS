### VisionTS ###
patience=15

# 修改数组定义方式
gpus=(1 2 3 4 5)
seeds=(1 2 3 4 5)
weight=logs/ushcn/pretrain/pretrain_weights/xxxx.pth

for i in {0..4}
do
    gpu=${gpus[$i]}
    seed=${seeds[$i]}
    echo "gpu: $gpu || seed: $seed || weight: $weight"

    python run_models.py --model VisionTS-B \
    --dataset ushcn --state 'def' --history 24 \
    --patience $patience --batch_size 64 --lr 1e-4 \
    --patch_size 1 --stride 1 --nhead 1 --tf_layer 1 --nlayer 3 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --train_mode fine-tune --finetune_type norm --mask_ratio 0.4 \
    --pretrain_weights_dir $weight --few_shot_ratio None \
    --seed $seed --gpu $gpu --log_dir finetune &

done

wait


