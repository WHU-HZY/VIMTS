### VisionTS ###
patience=15

# 修改数组定义方式
gpus=(1)

for i in {0..0}
do
    gpu=${gpus[$i]}
    echo "gpu: $gpu"

    python run_models.py --model VisionTS-B \
    --dataset ushcn --state 'def' --history 24 \
    --patience $patience --batch_size 64 --lr 1e-4 \
    --patch_size 1 --stride 1 --nhead 1 --tf_layer 1 --nlayer 3 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --train_mode pre-train --mask_ratio 0.4 --few_shot_ratio None \
    --seed 1 --gpu $gpu --log_dir pretrain &
done

wait