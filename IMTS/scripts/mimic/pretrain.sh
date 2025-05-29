### VisionTS ###
patience=15

# 修改数组定义方式
gpus=(1)

for i in {0..0}
do
    gpu=${gpus[$i]}
    echo "gpu: $gpu"

    python run_models.py --model VisionTS-B \
    --dataset mimic --state 'def' --history 24 \
    --patience $patience --batch_size 16 --lr 1e-4 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 3 \
    --te_dim 40 --node_dim 40 --hid_dim 40 --mask_ratio 0.4 \
    --train_mode pre-train --few_shot_ratio None \
    --seed 1 --gpu $gpu --log_dir pretrain &
done

wait