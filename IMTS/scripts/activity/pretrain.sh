### VisionTS ###
patience=15

# 修改数组定义方式
gpus=(1)

for i in {0..0}
do
    gpu=${gpus[$i]}
    echo "gpu: $gpu"

    python run_models.py --model VisionTS-B \
    --dataset activity --state 'def' --history 3000 \
    --patience $patience --batch_size 32 --lr 1e-4 \
    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 2 \
    --te_dim 5 --node_dim 5 --hid_dim 64 \
    --train_mode pre-train --mask_ratio 0.7 --few_shot_ratio None \
    --seed 1 --gpu $gpu --log_dir pretrain &
done

wait
