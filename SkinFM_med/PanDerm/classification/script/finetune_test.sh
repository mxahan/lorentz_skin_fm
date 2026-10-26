seed=0



#MODEL_PATH=('/home/share/FM_Code/Stage1/PanDerm/Model_Weights/panderm_bb_data6_checkpoint-499.pth')
MODEL_PATH=('/home/share/FM_Code/Stage1/PanDerm/Model_Weights/panderm_ll_data6_checkpoint-499.pth')

RESUME_PATH=/home/share/FM_Code/PanDerm/PAD_Res/checkpoint-best.pth
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
    --model PanDerm_Large_FT \
    --pretrained_checkpoint $MODEL_PATH \
    --nb_classes 6 \
    --batch_size 64 \
    --lr 5e-4 --update_freq 1 \
    --warmup_epochs 10 \
    --epochs 50 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 \
    --mixup 0.8 --cutmix 1.0 \
    --sin_pos_emb \
    --no_auto_resume \
    --exp_name 'pad test' \
    --imagenet_default_mean_and_std \
    --wandb_name "eval${seed}" \
    --output_dir /home/syyan/XJ/PanDerm-open_source/finetune/work_dir/PAD_Res/ \
    --csv_path /home/share/Uni_Eval/pad-ufes/2000.csv \
    --root_path /home/share/Uni_Eval/pad-ufes/images/ \
    --seed ${seed} \
    --resume ${RESUME_PATH} \
    --eval \
    --TTA
#RESUME_PATH=/home/share/FM_Code/PanDerm/HAM_Res/checkpoint-best.pth
#CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
#    --model PanDerm_Large_FT \
#    --pretrained_checkpoint $MODEL_PATH \
#    --nb_classes 7 \
#    --batch_size 256 \
#    --lr 5e-4 \
#    --update_freq 1 \
#    --warmup_epochs 1 \
#    --epochs 1 --layer_decay 0.65 --drop_path 0.2 \
#    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
#    --weights \
#    --monitor recall \
#    --sin_pos_emb \
#    --no_auto_resume \
#    --exp_name "ham finetune and eval" \
#    --imagenet_default_mean_and_std \
#    --wandb_name "Reproduce_HAM_FT${BATCH_SIZE}_${LR}_${seed}" \
#    --output_dir "/home/share/FM_Code/PanDerm/HAM_Res/" \
#    --csv_path /home/syyan/XJ/PanDerm-open_source/data/linear_probing/HAM-official-7-lp.csv \
#    --root_path /home/share/Uni_Eval/ISIC2018_reader/images/ \
#    --seed ${seed} \
#    --resume ${RESUME_PATH} \
#    --eval \
#    --TTA