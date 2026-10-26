seed=122

#MODEL_PATH=('/home/share/FM_Code/Stage1/PanDerm/Model_Weights/panderm_bb_data6_checkpoint-499.pth')
MODEL_PATH=('/home/share/FM_Code/Stage1/PanDerm/Model_Weights/panderm_ll_data6_checkpoint-499.pth')
## finetune and eval on PAD-UFES dataset

CUDA_VISIBLE_DEVICES=2 python3 run_class_finetuning.py \
    --model PanDerm_Large_FT \
    --pretrained_checkpoint $MODEL_PATH \
    --nb_classes 6 \
    --batch_size 128 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 10 \
    --epochs 50 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
    --weights \
    --monitor acc \
    --sin_pos_emb \
    --no_auto_resume \
    --exp_name "pad finetune and eval" \
    --imagenet_default_mean_and_std \
    --wandb_name "Reproduce_PAD_FT_${seed}" \
    --output_dir "/home/share/FM_Code/PanDerm/PAD_Res_base/" \
    --csv_path /home/share/Uni_Eval/pad-ufes/2000.csv \
    --root_path /home/share/Uni_Eval/pad-ufes/images/ \
    --seed ${seed} \
    --TTA

#Finetune and eval on HAM10000 dataset
#CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
#    --model PanDerm_Large_FT \
#    --pretrained_checkpoint $MODEL_PATH \
#    --nb_classes 7 \
#    --batch_size 128 \
#    --lr 5e-4 \
#    --update_freq 1 \
#    --warmup_epochs 10 \
#    --epochs 50 --layer_decay 0.65 --drop_path 0.2 \
#    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
#    --weights \
#    --monitor acc \
#    --sin_pos_emb \
#    --no_auto_resume \
#    --exp_name "ham finetune and eval_acc" \
#    --imagenet_default_mean_and_std \
#    --wandb_name "Reproduce_HAM_FT_${seed}" \
#    --output_dir "/home/share/FM_Code/PanDerm/HAM_Res_monitor_acc_TTA/" \
#    --csv_path /home/syyan/XJ/PanDerm-open_source/data/linear_probing/HAM-official-7-lp.csv \
#    --root_path /home/share/Uni_Eval/ISIC2018_reader/images/ \
#    --seed ${seed} \
#    --TTA