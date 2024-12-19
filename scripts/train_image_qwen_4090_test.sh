
BASE_CHECKPOINT="/home/uagirrl/models/Qwen2-0.5B-Instruct" # replace config.json with https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B_img/config.json
PATH_TO_JSON="/home/uagirrl/datasets/longvu/mini_onevision/CLEVR-Math_onevision.json"
PATH_TO_FOLDER="/home/uagirrl/datasets/longvu/mini_onevision"
VERSION="qwen"

export PYTHONPATH=longvu:$PYTHONPATH

CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=1 --nnodes=1 \
longvu/train.py \
--output_dir "/home/uagirrl/projects/xl_longvu/" \
--input_model_filename $BASE_CHECKPOINT \
--output_model_filename "./checkpoints/cambrian_qwen_4090_test/" \
--data_path $PATH_TO_JSON \
--image_folder $PATH_TO_FOLDER \
--model_max_length 8192 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /home/uagirrl/projects/xl_longvu/checkpoints/cambrian_qwen_4090_test/ \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_steps 500 \
--logging_steps 10 \
--save_strategy "steps" \
--report_to none \
--save_total_limit 1 \
--learning_rate 1e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--tf32 False \
--version $VERSION \
--mm_vision_select_layer "-2" \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--dataloader_num_workers 0 \
--lazy_preprocess True \
--tune_mm_mlp_adapter False \
--freeze_mm_mlp_adapter False \
--freeze_backbone False \
--gradient_checkpointing True \
--mm_projector_type sva \
--image_token_len 576 \
--query_num_list "[576]" \
