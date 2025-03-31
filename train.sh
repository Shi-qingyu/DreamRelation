export PYTHONPATH=.
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./data/hug"
export OUTPUT_DIR="./lora-weights/lora-trained-xl-hugging-0.1kp-0.001cal-0.6"
export HF_ENDPOINT="https://hf-mirror.com"

accelerate launch scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_epochs=1000 \
  --seed="0"