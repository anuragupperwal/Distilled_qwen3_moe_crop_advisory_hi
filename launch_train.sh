# Project Root and Python Path
# ensures that modified litgpt/model.py is used instead of any installed version
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# H100 Specific Optimizations
# Force use of TensorFloat-32 for faster matmuls on H100
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# 3. Model and Data Paths
STUDENT_PATH="checkpoints/Qwen/Qwen3-0.6B-moe-initial/lit_model.pth"
TEACHER_PATH="checkpoints/Qwen/Qwen3-14B/lit_model.pth"
DATA_PATH="data/agri_hi_train.parquet"

# 4. Launch Training
# Using 'torchrun' is best practice for H100 to handle process monitoring
echo "Starting CKA-Guided MoE Distillation on H100..."

python train_distill.py \
    --student_path $STUDENT_PATH \
    --teacher_path $TEACHER_PATH \
    --data_path $DATA_PATH \
    --batch_size 2 \
    --max_seq_length 2048 \
    --lr 1e-4

# 5. Instructions
# To run this script:
# chmod +x launch_train.sh
# ./launch_train.sh