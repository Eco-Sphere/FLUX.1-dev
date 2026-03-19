export ASCEND_RT_VISIBLE_DEVICES=2

export model_path=/home/data1/FLUX.1-dev

# Warning
echo "===================================================="
echo "If an error occurs, please modify the weight config"
echo "file as described in the README.md."
echo "===================================================="

# 在环境中导入以下环境变量提高推理性能
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2

# 等价优化
export RMSNORM_FUSE=1   
export ROPE_FUSE=1      
export POSEMB_CACHE=1   
export ENABLE_LA=0      
export ADALN_FUSE=1     
export FAST_GELU=0
export USE_NZ=0
export CV_PARALLEL_LEVEL=2

run_case() {
    local width=$1
    local height=$2
    local batch_size=$3

    python inference_flux.py \
       --path ${model_path} \
       --save_path "./res" \
       --prompt "A cat holding a sign that says hello world" \
       --device_id 0 \
       --device "npu" \
       --width $width \
       --height $height \
       --infer_steps 20 \
       --seed 42 \
       --batch_size $batch_size
}

for ((i=1; i<=2; i++)); do
    run_case 512 512 $i
    sleep 10
done

for ((i=1; i<=2; i++)); do
    run_case 1024 1024 $i
    sleep 10
done
