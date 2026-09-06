#!/bin/bash
#SBATCH --job-name=kv-layout-ab
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/rhome/deveshisingh/matmul-to-paged-attention
#SBATCH --output=/rhome/deveshisingh/matmul-to-paged-attention/experiments/vllm_layout/results/layout-ab-%j.out
#SBATCH --error=/rhome/deveshisingh/matmul-to-paged-attention/experiments/vllm_layout/results/layout-ab-%j.err

set -uo pipefail

readonly USER_ROOT="/rhome/deveshisingh"
readonly REPO_ROOT="${USER_ROOT}/matmul-to-paged-attention"
readonly ENV_ROOT="${USER_ROOT}/envs/vllm-rocm721"
readonly SCRIPT_ROOT="${REPO_ROOT}/experiments/vllm_layout"
readonly RESULT_ROOT="${SCRIPT_ROOT}/results/job-${SLURM_JOB_ID}"

readonly MODEL="${MODEL:-facebook/opt-125m}"
readonly INPUT_LEN="${INPUT_LEN:-1024}"
readonly OUTPUT_LEN="${OUTPUT_LEN:-32}"
readonly BATCH_SIZE="${BATCH_SIZE:-8}"
readonly WARMUPS="${WARMUPS:-3}"
readonly ITERATIONS="${ITERATIONS:-10}"
readonly LAYOUT_ORDER="${LAYOUT_ORDER:-NHD,HND}"

mkdir -p "${RESULT_ROOT}"
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:${ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TRITON_CACHE_DIR="${USER_ROOT}/.cache/triton"
export TOKENIZERS_PARALLELISM=false

echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Model: ${MODEL}"
echo "Layout order: ${LAYOUT_ORDER}"
echo "Results: ${RESULT_ROOT}"
echo "Visible devices: HIP=${HIP_VISIBLE_DEVICES:-unset} CUDA=${CUDA_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"

"${ENV_ROOT}/bin/python" - <<'PY'
import torch
import vllm

print("PyTorch:", torch.__version__)
print("PyTorch HIP:", torch.version.hip)
print("vLLM:", vllm.__version__)
print("Accelerator available:", torch.cuda.is_available())
if torch.version.hip is None:
    raise SystemExit("ERROR: PyTorch is not a ROCm build")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: the allocated AMD GPU is not visible to PyTorch")
if torch.cuda.is_available():
    print("Accelerator:", torch.cuda.get_device_name(0))
PY

overall_status=0
for layout in ${LAYOUT_ORDER//,/ }; do
  echo "Running ${layout}"
  export VLLM_KV_CACHE_LAYOUT="${layout}"

  "${ENV_ROOT}/bin/vllm" bench latency \
    --model "${MODEL}" \
    --load-format dummy \
    --attention-backend TRITON_ATTN \
    --dtype half \
    --kv-cache-dtype auto \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.50 \
    --enforce-eager \
    --seed 0 \
    --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --num-iters-warmup "${WARMUPS}" \
    --num-iters "${ITERATIONS}" \
    --output-json "${RESULT_ROOT}/${layout}.json" \
    >"${RESULT_ROOT}/${layout}.log" 2>&1

  status=$?
  echo "${layout} exit status: ${status}"
  if [[ ${status} -ne 0 ]]; then
    overall_status=${status}
  fi
done

if [[ -f "${RESULT_ROOT}/NHD.json" && -f "${RESULT_ROOT}/HND.json" ]]; then
  "${ENV_ROOT}/bin/python" "${SCRIPT_ROOT}/summarize_latency.py" \
    "${RESULT_ROOT}/NHD.json" \
    "${RESULT_ROOT}/HND.json" \
    | tee "${RESULT_ROOT}/summary.txt"
else
  echo "One or both layouts failed; inspect ${RESULT_ROOT}/NHD.log and HND.log"
  overall_status=1
fi

echo "Results: ${RESULT_ROOT}"
exit "${overall_status}"
