VLLM_WORKER_MULTIPROC_METHOD= CUDA_VISIBLE_DEVICES= HF_HOME= vllm serve google/medgemma-4b-it --tensor-parallel-size 1 --max_model_len 4096 --gpu-memory-utilization 0.95s

CUDA_VISIBLE_DEVICES= HF_HOME= python demo_local_pmc.py

