# 

1. Create an environment with python=3.10.

```sh
mamba create -n hipporag python=3.10
mamba activate hipporag
pip install -r requirements.txt
```

2. Run vllm on one GPU.

```sh
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 HF_HOME= vllm serve google/medgemma-4b-it --tensor-parallel-size 1 --max_model_len 4096 --gpu-memory-utilization 0.95s
```

3. When the vllm server is up, run the `demo_local_pmc` script with another gpu
```sh
CUDA_VISIBLE_DEVICES=1 HF_HOME= python demo_local_pmc.py
```
