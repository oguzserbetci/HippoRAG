# 

1. Create an environment with python=3.10.

```sh
mamba create -n hipporag python=3.10
mamba activate hipporag
pip install -r requirements.txt
```

2. Make sure you're logged into huggingface hub `hf auth login`. And you accepted the terms for the LLM you want to use on huggingface, e.g. [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

3. Run vllm on one GPU.

```sh
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 HF_HOME= vllm serve google/medgemma-4b-it --tensor-parallel-size 1 --max_model_len 4096 --gpu-memory-utilization 0.95
```

4. When the vllm server is up, run the `demo_local_pmc` script with another gpu

```sh
CUDA_VISIBLE_DEVICES=0 HF_HOME= python demo_local_pmc.py
```
