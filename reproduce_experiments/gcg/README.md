To reproduce the nanoGCG baseline, first make sure you have installed the optional dependencies `poetry install --with nanogcg`. To run the attacks, you can either use a config file: 

```shell
python reproduce_experiments/gcg/generate.py --config reproduce_experiments/gcg/configs/llama3.2_1b.toml 
```

Or use command line arguments: 

```shell
python reproduce_experiments/gcg/generate.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --num_steps 32 
```

The only mandatory argument is the `model_name`. Go to the original repository: <https://github.com/GraySwanAI/nanoGCG> for more information. 