<div align="center">

# Subspace Rerouting
<div><img src="assets/ssr_schema_screenshot.png" width="730" alt="Warp" /></div>

<div><b><a href="https://arxiv.org/abs/2503.06269">Paper</a> - <a href="/paper/out/main.pdf">Paper (Up-to-date version)</a> - <a href="https://sckathach.github.io/mech-interp/subspace-rerouting/">Website</a> - <a href="https://www.lesswrong.com/posts/mG7jioaAsBasnuD4b/subspace-rerouting-using-mechanistic-interpretability-to-2">Less Wrong post</a></b></div>

</div>

This is the repository for *Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models*, it contains:

- The core SSR algorithm 
- The three implementations: Probe SSR, Steering SSR, and Attention SSR 
- Notebooks to reproduce the experiments of the paper 
- Utilities to run and evaluate jailbreaks 
- Part of the generated jailbreaks 

## Change log *(since paper release)*

### Minor fixes (07 Jul 25)
- Fixed shape issues in `attention/attention_ssr.py` and `steering/steering_ssr.py`.
- Fixed cpu/cuda device issues in in `attention/attention_ssr.py` and `steering/steering_ssr.py`.
- Added `run_ssr_attention_dazzle.ipynb`, `run_ssr_attention_full.ipynb`, `run_ssr_steering.ipynb` examples.

### Major SSR v4 update (17 Mar 25)
- ~~Perturbation in suffix~~ &rarr; perturbation anywhere.
- The core algorithm (`ssr/core.py`) does not depend on `Lens` anymore (enhanced reproducibility).
- Probes and refusal directions can be stored and loaded.
- *Fixes and refactoring*.

## Installation
> [!IMPORTANT]
> The project needs **Python 3.12** 

1. **Install the environment**
    <details open>
    <summary>With miniconda (recommended way)</summary>
    Install miniconda: 

    ```shell
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ```

    Source or reload your shell:
    ```shell
    source ~/.bashrc
    ```

    Create the environment for SSR with python3.12:
    ```shell
    conda create -n ssr python=3.12 -y
    conda activate ssr
    ```
    </details>

    <details>
    <summary>With python venv</summary>
    
    ```shell 
    python -m venv .venv 
    source .venv/bin/activate
    ```
    </details>


2. **Install dependencies**
    <details open>
    <summary>Main dependencies</summary>

    This project uses Poetry to manage its dependencies (<https://python-poetry.org/>). If you get an error with the lock file (out-dated), you can remove it: `rm poetry.lock`. 

    ```shell
    pip install poetry 
    poetry install
    ```

    </details>
    
    <details>
    <summary>Evaluation (Optional)</summary>

    Main libraries: `openai`, `ollama` and `google-genai`. These libraries are used in `ssr/evaluation.py` to call APIs. I'm using LMStudio, which you can install here: <https://lmstudio.ai/download>. However, any *OpenAI*-compatible API can work. To modify the APIs, directly change the `ssr/evaluation.py` script. 

    ```shell
    poetry install --with api
    ```
    </details>

    <details>
    <summary>Experiments (Optional)</summary>

    Main libraries: `jupyter`, `matplotlib`, and `plotly`. Used in `reproduce_experiments/**/*.ipynb`. Install this group if you want to reproduce the experiments provided in the notebooks. 

    ```shell
    poetry install --with notebook
    ```
    </details>

    <details>
    <summary>Developpement (Optional)</summary>

    Main libraries: `mypy` and `ruff`. Install this group if you want type-checking and formatting. 

    ```shell
    poetry install --with dev
    mypy --install-types
    ```
    </details>

    <details>
    <summary>Reproduce GCG baseline (Optional)</summary>

    Main libraries: `nanogcg` and `bitsandbytes`. Is used in `reproduce_results/gcg/generate.py`. Install this group if you want to generate the baseline GCG attacks. See [nanoGCG baseline](#nanogcg-baseline) for more information. 

    ```shell
    poetry install --with nanogcg
    ```
    </details>

    <details>
    <summary><i>Install everything all at once (Optional)</i></summary>


    ```shell
    poetry install --all-groups
    mypy --install-types
    ```
    </details>





3. **Add tokens**
    <details open>
    <summary>Hugging Face</summary>

    The model used are gated ones from HuggingFace, thus a token is mandatory to access them. You can get a token from your profile page: <https://huggingface.co/settings/profile>, and then ask to have access to the desired models directly on their page: 
    - Gemma 2 2b: <https://huggingface.co/google/gemma-2-2b-it>
    - Llama 3.2 1b & 3b: <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct> 
    - Qwen2.5 does not require token


    Create a `.env` file and add the tokens: 
    ```toml
    HF_TOKEN="hf_Y3JvaXNzYW50LWNyb2lzc2FudC1jcm9pc3NhbnQ="
    ```
    </details>

    <details>
    <summary>Google API (Optional)</summary>

    To evaluate the attempts with Gemini-as-a-judge, a token from Gemini's API has to be provided, more information in the [evaluation section](#evaluation). You can get a token at: <https://aistudio.google.com>.

    Add the token to your `.env` file:
    ```toml
    GOOGLE_API_KEY="c2VjcmV0cy1pbi10aGUtcmVhZG1lLWdvZXMtYnJycg="
    ```
    </details>


## Reproduce experiments 

- Introduction to the `Lens` class used in the whole project: [`reproduce_experiments/using_lens.ipynb`](reproduce_experiments/using_lens.ipynb).
- Component attribution: [`reproduce_experiments/component_attribution.ipynb`](reproduce_experiments/component_attribution.ipynb).
- Layer differences: [`reproduce_experiments/layer_diffs.ipynb`](reproduce_experiments/layer_diffs.ipynb).
- Using SSR Probe: [`reproduce_experiments/run_ssr/run_ssr_probes.ipynb`](reproduce_experiments/run_ssr/run_ssr_probes.ipynb).
- Using SSR Steering: [`reproduce_experiments/run_ssr/run_ssr_steering.ipynb`](reproduce_experiments/run_ssr/run_ssr_steering.ipynb).
- Using SSR Attention with Dazzle loss: [`reproduce_experiments/run_ssr/run_ssr_attention_dazzle.ipynb`](reproduce_experiments/run_ssr/run_ssr_attention_dazzle.ipynb).
- Using SSR Attention with Dazzle, DazzleScore, Ablation, and FullAblation losses: [`reproduce_experiments/run_ssr/run_ssr_attention_full.ipynb`](reproduce_experiments/run_ssr/run_ssr_attention_full.ipynb).
- Out of distribution discussion: [`reproduce_experiments/out_of_distribution_discussion/steering_out_of_distribution.ipynb`](reproduce_experiments/out_of_distribution_discussion/steering_out_of_distribution.ipynb).
- nanoGCG baseline: [`reproduce_experiments/gcg/README.md`](reproduce_experiments/gcg/README.md).
- Multi-layer targeting: [`reproduce_experiments/multi_layers/multi_layers.ipynb`](reproduce_experiments/multi_layers/multi_layers.ipynb).

## Generated datasets
Some generated datasets are available on HuggingFace: <https://huggingface.co/papers/2503.06269>. 

Generated jailbreaks are stored in `generated_jailbreaks/`, for the moment in JSON line format. I chose JSON line as a practical solution for storing experiences. This way, each line is self-sufficient, all the information is grouped together, and there's no need to go back and forth to other files. Lines can be load either with `load_jsonl: str (filepath) -> List[dict]` (in `ssr/__init__.py`), or `load_attempts_jsonl: str (filepath) -> Attempt` (in `ssr/evaluation/files.py`).

Attempt structure: 
```python
class Attempt(BaseModel):
    model_name: str                         # (sur)name of the model being attacked 
    vanilla_input: str                      # vanilla harmful instruction ("How to create a bomb?")
    adversarial_input: str                  # adversarial candidate generated by SSR 

    # System message used during the generation of the attack
    # If None, the chat template is used without system instruction
    system_message: Optional[str] = None

    # ProbeSSR and SteeringSSR have the same config parameters, it is still possible to distinguish
    # SteeringSSR from ProbeSSR with the initial loss and final loss, but it's kinda black magic
    ssr_implementation: Literal[
        "probe",
        "steering",
        "attention_dazzle",
        "attention_dazzle_score",
        "attention_ablation",
        "attention_full_ablation",
    ]

    inital_loss: float
    final_loss: float
    duration: int                           # duration in seconds

    config: ProbeSSRConfig | SteeringSSRConfig | AttentionSSRConfig

    responses: List[Response]


class Response(BaseModel):
    model_name: str
    response: str
    bow: int

    # System message used during inference on the API
    system_message: Optional[str] = None

    # Optional evaluations
    guard: Optional[bool] = None
    judge: Optional[JudgeScore] = None
    human: Optional[int] = None


class JudgeScore(BaseModel):
    answers: bool       
    contains_harmful_knowledge: bool
    score: int                              # Between 0 and 10, >= 8 is considered a success
}
```

For the moment (15 Mar 25), the datasets published on Hugging Face are using the `Attempt v3` class. I will try to migrate everything to the new structure in the coming weeks. The `Attempt v3`'s structure is as follows: 

```python
class Attempt(BaseModel):
    model_name: str                             
    instruction: str                            
    suffix: str                             # suffix generated by the SSR attack 

    inital_loss: float      
    final_loss: float
    duration: int                               

    config: ProbeSSRConfig | SteeringSSRConfig | AttentionSSRConfig     

    responses: List[Response]
```

## Code structure
The code is messy but the main algorithm `ssr/core.py` does not depend on the rest, and should be easily understandable/ reusable.

Other experiments rely on a custom `Lens` class, with lots of helpers. I've drawn up a dependency graph to help you see things more clearly:
![dependency graph](/assets/graph.svg)

## Trouble shooting 
Text "HELP" at thomas.winninger@telecom-sudparis.eu, or open an issue :)

Especially for the notebooks. I try to keep them updated whenever I modify the main code as well as I can, but as I do it manually, there might be some refactoring mistakes. 

## Citation
If you found this repository or the paper useful, please cite as: 
```
@article{winninger_mechanistic_2025,
      title={Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models}, 
      author={Thomas Winninger and Boussad Addad and Katarzyna Kapusta},
      year={2025},
      eprint={2503.06269},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.06269}, 
}
```
