## LLM-for-Chip

LLM4Chip-R1: Leveraging LLM agents and Reinforcement Learning to design and optimize semiconductor devices and chips, such as semiconductor lasers.

This is the code repo for our paper "LLM4Chip-R1: End-to-End Design and Generation of Semiconductor Chips with LLMs Boosted by Reward-Driven Reinforcement Learning" under review at KDD'26.

<img width="780" height="341" alt="Screenshot 2025-12-13 at 20 56 22" src="https://github.com/user-attachments/assets/c26eee3b-074b-4434-9e93-0623659ed281" />
<img width="823" height="485" alt="Screenshot 2025-12-13 at 20 56 33" src="https://github.com/user-attachments/assets/b3620ab3-bfbc-4620-abd8-f1f3165c2c89" />

### Setup

**Requirements:** Python 3.9 or newer (3.10–3.12 are fine), 64-bit. A GPU is optional; PyTorch will use CUDA if installed.

1. Create and activate a virtual environment (recommended):

```bash
cd /path/to/LLM-4-Chip-main
python3 -m venv .venv
source .venv/bin/activate   # windows: .venv\Scripts\activate
```

2. Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For **GPU-accelerated PyTorch**, install the CUDA build that matches your system from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) *after* step 2 (or reinstall `torch` with the provided `pip` command there).

3. **MIT Meep (FDTD)** is used by the `meep_*.py` and `PCSEL_hex_side.py` scripts. It is not installed via `requirements.txt` because installs are platform-specific. Typical approach:

```bash
conda install -c conda-forge pymeep
```

See the [Meep installation guide](https://meep.readthedocs.io/en/latest/Installation/).

4. **RL scripts (`DQN_chip_optimization.py`, `Bayesian_optimization_chip.py`)** register a Gym environment `Fdtd_NB-v0` with `entry_point='envs:FdtdEnv'`. You will use the provided module named `envs` containing `FdtdEnv` (the FDTD-backed environment) and the moduled named 'src' containing FdtdRlNanobeam.py and run with that package on `PYTHONPATH`, for example:

```bash
export PYTHONPATH="/path/to/parent/of/envs:$PYTHONPATH"
python DQN_chip_optimization.py
```

That environment code is not included in this repository snapshot; without it, those scripts will fail at `gym.make('Fdtd_NB-v0')`.

### Running the code

| Component | Command / notes |
|-----------|------------------|
| DQN training | `python DQN_chip_optimization.py` (requires `envs.FdtdEnv` as above). Logs: TensorBoard default dir (`runs/`). |
| Multi-objective Bayesian optimization | `python Bayesian_optimization_chip.py` (same `envs` requirement). |
| Meep simulations | `python meep_1.py`, `python meep_pcsel_latest.py`, or `python PCSEL_hex_side.py` (requires Meep installed). |
| Vision LLM fine-tuning (notebook) | Open `Photonic_LLM_Sim.ipynb` in Jupyter or VS Code. It expects **Unsloth** and related packages (see the notebook’s install cells); that workflow is separate from `requirements.txt`. |

View TensorBoard logs, for example:

```bash
tensorboard --logdir=runs
```

### Dependency check

After `pip install -r requirements.txt`, `pip check` may report `grpcio ... is not supported on this platform` on some macOS setups; this is a known false positive for TensorBoard’s dependency. If `import tensorboard` and `import grpc` succeed, you can ignore it.

### Abstract

Conventional design and optimization of Photonic Crystal Surface Emitting Lasers
(PCSEL) usually requires expert knowledge in semiconductor physics and optimization
algorithms, which is also known as the inverse design problem. However,
with the trend towards automation and depersonalization of the entire integrated
circuits (IC) industry, the conventional method, with the drawback of being relatively
labor-intensive and sub-optimal, warrants further refinement. This technical
dilemma remained until the emergence of Large Language Models (LLMs), such
as OpenAI’s ChatGPT and Google’s Bard. This paper explores the possibility of
applying LLMs to machine learning-based design and optimization of PCSELs.
Specifically, we utilize GPT-3.5 and GPT-4. By simply having conversations, GPT
assisted us with writing Finite Difference Time Domain (FDTD) simulation code
and deep reinforcement learning code to acquire the optimized PCSEL solution,
spanning from the proposition of ideas to the realization of algorithms. Given that
GPT will perform better when given detailed and specific questions, we break down
the PCSEL design problem into a series of sub-problems and converse with GPT
by posing open-ended heuristic questions rather than definitive commands. This
paper shows that LLMs, such as ChatGPT, can guide the nanophotonic design and
optimization processes, on both the conceptual and technical level, and we propose
new human–AI co-design strategies and show their practical implications. We
achieve a significant milestone for the first step towards an automated end-to-end
nanophotonic design and production pipeline.

### Algorithms and software

Deep Q learning (DQN) and [MIT Meep](https://meep.readthedocs.io/en/latest/)

PyTorch was used as the ML library and OpenAI Gym was used for building the envs.  
Meep FDTD was used as the environment for simulating nanophotonics.

### File structure

```
.
├── DQN_chip_optimization.py      # dqn training
├── Bayesian_optimization_chip.py # multi-objective bayesian optimization
├── meep_1.py                     # meep fdtd (human-written baseline)
├── meep_pcsel_latest.py          # meep fdtd (gpt-assisted variant)
├── PCSEL_hex_side.py             # hex lattice pcsel meep script
├── Photonic_LLM_Sim.ipynb        # qwen / unsloth vision fine-tuning notebook
├── Multi-turn Convo/             # conversation logs with gpt
├── requirements.txt              # python dependencies for rl + bo + plotting
└── README.md
```

### Citation

If you used our code or idea for your research, please consider citing the paper as:
TBD
