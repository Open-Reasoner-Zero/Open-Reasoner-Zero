<div align="center">

# Open Reasoner Zero

<img src="figure/logo.jpg" width="300"/>

<div>

An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
</div>
</div>

<div align="center" style="line-height: 1;">
    <a href="https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero" style="margin: 2px;"><img alt="Code" src="https://img.shields.io/badge/Open%20Reasoner%20Zero-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
  
  <a href="https://huggingface.co/Open-Reasoner-Zero" target="_blank"><img alt="Hugging Face"
    src="https://img.shields.io/badge/HuggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor"/></a>

  <a href="https://yasminezhang.notion.site/Open-Reasoner-Zero-19e12cf72d418007b9cdebf44b0e7903" target="_blank">
  <img alt="Notion Page"
    src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white"/></a>

  <br>
  <a href="https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf"><b>Paper PDF Link [WIP]</b>👁️</a>
</div>

<div>
<br>

</div>

## Overview 🌊
We introduce **Open-Reasoner-Zero**, the first open source implementation of large-scale reasoning-oriented RL training focusing on scalability, simplicity and accessibility.

To enable broader participation in this pivotal moment we witnessed and accelerate research towards artificial general intelligence (AGI), 
we release our source code, parameter settings, training data, and model weights.
Please refer to our [paper](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf) for more insights across various model sizes. 

**Let the Reasoner-Zero tide rise!**


## Main Results 🏆

![](figure/teaser.png)

*Figure 1 | Evaluation performance of Open-Reasoner-Zero-\{7B, 32B\}. Evaluation performance of Open-Reasoner-Zero-\{7B, 32B\} on benchmarks (averaged on 16 responses) during training. Using the same base model as DeepSeek-R1-Zero-Qwen-32B, Open-Reasoner-Zero-32B achieves superior performance on AIME2024, MATH500, and GPQA Diamond benchmark-requiring only a tenth of the training steps.*

![](figure/train_curve.png)
*Figure 2 | Train-time Scale up on Train Reward and Response Length of Open-Reasoner-Zero (ORZ) - \{0.5B, 1.5B, 7B, 32B\}. Train Reward and Response Length increase steadily, demonstrating consistent scalability across model sizes. Interestingly, the ORZ-32B Response Length exhibits fluctuations without negatively impacting training stability, highlighting the robustness of our minimalist recipe.*

## Releases 📦

<strong>[2025/03/31]</strong>
We announce a major milestone for `Open-Reasoner-Zero`:

- 🌊 [Updated Paper](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf) with new results.
- 🔭 [Easy-to-use Training Scripts](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/playground):
  - [ORZ-1.5B training scripts](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/playground/orz_1p5b_ppo.py) and [ORZ-0.5B training scripts](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/playground/orz_0p5b_ppo.py) (main results in Figure 2). 
  - [Minimal resource training scripts](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/playground/orz_0p5b_ppo_1gpu.py): ORZ-0.5B can be run on a single A800/H800 card!
- 🤩 [Updated Curated Datasets](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/data): 
  - 129k data in total:
    - [original 57k data](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/data/orz_math_57k_collected.json).
    - [extended 72k data](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/data/orz_math_72k_collection_extended.json).
  - [13k hard data](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/data/orz_math_13k_collection_hard.json)
    - mined from the above 129k data
    - used in the "annealing" stage of ORZ-32B training: **AIME2024 from ~41% to ~48%**!
- 🤗 More HF Models: 
  - Updated HF Models: [`Open-Reasoner-Zero-7B`](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-7B) and [`Open-Reasoner-Zero-32B`](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B).
  - Released HF Models: [`Open-Reasoner-Zero-1.5B`](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B) and [`Open-Reasoner-Zero-0.5B`](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B).
- 🚀 Full Suite of Critic Models for in-depth research: `Open-Reasoner-Zero-Critic-`{[0.5B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-Critic-0.5B), [1.5B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-Critic-1.5B), [7B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-Critic-7B),  [32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-Critic-32B)}.

<strong>[2025/02/18]</strong>
We release `Open-Reasoner-Zero`. 

As part of this release, we open-source:
- 🌊 [Paper](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf) on our comprehensive analysis and insights in Reasoner-Zero training
- 🤗 HF Model [`Open-Reasoner-Zero-7B`](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-7B) and [`Open-Reasoner-Zero-32B`](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B)
- 🎁 [`Our curated 57k training data`](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/data)
- 📄 [Training Scripts](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/playground) to enjoy your own Reasoner-Zero journey!

## Key Features in Codebase 🔑

- Adopt single controller trainer design, flexible and researcher-friendly.
- Colocate training and generation in the same GPUs to maximize GPU utilization.

## Getting Started 🚀
### Data

We release all of curated high-quality training data in the [`data`](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/data) folder:
* curated 129k data:
  * [original 57k](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/data/orz_math_57k_collected.json), collected from various sources, including AIME (up to 2023), MATH, Numina-Math collection and Tulu3 MATH.
  * [extended 72k](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/data/orz_math_72k_collection_extended.json), mainly cleaned from OpenR1-Math-220k.
* [hard 13k](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/data/orz_math_13k_collection_hard.json), mined from the first stage of ORZ-32B training.

The details for how to collect data are described in our [paper](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf).

### Installation & Training Scripts
We release our [Dockerfile](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/docker/Dockerfile) in [docker](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/docker) folder to facilitate the reproducibility of our training.

To install the package, run:
```bash
pip install -e .
```

#### Start ORZ-32B PPO Training
Here are the starting commands in 16 nodes. 

First on master node, run:
```bash
ray start --head
# you will see logging like:
# Next steps
#  To add another node to this Ray cluster, run
#    ray start --address='<master-node-ip>:<master-node-port>'
```

then on all other nodes, run:
```bash
ray start --address='<master-node-ip>:<master-node-port>' # <master-node-ip> and <master-node-port> are from above loggings!
```

finally on master node, just run:
```bash
python -m playground.orz_32b_ppo
```
Your training log will be shown in the master node terminal.

------

#### Start ORZ-0.5B PPO Training
You can start the ORZ-0.5B PPO training in single A800/H800 node:
```bash
python -m playground.orz_0p5b_ppo
```

You can even run in **a single A800/H800**: 
```bash
python -m playground.orz_0p5b_ppo_1gpu
```

note: since we are not in multi-node setting, no `ray start` like logics are needed.

------

#### Start ORZ-7B PPO Training

Multi-node Training on 4 nodes:
```bash
# set up for multi-node training
ray start --head # on master node
ray start --address='<master-node-ip>:<master-node-port>' # then on other nodes
```
then on master node, run:
```bash
python -m playground.orz_7b_ppo
```

Your training log will be shown in the master node terminal.

-----

#### Start ORZ-1.5B PPO Training

Multi-node Training on 2 nodes:
```bash
# set up for multi-node training
ray start --head # on master node
ray start --address='<master-node-ip>:<master-node-port>' # then on other nodes
```
then on master node, run:
```bash
python -m playground.orz_1p5b_ppo
```

----

#### Debug Settings
In the code, we leave an environment variable `DEBUG_MODE` to run in debug setting for researcher to iterate. (Thought for now, we recommend using `python -m playground.orz_0p5b_ppo_1gpu` for debugging.)
The debug running command examples:
```bash
# NOTE: just for debug, not final setting!

## Debug command in a single GPU with `EleutherAI/pythia-14m`
DEBUG_MODE=True python -m playground.orz_14m_ppo_mini

## Debug command in a single node (8 GPUs) with `Qwen/Qwen2.5-7B`
DEBUG_MODE=True python -m playground.orz_7b_ppo
```

## Acknowledgements 💖 

- This work was supported by computing resources and valuable feedback provided by [StepFun](https://www.stepfun.com/) and Tsinghua University.
- Our training framework is built on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [vllm](https://github.com/vllm-project/vllm), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) and [ray](https://github.com/ray-project/ray).
- Our model is based on [Qwen2.5 Series](https://qwenlm.github.io/blog/qwen2.5-llm/) of **base models**, including [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) and [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B).
- We thank [Project Numina](https://projectnumina.ai/), [Tulu3](https://allenai.org/blog/tulu-3-technical) and [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) for their collected open sourced data.

## Advertisement Time 📣

We are hiring talented researchers and engineers to join our team. If you are interested in our project and would like to contribute to the reasoner scale-up all the way to AGI, please feel free to reach out to us at hanqer@stepfun.com


[![Star History Chart](https://api.star-history.com/svg?repos=Open-Reasoner-Zero/Open-Reasoner-Zero&type=Timeline)](https://star-history.com/#Open-Reasoner-Zero/Open-Reasoner-Zero&Timeline)

## Community Discussions 🍺

We have several wechat groups to help discussions and sharing, you can scan the QR code below to join the latest group.

<img src="figure/WeChatGroup.png" width="300" style="display: block; margin: 0 auto;"/>

## Citation

```bibtex
@misc{OpenReasonerZero2025,
  title={Open-Reasoner-Zero: An Open Source Approach to Scaling Reinforcement Learning on the Base Model},
  author={Jingcheng Hu and Yinmin Zhang and Qi Han and Daxin Jiang and Xiangyu Zhang, Heung-Yeung Shum},
  year={2025},
  howpublished={\url{https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero}},
}
```
