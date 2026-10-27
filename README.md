<div align="center">

<h1>LLMs Can Tackle High Energy Theoretical Physics, yet Gaps Remain:<br>Part I, Quantum Field Theory</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.33134">📄 Paper</a> |
  <a href="https://huggingface.co/datasets/Meikel/ProactEgoMem">🤗 Data</a> 
</p>
</div>

<!--
## 📣 News
- **[2026-10] Part I: QFT-Bench, released**
-->

<h2>Overview</h2>
<p align="center"><img src="figs/qft_model_timeline.png" width="99%"><br><b>LLM Scaling Law on QFT-Bench</b></p>

<h2> How to use the data? </h2> 

```python
from datasets import load_dataset

dataset = load_dataset("LLMsHEP/QFT-Bench", split="test")

print(f"Number of problems: {len(dataset)}")
print(dataset[0]["question"])
```



<h2>Citation</h2>
<pre>@article{llmsqft,
  title   = {LLMs Can Tackle High Energy Theoretical Physics, yet Gaps Remain: Part I, Quantum Field Theory},
  author  = {Liu, Tong and Li, Bohan and Cheng, Gongrui and Yu, Zixiong and He, Yulan and Tresp, Volker},
  journal = {arXiv preprint arXiv:2610.33134},
  url     = {https://qft-bench.github.io/},
  year    = {2026}
}</pre>
