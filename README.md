# Let's Think Dot by Dot: Hidden Computation in Transformer Language Models

Chain-of-thought responses from language models improve performance across most benchmarks. However, it remains unclear to what extent these performance gains can be attributed to human-like task decomposition or simply the greater computation that additional tokens allow. We show that transformers can use meaningless filler tokens (e.g., '......') in place of a chain of thought to solve two hard algorithmic tasks they could not solve when responding without intermediate tokens. However, we find empirically that learning to use filler tokens is difficult and requires specific, dense supervision to converge. We also provide a theoretical characterization of the class of problems where filler tokens are useful in terms of the quantifier depth of a first-order formula. For problems satisfying this characterization, chain-of-thought tokens need not provide information about the intermediate computational steps involved in multi-token computations. In summary, our results show that additional tokens can provide computational benefits independent of token choice. The fact that intermediate tokens can act as filler tokens raises concerns about large language models engaging in unauditable, hidden computations that are increasingly detached from the observed chain-of-thought tokens.

Our paper is [here](https://arxiv.org/abs/2404.15758)

# Setup
pip install -r requirements.txt

# Example Usage
```
python -m scripts.data_match3 --name minidata --length 10 --train_samples 100000
python -m scripts.run_match3 -dn minidata -e 1 -de minidata -ma P -m 'llama' -cc llama_d384l4h6.json
```

# Citation

If you use our work, consider citing it using the below!

```
@article{pfau2024lets,
      title={Let's Think Dot by Dot: Hidden Computation in Transformer Language Models}, 
      author={Jacob Pfau and William Merrill and Samuel R. Bowman},
      year={2024},
      eprint={2404.15758},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
