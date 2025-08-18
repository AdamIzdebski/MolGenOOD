# SyntheticOOD
A benchmark suite for evaluating molecular property prediction models under out-of-distribution shifts, highlighting performance degradation across distributional splits induced by synthetic molecules.


---
## Installation


**Local development**
```bash
uv pip install -e .
```

---
Logic: I will train and save probing methods on train from ZINC:
- kNN, RF, GB and later Chemprop and Hyformer

I will load the probes to make predictions across datasets and evaluate. 