# CFT

# Train
```shell
bash run.sh

```
# Inference 
```shell
bash infer.sh

```
You should use your own model and data path.

# Evaluate 

The original grounding method(BIGRec):
```
python evaluate_acc.py
```

The adjusted grounding method:
```
python eval_acc_2.py
```

To get the distribution of item popularity, you can run the following code:
```
python evaluate_all_pop.py
```

  journal={arXiv preprint arXiv:2410.22809},
  year={2024}
}
```

