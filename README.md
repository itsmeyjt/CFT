# CFT
This is the implementation of our work **[Causality-Enhanced Behavior Sequence Modeling in LLMs for Personalized Recommendation](https://arxiv.org/abs/2410.22809)**

# Data Preprocessing 

Please refer to this [repo](https://github.com/SAI990323/DecodingMatters).

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

If you're using this code in your research or applications, please cite our paper using this BibTeX:
```bibtex
@article{zhang2024causality,
  title={Causality-Enhanced Behavior Sequence Modeling in LLMs for Personalized Recommendation},
  author={Zhang, Yang and You, Juntao and Bai, Yimeng and Zhang, Jizhi and Bao, Keqin and Wang, Wenjie and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2410.22809},
  year={2024}
}
```

