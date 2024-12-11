# NeuroXAI: Adaptive, Robust, Explainable Surrogate Framework for Determination of Channel Importance in EEG Application

>**Choel-Hui Lee, Daesun Ahn, Hakseung Kim, Eun Jin Ha, Jung-Bin Kim and Dong-Joo Kim**

[[`Paper`](https://www.sciencedirect.com/science/article/pii/S0957417424022310)]] [[`BibTeX`](#license-and-citation-)] [[`Paper With Code`](https://paperswithcode.com/paper/neuroxai-adaptive-robust-explainable)]

## Abstract 🔥
Electroencephalogram (EEG)-based applications often require numerous channels to achieve high performance, which limits their widespread use. Various channel selection methods have been proposed to identify minimum EEG channels without compromising performance. However, most of these methods are limited to specific data paradigms or prediction models. We propose NeuroXAI, a novel method that identifies channel importance regardless of the type of EEG application. It integrates the surrogate analysis algorithm to optimize EEG signals and the data sampling algorithm, which effectively selects from highly voluminous EEG data. The efficacy of channel selection via the proposed method was evaluated through three datasets acquired under different paradigms (motor imagery, steady-state visually evoked potentials, and event-related potentials). On datasets based on these paradigms, NeuroXAI-based channel selection reduced the number of channels while maintaining or enhancing performance. The advantages of the proposed method include enhanced performance and robustness over varying data paradigms and the type of prediction model. The XAI technique enables intuitive interpretation regarding the constructed model operation, making it applicable in various fields such as model debugging and model interpretation. NeuroXAI has the potential to be used as a practical tool to develop better EEG applications.

![NeuroXAI Architecture](https://github.com/dlcjfgmlnasa/NeuroXAI/blob/main/figure/figure1.png)

## Installation 💿
The code requires `python>=3.5`, as well as `lime` and `mne >= 1.8.0`.

install *NeuroXAI*
```bash
>> pip install neuroxai
```

or clone the repository locally and install with
```bash
>> git clone git@github.com:dlcjfgmlnasa/NeuroXAI.git
>> cd NeuroXAI
>> pip install -e .
```

## Usage 🤖
**(Example 1)** *NeuroXAI* for sinlge EEG sample

```python
from neuroxai.explanation import BrainExplainer

kernel_size = 25             # exponential smoothing kernel size
num_samples = 500            # training sample size
replacement_method = 'mean'  # permutation type {mean, zero, noise}
label_idx_list = [i for i, _ in enumerate(class_names)]

explainer = BrainExplainer(kernel_size=kernel_size, class_names=class_names)
exp = explainer.explain_instance(data=x,                       # x => (channel_num, signal_length)
                                 classifier_fn=classifier_fn,  # pretrained model
                                 labels=label_idx_list,
                                 num_samples=num_samples,
                                 replacement_method=replacement_method)

exp.as_list(label=class_idx)                                   # [result] => channel importance for each class
```

**(Example 2)** *NeuroXAI* for entire EEG samples
```python
from neuroxai.explanation import BrainExplainer
from neuroxai.explanation import GlobalBrainExplainer

kernel_size = 25             # exponential smoothing kernel size
num_samples = 500            # training sample size
replacement_method = 'mean'  # permutation type {mean, zero, noise}

explainer = BrainExplainer(kernel_size=kernel_size, class_names=class_names)
global_exp = GlobalBrainExplainer(explainer=explainer)
global_exp.explain_instance(x=x,                          # x => (epoch_size, channel_num, signal_length)
                            y=y,                          # y => (epoch_size, 1)
                            classifier_fn=classifier_fn,  # pretrained model
                            num_samples=num_samples,
                            replacement_method=replacement_method)

global_exp.explain_classes_channel_importance()           # [result1] => Global channel importance for each class
global_exp.explain_global_channel_importance()            # [result2] => Global channel importance
```

## Colab Tutorials <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white"/>
- [NeuroXAI for FBCSP (Filter Bank Common Spatial Pattern)](https://colab.research.google.com/drive/1iwLv_ldgXw-vT3zWjrejyMcCNcUdfU6l?usp=sharing) 
- NeuroXAI for EEGNet
- NeuroXAI based Channel Selection
- NeuroXAI based Model Debugging

## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code
```
@article{lee2025neuroxai,
  title={NeuroXAI: Adaptive, robust, explainable surrogate framework for determination of channel importance in EEG application},
  author={Lee, Choel-Hui and Ahn, Daesun and Kim, Hakseung and Ha, Eun Jin and Kim, Jung-Bin and Kim, Dong-Joo},
  journal={Expert Systems with Applications},
  volume={261},
  pages={125364},
  year={2025},
  publisher={Elsevier}
}
```
