# NeuroXAI: Adaptive, Robust, Explainable Surrogate Framework for Determination of Channel Importance in EEG Application

>**Choel-Hui Lee, Daesun Ahn, Hakseung Kim, Eun Jin Ha, Jung-Bin Kim and Dong-Joo Kim**

**Full code coming soon^^**

## Introduction 🔥
Electroencephalogram (EEG)-based applications often require numerous channels to achieve high performance, which limits their widespread use. Various channel selection methods have been proposed to identify minimum EEG channels without compromising performance. However, most of these methods are limited to specific data paradigms or prediction models. We propose NeuroXAI, a novel method that identifies channel importance regardless of the type of EEG application. It integrates the surrogate analysis algorithm to optimize EEG signals and the data sampling algorithm, which effectively selects from highly voluminous EEG data. The efficacy of channel selection via the proposed method was evaluated through three datasets acquired under different paradigms (motor imagery, steady-state visually evoked potentials, and event-related potentials). On datasets based on these paradigms, NeuroXAI-based channel selection reduced the number of channels while maintaining or enhancing performance. The advantages of the proposed method include enhanced performance and robustness over varying data paradigms and the type of prediction model. The XAI technique enables intuitive interpretation regarding the constructed model operation, making it applicable in various fields such as model debugging and model interpretation. NeuroXAI has the potential to be used as a practical tool to develop better EEG applications.

## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code:
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
