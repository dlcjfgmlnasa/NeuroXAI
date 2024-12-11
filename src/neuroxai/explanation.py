# -*- coding:utf-8 -*-
import numpy as np
from lime import explanation
from lime import lime_base
from typing import Callable, Optional, List
from sklearn.metrics.pairwise import pairwise_distances


class BrainExplainer(object):
    def __init__(self, kernel_width: int = 25, class_names: Optional[List] = None):
        # exponential kernel
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        self.base = lime_base.LimeBase(kernel_fn=kernel)
        self.class_names = class_names

    def explain_instance(self,
                         data: List[float],
                         classifier_fn: Callable,
                         num_samples: int,
                         labels=(1,),
                         replacement_method='mean',
                         top_labels=None):

        permutations, predictions, distances = self.data_labels_distances(
            time_series=data,
            classifier_fn=classifier_fn,
            num_samples=num_samples,
            replacement_method=replacement_method
        )
        if self.class_names is None:
            self.class_names = [str(x) for x in range(predictions[0].shape[0])]

        domain_mapper = explanation.DomainMapper()
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = predictions[0]

        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels:]
            ret_exp.top_labels = list(predictions)
            ret_exp.top_labels.reverse()

        for label in labels:
            (ret_exp.intercept[int(label)],
             ret_exp.local_exp[int(label)],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                neighborhood_data=permutations,
                neighborhood_labels=predictions,
                distances=distances,
                label=label,
                num_features='none',
                feature_selection='none',
                model_regressor=None,
            )
        return ret_exp

    @staticmethod
    def data_labels_distances(time_series,
                              classifier_fn,
                              num_samples,
                              replacement_method='mean'):
        def distance_fn(x):
            return pairwise_distances(x, x[0].reshape(1, -1), metric='cosine').ravel()

        data_dims = time_series.shape[0]
        feature_range = range(data_dims)
        detect_per_slice = np.random.randint(1, data_dims + 1, num_samples - 1)
        perturbation_matrix = np.ones((num_samples, data_dims))
        original_data = [time_series.copy()]

        for i, num_inactive in enumerate(detect_per_slice, start=1):
            # choose random slices indexes to deactivate
            inactive_idx = np.random.choice(feature_range, num_inactive, replace=False)
            perturbation_matrix[i, inactive_idx] = 0
            tmp_series = time_series.copy()

            # permutation signals
            if replacement_method == 'mean':
                tmp_series[inactive_idx] = tmp_series.mean()
            elif replacement_method == 'noise':
                tmp_series[inactive_idx] = np.random.uniform(tmp_series.min(), tmp_series.max())
            elif replacement_method == 'zero':
                tmp_series[inactive_idx] = 0

            original_data.append(tmp_series)

        original_data = np.array(original_data)
        predicate = classifier_fn(original_data)
        distances = distance_fn(perturbation_matrix)
        return perturbation_matrix, predicate, distances


class GlobalBrainExplainer(object):
    def __init__(self, explainer: BrainExplainer):
        super().__init__()
        self.explainer = explainer
        self.class_names = explainer.class_names
        self.class_idx_list = [i for i in range(len(explainer.class_names))]
        self.exp_inst_bulk = []
        self.prob_threshold = None

    def explain_instance(self,
                         x: List[float], y: List[int],
                         classifier_fn: Callable,
                         num_samples: int,
                         replacement_method='mean'):
        self.exp_inst_bulk = []
        for data, label in zip(x, y):
            exp = self.explainer.explain_instance(data=data,
                                                  classifier_fn=classifier_fn,
                                                  labels=self.class_idx_list,
                                                  num_samples=num_samples,
                                                  replacement_method=replacement_method)
            self.exp_inst_bulk.append({'label': label, 'exp': exp})

        prob_list = []
        for exp_inst in self.exp_inst_bulk:
            label, exp = exp_inst['label'], exp_inst['exp']
            if label == exp.predict_proba.argmax():
                prob = exp.predict_proba[exp.predict_proba.argmax()]
                prob_list.append(prob)
        self.prob_threshold = np.mean(prob_list)

    def explain_classes_channel_importance(self):
        if len(self.exp_inst_bulk) == 0:
            assert 'Please execute "explain_instance" first.'

        feature_names = [t[0] for t in self.exp_inst_bulk[0]['exp'].as_list()]
        feature_names.sort()

        temp = {class_name: {feature_name: [] for feature_name in feature_names}
                for class_name in self.class_names}

        for exp_inst in self.exp_inst_bulk:
            label, exp = exp_inst['label'], exp_inst['exp']
            if exp.predict_proba.argmax() == label and \
                    exp.predict_proba[exp.predict_proba.argmax()] >= self.prob_threshold:
                for feature_idx, weight in exp.as_list(label=label):
                    temp[self.class_names[label]][feature_names[feature_idx]].append(
                        weight
                    )

        classes_channel_importance = {class_name: {feature_name: None for feature_name in feature_names}
                                      for class_name in self.class_names}
        for class_name in self.class_names:
            for feature_name, feature_im in temp[class_name].items():
                classes_channel_importance[class_name][feature_name] = np.mean(feature_im)

        return classes_channel_importance

    def explain_global_channel_importance(self):
        feature_names = [t[0] for t in self.exp_inst_bulk[0]['exp'].as_list()]
        feature_names.sort()

        total_channel_importance = {feature_name: [] for feature_name in feature_names}
        for exp_inst in self.exp_inst_bulk:
            label, exp_inst = exp_inst['label'], exp_inst['exp']
            if exp_inst.predict_proba.argmax() == label and \
                    exp_inst.predict_proba[exp_inst.predict_proba.argmax()] >= self.prob_threshold:

                for values in exp_inst.as_map().values():
                    channels = np.array([value[0] for value in values])
                    weights = np.array(np.abs([value[1] for value in values]))

                    for channel_idx, weight in zip(channels, weights):
                        total_channel_importance[feature_names[channel_idx]].append(
                            weight
                        )

        total_channel_importance = {feature_name: np.mean(feature_im) for feature_name, feature_im
                                    in total_channel_importance.items()}
        return total_channel_importance
