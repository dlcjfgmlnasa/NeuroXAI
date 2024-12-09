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
