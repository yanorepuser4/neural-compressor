# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from neural_compressor.common.config import TuningCriterion, AccuracyCriterion

class FakeTuningConfig:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "FakeTuningConfig"

    def _set_objectives(
        accuracy_criterion: AccuracyCriterion,
        tuning_criterion: TuningCriterion
        ):
        """
        TODO
        1. add the implementation
            1) add one metric
            2) multiple metrics
        2. should we move it as the function of sampler?
        """
        ### New Impl
        
        from neural_compressor.common.objective import MultiObjective
        # set objectives
        def _use_multi_obj_check(obj):
            if isinstance(obj, list):
                return len(obj) > 1
            elif isinstance(obj, dict):
                return len(obj.get("objective", [])) > 1

        higher_is_better = bool(accuracy_criterion.higher_is_better)
        obj_higher_is_better = None
        obj_weight = None
        obj = tuning_criterion.objective
        use_multi_objs = _use_multi_obj_check(obj)
        if use_multi_objs:
            obj_higher_is_better = obj.get("higher_is_better", None)
            obj_weight = obj.get("weight", None)
            obj_lst = obj.get("objective", [])
            objectives = [i.lower() for i in obj_lst]
        else:
            objectives = [val.lower() for val in obj]

        # set metric
        metric_name = ["Accuracy"]
        metric_criterion = [higher_is_better]
        metric_weight = None
        use_multi_metrics = False
        # NOTE: is not correct
        #
        _eval_metric = tuning_criterion.objective
        if _eval_metric:
            # metric name
            # 'weight','higher_is_better', 'metric1', 'metric2', ...
            if len(_eval_metric.keys()) >= 4:
                metric_name = _eval_metric.keys() - {"weight", "higher_is_better"}
                use_multi_metrics = True
            metric_higher_is_better = _eval_metric.get("higher_is_better", None)
            # metric criterion
            if use_multi_metrics:
                if metric_higher_is_better is not None:
                    metric_criterion = [metric_higher_is_better] * len(metric_name)
                else:
                    metric_criterion = [True] * len(metric_name)
            # metric weight
            metric_weight = _eval_metric.get("weight", None)

        accuracy_criterion = {"relative": 0.01, "higher_is_better": True}
        accuracy_criterion[accuracy_criterion.criterion] = accuracy_criterion.tolerable_loss
        accuracy_criterion["higher_is_better"] = accuracy_criterion.higher_is_better
        objectives = MultiObjective(
            objectives=objectives,
            accuracy_criterion=accuracy_criterion,
            metric_criterion=metric_criterion,
            metric_weight=metric_weight,
            obj_criterion=obj_higher_is_better,
            obj_weight=obj_weight,
        )
        return objectives
