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

from typing import List, Set
import numpy as np

from neural_compressor.common.strategy.sampler import BaseSampler
from neural_compressor.utils import logger
from neural_compressor.common.config import TuningConfig, AccuracyCriterion, TuningCriterion


class Strategy:
    def __init__(self, fp32_model, tuning_config) -> None:
        self.sampler_set: Set[BaseSampler] = set()

    def need_stop(self, acc) -> bool:
        """Determine whether to stop sampling.

        TODO (update it)
        1. should we move it into sampler class?
        The different sampler may have different stop criterion.
        quant_level = 0, continue sampling even if the accuracy is not satisfied.
        quant_level = 1 with basic, stop sampling if the accuracy is satisfied.
        
        # exit policy
        1. not_tuning(performance_only): only quantize the model without tuning or evaluation.
        2. timeout = 0, exit the tuning process once it is found model meets the accuracy requirement.
        3. max_trials, the number of the actually trials is less or equal to the max_trials
        4. quant_level = 0, to seek the better performance, continue the tuning process even if the accuracy is satisfied.
        There are two ways to use max_trials to dominate the exit policy.
        1) timeout = 0, the tuning process exit when the actual_trails_count >= max_trials or
           a quantized model meets the accuracy requirements
        2) timeout = inf, the tuning process exit until the trials_count >= max_trials
        Some use case:
        1) Ending tuning process after a quantized model meets the accuracy requirements
           max_trials = inf, timeout = 0 (by default) # the default max_trials is 100
        value of timeout. max_trials control the exit policy
        2) Even after finding a model that meets the accuracy goal, we may want to continue the
           tuning process for better performance or other objectives.
           timeout = 100000 (seconds), max_trials = 10 # Specifics a fairly large timeout, use max_trials
                                             # to control the exit policy.
        3) Only want to try a certain number of trials
           timeout = 100000 (seconds), max_trials = 3 # only want to try the first 3 trials
        """
        return False

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
        use_multi_objective = False
        if use_multi_objs:
            obj_higher_is_better = obj.get("higher_is_better", None)
            obj_weight = obj.get("weight", None)
            obj_lst = obj.get("objective", [])
            objectives = [i.lower() for i in obj_lst]
            use_multi_objective = True
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
        accuracy_criterion_conf = accuracy_criterion
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

    def _evaluate(self, eval_func, model):
        """Interface of evaluating model.
        #TODO adaptor must pass a eval_func to tuner, if it call tuner.
        # It's adaptor's responsibility to create eval_func according to the eval_metric and 


        Args:
            model (object): The model to be evaluated.

        Returns:
            Objective: The objective value evaluated.
        """
        # New add
        
        val = self.objectives.evaluate(eval_func, model)
        return val
        # End new add
        
        
        if self.eval_func:
            # TODO 
            # if options.tensorboard:
            #     # Pytorch can insert observer to model in this hook.
            #     # Tensorflow don't support this mode for now
            #     model = self.adaptor._pre_eval_hook(model)
            val = self.objectives.evaluate(self.eval_func, model if self.framework == "pytorch_ipex" else model.model)
            # TODO
            # if options.tensorboard:
            #     # post_eval_hook to deal the tensor
            #     self.adaptor._post_eval_hook(model, accuracy=val[0])
        else:
            assert not self._not_tuning, "Please set eval_dataloader and eval_metric for create eval_func"
            postprocess_cfg = None
            metric_cfg = self.eval_metric
            iteration = -1
            eval_func = create_eval_func(
                self.framework,
                self.eval_dataloader,
                self.adaptor,
                metric_cfg,
                postprocess_cfg,
                iteration,
                tensorboard=options.tensorboard,
                fp32_baseline=self.baseline is None,
            )

            if getattr(self.eval_dataloader, "distributed", False):
                if "tensorflow" in self.framework:
                    import horovod.tensorflow as hvd
                elif self.framework in ["pytorch_ipex", "pytorch", "pytorch_fx"]:
                    import horovod.torch as hvd
                else:
                    raise NotImplementedError(
                        "Currently only TensorFlow and PyTorch " "support distributed inference in PTQ."
                    )
                hvd.init()
                try:
                    len_dataloader = len(self.eval_dataloader)
                except:
                    logger.info(
                        "The length of the distributed dataloader is unknown."
                        "When the iteration of evaluation dataloader in each "
                        "process is inconsistent, an error may occur."
                    )
                else:
                    list_len_dataloader = hvd.allgather_object(len_dataloader)
                    if hvd.rank() == 0:
                        for i in range(len(list_len_dataloader) - 1):
                            if list_len_dataloader[i] != list_len_dataloader[i + 1]:
                                raise AttributeError(
                                    "The evaluation dataloader's iteration is"
                                    "different between processes, please reset "
                                    "dataloader's batch_size."
                                )
            val = self.objectives.evaluate(eval_func, model)
        if isinstance(val[0], list):
            assert all(
                [np.isscalar(i) for i in val[0]]
            ), "The eval_func should return a scalar or list of scalar, " "but not {}!".format(
                str([type(i) for i in val[0]])
            )
        else:
            assert np.isscalar(val[0]), "The eval_func should return a scalar or list of scalar, " "but not {}!".format(
                str(type(val[0]))
            )

        return val


    def add_sampler(self, sampler: BaseSampler):
        self.sampler_set.add(sampler)

    def tuning_summary(self):
        """Show tuning summary."""
        logger.info("*" * 40)
        logger.info("There are {} tuning stages in total.".format(len(self.sampler_set)))
        for i, sampler in enumerate(self.sampler_set):
            logger.info("Stage {}: {}.".format(i, sampler.name))
        logger.info("*" * 40)

    def traverse(self, quantizer):
        self.tuning_summary()
        for cur_sampler in self.sampler_set:
            logger.info("Start tuning stage: {}".format(cur_sampler.name))
            for config in cur_sampler:
                q_model = quantizer.internal_quantize(config)
                acc = self.evaluate(q_model)
                if cur_sampler.need_stop(acc):
                    return q_model
            logger.info("*" * 40)
    
    
