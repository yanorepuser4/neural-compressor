import tensorflow as tf
import numpy as np
import logging
import time
import math
from typing import Optional, Literal
from dataclasses import dataclass, field
from itertools import chain
import transformers
import datasets
from transformers.tf_utils import stable_softmax, shape_list, check_embeddings_within_bounds
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TFTrainingArguments,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    set_seed,
)
from datasets import load_dataset
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r benchmarks/language_modeling/tensorflow/gpt_j/requirements.txt")
MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to use.
    """

    model_name_or_path: Optional[str] = field(
        default="EleutherAI/gpt-j-6B",
        metadata={
            "help": (
                "The model checkpoint for GPT-J weights."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    precision: Optional[str] = field(
        default="fp32",
        metadata={"help": "The precision that we want to run with."},
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for evaluation.
    """

    dataset_name: Optional[str] = field(
        default="EleutherAI/lambada_openai", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
def _expand_inputs_for_generation(
        input_ids: Optional[tf.Tensor] = None,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        expand_in_new_axis: bool = False,
        **model_kwargs,
    ):
        """
        Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...] or [batch_size, expand_size, ...],
        depending on `expand_in_new_axis`. Beam-based approaches expect this function to be used with
        `expand_in_new_axis=True`
        """

        def _expand_tensor(tensor: tf.Tensor):
            if expand_in_new_axis:
                shape = shape_list(tensor)
                return tf.broadcast_to(tensor[:, None], (shape[0], expand_size) + tuple(shape[1:]))
            else:
                return tf.repeat(tensor, expand_size, axis=0)

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], tf.Tensor):
                    dict_to_expand[key] = _expand_tensor(dict_to_expand[key])
            return dict_to_expand

        if input_ids is not None:
            input_ids = _expand_tensor(input_ids)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs
    
    
class Inference():
    def __init__(self, warmup_steps=1, batch_size=1, steps=1, input_tokens=32, token_to_generate=32, data=None) -> None:
        self.dur = []
        self.infer = None
        self.batch_size = batch_size
        self.num_beams = 4
        self.warmup_steps = warmup_steps
        self.steps = steps
        self.input_tokens = input_tokens
        self.token_to_generate = token_to_generate
        self.data = data
        self.correct = 0
    
    def _extract_past_from_model_output(self, outputs):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs['past_key_values']
        #elif "mems" in outputs:
            #past_key_values = outputs.mems
        #elif "past_buckets_states" in outputs:
            #past_key_values = outputs.past_buckets_states
        return past_key_values

    def prepare_attention_mask_for_generation(
        self,
        inputs: tf.Tensor,
        pad_token_id=50256,
        eos_token_id=50256,
    ) -> tf.Tensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in (tf.int32, tf.int64)
        is_pad_token_in_inputs = (pad_token_id is not None) and tf.math.reduce_any(inputs == pad_token_id)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return tf.cast(tf.math.not_equal(inputs, pad_token_id), dtype=tf.int32)
        else:
            return tf.ones(inputs.shape[:2], dtype=tf.int32)
        
    '''
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    '''
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if not self.is_first_iteration:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if not self.is_first_iteration:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)
        #"position_ids": position_ids,
        #"past_key_values": past_key_values,
            #"use_cache": use_cache,
            #"token_type_ids": token_type_ids,
        if self.is_first_iteration:
            return {
                "input_ids": inputs,
                "attention_mask": attention_mask
            }
        else:
            return {
                "input_ids": inputs,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values
            }
    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder: bool = False
    ):
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = tf.concat(
                    [attention_mask, tf.ones((shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1
                )

        return model_kwargs
    
    def greedy_search_cond_fn(self, generated, finished_sequences, cur_len, model_kwargs):
        """state termination condition fn."""
        return ~tf.reduce_all(finished_sequences)

        # define condition fn
    def greedy_search_body_fn(self, generated, finished_sequences, cur_len, model_kwargs):
        """state update fn."""
        if model_kwargs.get("past_key_values") is None:
            input_ids = generated[:, :cur_len]
        else:
            input_ids = tf.expand_dims(generated[:, cur_len - 1], -1)
        #model_inputs = {'input_ids': input_ids, 'attention_mask': model_kwargs['attention_mask']}
        model_inputs = self.prepare_inputs_for_generation(input_ids, use_cache=True, **model_kwargs)
    
        start = time.time()
        model_outputs = self.infer(**model_inputs)
        end = time.time()

        self.dur += [end - start]
        if 'logits' in model_outputs.keys():
            next_token_logits = model_outputs['logits'][:, -1]
        else:
            next_token_logits = model_outputs['output_0'][:, -1]

        # pre-process distribution
        next_tokens_scores = next_token_logits

        # argmax
        next_tokens = tf.argmax(next_tokens_scores, axis=-1, output_type=tf.int32)
        #print("next token before")
        #print(next_tokens)

        pad_token_id = 50256
        eos_token_id = [50256]

        unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
        next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
        next_token_is_eos = tf.math.reduce_any(
            tf.equal(
                tf.broadcast_to(next_tokens, (len(eos_token_id), self.batch_size)), tf.expand_dims(eos_token_id, -1)
            ),
            axis=0,
        )
        finished_sequences = finished_sequences | next_token_is_eos
        
        # update `generated` and `cur_len`
        update_indices = tf.stack([tf.range(self.batch_size), tf.broadcast_to(cur_len, [self.batch_size])], axis=-1)
        
        generated = tf.tensor_scatter_nd_update(tensor=generated, indices=update_indices, updates=next_tokens)
        cur_len += 1
        
        
        #update attention_masks
        '''
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = tf.concat(
                    [attention_mask, tf.ones((attention_mask.shape[0], 1), dtype=tf.int32)], axis=-1
                )
        '''
        model_kwargs = self._update_model_kwargs_for_generation(
                    model_outputs, model_kwargs, is_encoder_decoder=False
                )
        
        
        return generated, finished_sequences, cur_len, model_kwargs
        
    def generate(self, data, model):
        #input_ids = tf.convert_to_tensor([data[:-1]], dtype=tf.int32)
        input_ids = tf.convert_to_tensor([data[:-1]], dtype=tf.int32)
        #4. Define model inputs
        #input_ids = tf.convert_to_tensor(data, dtype=tf.int32)
        
        pad_token_id = 50256
        #cur_len = data.shape[-1]
        cur_len = len(data) - 1
        input_ids_padding = tf.ones((self.batch_size, self.token_to_generate), dtype=tf.int32) * (pad_token_id or 0)
        #input_ids_padding = tf.ones((self.batch_size, 1), dtype=tf.int32) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        
        model_kwargs = {'attention_mask': self.prepare_attention_mask_for_generation(input_ids)}
        #finished_sequences = tf.convert_to_tensor([False])
        finished_sequences = tf.zeros((self.batch_size,), dtype=tf.bool)
        
        self.is_first_iteration = 1
        if isinstance(model, str):
            loaded_model = tf.saved_model.load(model)
        else:
            loaded_model = model
        self.infer = loaded_model.signatures["serving_first_iteration"]
        # 1st generation step has to be run before to initialize `past_key_values`
        generated, finished_sequences, cur_len, model_kwargs = self.greedy_search_body_fn(
            generated, finished_sequences, cur_len, model_kwargs
        )
        self.infer = loaded_model.signatures["serving_default"]
        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        self.is_first_iteration = 0
        maximum_iterations = self.token_to_generate - 1
        generated, _, cur_len, _ = tf.while_loop(
            self.greedy_search_cond_fn,
            self.greedy_search_body_fn,
            (generated, finished_sequences, cur_len, model_kwargs),
            maximum_iterations=maximum_iterations,
        )
        

        return generated

   
    @staticmethod
    def _gather_beams(nested, beam_indices, batch_axis=0):
        """Gathers the beam slices indexed by beam_indices into new beam array."""

        def gather_fn(tensor):
            if batch_axis > 0:
                # pushes all dimentions before the batch to the end, so we get (batch, beam_id, ...)
                perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
                tensor = tf.transpose(tensor, perm=perm)

            gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)
            if batch_axis > 0:
                # transposes back to the original dimensions
                perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
                perm = tf.math.invert_permutation(perm)
                gathered_tensor = tf.transpose(gathered_tensor, perm=perm)

            return gathered_tensor

        return tf.nest.map_structure(gather_fn, nested)
    
    def beam_search(self, data, model):
        self.is_first_iteration = 1
        def flatten_beam_dim(tensor, batch_axis=0):
            """Flattens the first two dimensions of a non-scalar array."""
            shape = shape_list(tensor)
            return tf.reshape(
                tensor,
                shape[:batch_axis] + [shape[batch_axis] * shape[batch_axis + 1]] + shape[batch_axis + 2 :],
            )

        def unflatten_beam_dim(tensor, num_beams, batch_axis=0):
            """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
            shape = shape_list(tensor)
            return tf.reshape(tensor, shape[:batch_axis] + [-1, num_beams] + shape[batch_axis + 1 :])
            #return tf.reshape(tensor, [self.batch_size, num_beams] + shape[1:])
        
        
        input_ids = tf.convert_to_tensor(data, dtype=tf.int32)
        num_return_sequences = 1
        max_length = self.token_to_generate + self.input_tokens
        pad_token_id = 50256
        length_penalty = 1.0
        early_stopping = False
        cur_len = data.shape[-1]
        
        model_kwargs = {'attention_mask': self.prepare_attention_mask_for_generation(input_ids)}
        #TODO: May need to initialize model_kwargs = {'attention_mask': self.prepare_attention_mask_for_generation(input_ids), 'past_key_values': None} to use_cache
        #TODO: change expand_size back to 1
        input_ids, model_kwargs = _expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=1,
                is_encoder_decoder = False,
                expand_in_new_axis=True,
                **model_kwargs,
            )

        
        # per batch, beam-item holding current token in loop, pre-populated with `pad_token_id`
        #TODO: Change this back to 1 instead of num_beam
        input_ids_padding = tf.ones((self.batch_size, 1, max_length - cur_len), dtype=tf.int32) * (
            pad_token_id or 0
        )
        running_sequences = tf.concat([input_ids, input_ids_padding], axis=-1)
        sequences = tf.ones((self.batch_size, self.num_beams, max_length), dtype=tf.int32) * (pad_token_id or 0)
        # per batch,beam-item state bit indicating if sentence has finished.
        is_sent_finished = tf.zeros((self.batch_size, self.num_beams), dtype=tf.bool)
        
        # per batch, beam-item score, logprobs
        running_scores = tf.tile(
            tf.expand_dims(tf.convert_to_tensor([0.0] + [-1.0e9] * (self.num_beams - 1)), axis=0), [self.batch_size, 1]
        )
        scores = tf.ones((self.batch_size, self.num_beams)) * -1.0e9
        
        # per batch beam indices
        running_beam_indices = tf.ones((self.batch_size, self.num_beams, max_length), dtype=tf.int32) * -1
        beam_indices = tf.ones((self.batch_size, self.num_beams, max_length), dtype=tf.int32) * -1
        
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = flatten_beam_dim(model_kwargs["attention_mask"])
        
        def beam_search_cond_fn(
            cur_len,
            running_sequences, 
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            model_kwargs,
        ):
            
            # 1. is less than max length?
            not_max_length_yet = cur_len < max_length
            best_running_score = running_scores[:, :1] / (tf.cast(cur_len, dtype=tf.float32) ** length_penalty)
            
            worst_finished_score = tf.where(
                is_sent_finished, tf.math.reduce_min(scores, axis=1, keepdims=True), -1.0e9
            )
            improvement_still_possible = tf.math.reduce_any(best_running_score > worst_finished_score)

            # 3. is there still a beam that has not finished?
            still_open_beam = ~(tf.math.reduce_all(is_sent_finished) & (early_stopping is True))

            return not_max_length_yet & still_open_beam & improvement_still_possible
            
        def beam_search_body_fn(
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            model_kwargs,
        ):
            """
            Beam Search iterative update function -- each iteration adds a new token and updates the best sequences
            seen so far
            """
            
            # 1. Forward current tokens
            if model_kwargs.get("past_key_values") is None:
                input_ids = running_sequences[:, :, :cur_len]
            else:
                input_ids = tf.expand_dims(running_sequences[:, :, cur_len - 1], -1)
            #print("inside beam_search_body, input_ids: ", input_ids.shape)
            #model_inputs = {'input_ids': flatten_beam_dim(input_ids), 'attention_mask': model_kwargs['attention_mask']}
            #TODO: May need to change to this to enable k, v cache
            model_inputs = self.prepare_inputs_for_generation(
                flatten_beam_dim(input_ids), use_cache=True, **model_kwargs
            )
            
            start = time.time()
            model_outputs = self.infer(**model_inputs)
            end = time.time()
            self.dur += [end - start]
            if self.is_first_iteration:
                running_sequences = tf.tile(running_sequences, [1, self.num_beams, 1])
                model_kwargs['attention_mask'] = tf.tile(model_kwargs['attention_mask'], [self.num_beams, 1])
                logits = unflatten_beam_dim(tf.tile(model_outputs['logits'][:, -1], [self.num_beams, 1]), self.num_beams)
                model_outputs["past_key_values"] = tf.tile(model_outputs["past_key_values"], [1, 1, self.num_beams, 1, 1, 1])
            else:
                logits = unflatten_beam_dim(model_outputs['logits'][:, -1], self.num_beams)
            log_probs = tf.nn.log_softmax(logits)
            
            log_probs_processed = log_probs
            log_probs = log_probs + tf.expand_dims(running_scores, axis=2)
            vocab_size = log_probs.shape[2]
            log_probs = tf.reshape(log_probs, (self.batch_size, self.num_beams * vocab_size))
            
            # 3. Retrieve top-K
            beams_to_keep = 2 * self.num_beams
            topk_log_probs, topk_indices = tf.math.top_k(log_probs, k=beams_to_keep)
            
            topk_current_beam_indices = topk_indices // vocab_size
            
            topk_running_beam_indices = self._gather_beams(running_beam_indices, topk_current_beam_indices)
            topk_running_sequences = self._gather_beams(running_sequences, topk_current_beam_indices)
            topk_ids = topk_indices % vocab_size
            
            
            indices_batch = tf.repeat(tf.range(self.batch_size), [beams_to_keep])
            indices_beam = tf.tile(tf.range(beams_to_keep), [self.batch_size])
            update_indices = tf.stack(
                [indices_batch, indices_beam, tf.broadcast_to(cur_len, [self.batch_size * beams_to_keep])], axis=-1
            )
            topk_sequences = tf.tensor_scatter_nd_update(
                tensor=topk_running_sequences,
                indices=update_indices,
                updates=tf.reshape(topk_ids, [self.batch_size * beams_to_keep]),
            )
            
            
            # we want to store the beam indices with batch information -> real beam index = beam index % num beams
            batch_modified_indices = topk_current_beam_indices + tf.broadcast_to(
                tf.expand_dims(tf.range(self.batch_size) * self.num_beams, axis=1), topk_current_beam_indices.shape
            )
            topk_beam_indices = tf.tensor_scatter_nd_update(
                tensor=topk_running_beam_indices,
                indices=update_indices,
                updates=tf.reshape(batch_modified_indices, [self.batch_size * beams_to_keep]),
            )
            
            # 4. Check which sequences have ended
            # Update current sequences: Did the top `num_beams` sequences reach an end marker?
            # To prevent these just finished sequences from being added to the current sequences
            # set of active beam search sequences, set their log probs to a very large negative value.
            eos_token_id = [50256]
            eos_in_next_token = tf.math.reduce_any(
                    tf.equal(
                        tf.broadcast_to(
                            topk_sequences[:, :, cur_len], [len(eos_token_id)] + topk_sequences[:, :, cur_len].shape
                        ),
                        tf.expand_dims(tf.expand_dims(eos_token_id, -1), -1),
                    ),
                    axis=0,
                )
            did_topk_just_finished = eos_in_next_token & tf.broadcast_to(
                tf.concat((tf.ones((self.num_beams), dtype=tf.bool), tf.zeros((self.num_beams), dtype=tf.bool)), axis=0),
                shape_list(eos_in_next_token),
            )
            
            did_topk_just_finished = eos_in_next_token & tf.broadcast_to(
                tf.concat((tf.ones((self.num_beams), dtype=tf.bool), tf.zeros((self.num_beams), dtype=tf.bool)), axis=0),
                shape_list(eos_in_next_token),
            )

            # non-top `num_beams` eos tokens can't be used to finish a beam, but the others can't be used in the next
            # running sentences either
            running_topk_log_probs = topk_log_probs + tf.cast(eos_in_next_token, tf.float32) * -1.0e9

            # 5. Get running sequences scores for next
            # Determine the top k beam indices (from top 2*k beams) from log probs and gather top k beams
            # (from top 2*k beams).
            next_topk_indices = tf.math.top_k(running_topk_log_probs, k=self.num_beams)[1]
            next_running_sequences, next_running_scores, next_running_beam_indices = self._gather_beams(
                [topk_sequences, running_topk_log_probs, topk_beam_indices], next_topk_indices
            )

            # 6. Process topk logits
            # Further process log probs:
            # - add length penalty
            # - make sure no scores can be added anymore if beam is full
            # - make sure still running sequences cannot be chosen as finalized beam
            topk_log_probs = topk_log_probs / (tf.cast(cur_len, dtype=tf.float32) ** length_penalty)
            beams_in_batch_are_full = tf.broadcast_to(
                tf.math.reduce_all(is_sent_finished, axis=-1, keepdims=True), shape_list(did_topk_just_finished)
            ) & (early_stopping is True)
            add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
            topk_log_probs += tf.cast(add_penalty, tf.float32) * -1.0e9

            # 7. Get scores, sequences, is sentence finished for next.
            # Combine sequences, scores, and flags along the beam dimension and compare new finished sequence scores
            # to existing finished scores and select the best from the new set of beams
            merged_sequences = tf.concat([sequences, topk_sequences], axis=1)
            merged_scores = tf.concat([scores, topk_log_probs], axis=1)
            merged_beams = tf.concat([beam_indices, topk_beam_indices], axis=1)
            merged_is_sent_finished = tf.concat([is_sent_finished, did_topk_just_finished], axis=1)
            topk_merged_indices = tf.math.top_k(merged_scores, k=self.num_beams)[1]
            next_sequences, next_scores, next_beam_indices, next_is_sent_finished = self._gather_beams(
                [merged_sequences, merged_scores, merged_beams, merged_is_sent_finished], topk_merged_indices
            )

            # 8. Prepare data for the next iteration
            # Determine the top k beam indices from the original set of all beams. With these, gather the top k
            # beam-associated caches.
            cur_len = cur_len + 1
            #TODO: May need to un-comment this to enable k, v cache, batch_axis=cache_batch_axis, cache_batch_axis is 0 for gpt-j
            #print(model_outputs["past_key_values"].shape)
            if "past_key_values" in model_outputs:
                cache = tf.nest.map_structure(
                    lambda tensor: unflatten_beam_dim(tensor, self.num_beams, batch_axis=2),
                    model_outputs["past_key_values"],
                )
                next_running_indices = self._gather_beams(topk_current_beam_indices, next_topk_indices)
                next_cache = self._gather_beams(cache, next_running_indices, batch_axis=2)
                model_outputs["past_key_values"] = tf.nest.map_structure(
                    lambda tensor: flatten_beam_dim(tensor, batch_axis=2), next_cache
                )
            
            #update model_kwargs for generation
            #attention_mask = model_kwargs["attention_mask"]
            #model_kwargs["attention_mask"] = tf.concat(
                    #[attention_mask, tf.ones((attention_mask.shape[0], 1), dtype=tf.int32)], axis=-1
               # )
            #TODO: Use the below instead to enable k, v cache
            
            next_model_kwargs = self._update_model_kwargs_for_generation(
                    model_outputs, model_kwargs, is_encoder_decoder=False
                )
            
            return (
                cur_len,
                next_running_sequences,
                next_running_scores,
                next_running_beam_indices,
                next_sequences,
                next_scores,
                next_beam_indices,
                next_is_sent_finished,
                next_model_kwargs,
            )
        
        if isinstance(model, str):
            loaded_model = tf.saved_model.load(model)
        else:
            loaded_model = model
        self.infer = loaded_model.signatures["serving_first_iteration"]
        
        (
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            model_kwargs,
        ) = beam_search_body_fn(
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            model_kwargs,
        )
        
        
        # 2-to-n generation steps can then be run in autoregressive fashion (only in case 1st generation step does
        # NOT yield EOS token though)
        self.is_first_iteration = 0
        self.infer = loaded_model.signatures["serving_default"]
        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        maximum_iterations = self.token_to_generate - 1
        (
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            _,
        ) = tf.while_loop(
            beam_search_cond_fn,
            beam_search_body_fn,
            (
                cur_len,
                running_sequences,
                running_scores,
                running_beam_indices,
                sequences,
                scores,
                beam_indices,
                is_sent_finished,
                model_kwargs,
            ),
            maximum_iterations=maximum_iterations,
        )
        
        
        # 6. prepare outputs
        # Account for the edge-case where there are no finished sequences for a particular batch item. If so, return
        # running sequences for that batch item.
        none_finished = tf.math.reduce_any(is_sent_finished, axis=1)
        sequences = tf.where(none_finished[:, None, None], sequences, running_sequences)
        beam_indices = tf.where(none_finished[:, None, None], beam_indices, running_beam_indices)

        # Apply the length penalty so that running scores match the finalized scores if they are used
        running_scores = running_scores / (tf.cast(cur_len, dtype=tf.float32) ** length_penalty)
        scores = tf.where(none_finished[:, None], scores, running_scores)

        # Take best beams for each batch (the score is sorted in descending order)
        sequences = flatten_beam_dim(sequences[:, :num_return_sequences, :])
        scores = flatten_beam_dim(scores[:, :num_return_sequences])
        beam_indices = flatten_beam_dim(beam_indices[:, :num_return_sequences, :])
        
        
        # Cut for backward compatibility
        sequences = sequences[:, :cur_len]
        beam_indices = beam_indices[:, :cur_len]
        
        return sequences

    
    def evaluate(self, model, model_name_or_path="EleutherAI/gpt-j-6B"):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        for idx, single_data in enumerate(self.data):
            tic = time.time()
            print("========Original sentence========")
            decoded = tokenizer.batch_decode(single_data, skip_special_tokens=True)
            print(decoded)
            #output = self.beam_search(mydata[i*self.batch_size: (i+1)*self.batch_size], model)
            output = self.generate(single_data, model)
            toc = time.time()
            decoded = tokenizer.batch_decode(output[0], skip_special_tokens=True)
            print("=========Decoded sentence====")
            print(decoded)
            if idx >= 5: break

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, run_args = parser.parse_args_into_dataclasses()

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if run_args.seed is not None:
        set_seed(run_args.seed)

    raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.checkpoint,
            use_auth_token=None,
        )
        

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.checkpoint,
                use_auth_token=None,
            )
            
        
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    column_names = raw_datasets["test"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    mydata = tokenizer(raw_datasets["test"][text_column_name], return_tensors="np").input_ids
    gpt_j_model_path = "/home/dataset_broad/dataset/users/zehaohua/gpt-j-6B-2-signatures-first-second-iter"
    infer = Inference(data=mydata, input_tokens=len(mydata[0]) - 1)
    infer.evaluate(gpt_j_model_path)