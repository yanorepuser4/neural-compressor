from transformers import AutoTokenizer, TFAutoModelForCausalLM
import tensorflow as tf
 
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = TFAutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
 
 
@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="attention_mask"),
])
 
def serve_first_iteration(input_ids, attention_mask):
    outputs =  model(input_ids, attention_mask=attention_mask)
    p_k_v = outputs['past_key_values']#tupe of tuple, each sub-tupe is (k, v) of i-th layer
   
    temp = [tf.stack([k, v], axis=0) for k, v in p_k_v]
    temp = tf.stack(temp, axis=0)
   
    return {'logits': outputs['logits'], 'past_key_values': temp}
 
@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="attention_mask"),
    tf.TensorSpec(shape=(28, 2, None, None, None, None), dtype=tf.float32, name="past_key_values"),#n_layer, 2, bs, num_head, seq_len, hidden_state
])
def serve(input_ids, attention_mask, past_key_values):
    past_key_values = tf.split(past_key_values, num_or_size_splits=28, axis=0)
    past_key_values = [tf.squeeze(u, 0) for u in past_key_values]
    past_key_values = [(k_v[0], k_v[1]) for k_v in past_key_values]
 
    outputs =  model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
    p_k_v = outputs['past_key_values']#tupe of tuple, each sub-tupe is (k, v) of i-th layer
   
    temp = [tf.stack([k, v], axis=0) for k, v in p_k_v]
    temp = tf.stack(temp, axis=0)
   
    return {'logits': outputs['logits'], 'past_key_values': temp}
signatures = {
    "serving_default": serve,
    "serving_first_iteration": serve_first_iteration
}
model.save_pretrained('./gpt-j-6B-2-signatures-first-second-iter', saved_model=True, signatures=signatures)
