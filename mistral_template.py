#Encode and Decode with mistral_common
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
 
mistral_models_path = "MISTRAL_MODELS_PATH"
 
tokenizer = MistralTokenizer.v1()
 
completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
 
tokens = tokenizer.encode_chat_completion(completion_request).tokens

#Inference with mistral_inference
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

model = Transformer.from_folder(mistral_models_path)
out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)

result = tokenizer.decode(out_tokens[0])

print(result)

#Inference with hugging face transformers
from transformers import AutoModelForCausalLM
 
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.to("cuda")
 
generated_ids = model.generate(tokens, max_new_tokens=1000, do_sample=True)

# decode with mistral tokenizer
result = tokenizer.decode(generated_ids[0].tolist())
print(result)

# Instruction format
# In order to leverage instruction fine-tuning, your prompt should be surrounded by [INST] and [/INST] tokens. 
# The very first instruction should begin with a begin of sentence id. The next instructions should not.
# The assistant generation will be ended by the end-of-sentence token id.
# text = "<s>[INST] What is your favourite condiment? [/INST]"
"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
"[INST] Do you have mayonnaise recipes? [/INST]"

# This format is available as a chat template via the apply_chat_template() method:
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
