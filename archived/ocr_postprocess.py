

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from transformers import TextStreamer
import os
import torch

model_id = 'llm'

tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Loading Model... \n\n")
model = AutoModelForCausalLM.from_pretrained(
    model_id
)

def extract_info_from_ocr(ocr_text):

    inst = f"""Extract the medication name from this OCR text of a medicine package:
    {ocr_text} 

    Return only the generic medication name, manufacturer/laboratory, dosage, and packaging quantity of the OCR text, don't add addresses or any labeling. Return them seperated by |."""

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": inst},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)

    # Tokenize the sample
    inputs = tokenizer([input_text], return_tensors='pt')

    # Call generate on the inputs
    out = model.generate(
        **inputs,
        max_new_tokens=96,
        streamer=TextStreamer(tokenizer=tokenizer, skip_special_tokens=True),
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

    extracted_query = tokenizer.batch_decode(out)[0]
    #extracted_query = extracted_query[extracted_query.index('<|im_end|>\n<|im_start|>system\n')+len('<|im_end|>\n<|im_start|>system\n'):]
    #extracted_query = extracted_query.replace('<|im_end|>', '')
    extracted_query = extracted_query[len(input_text):]
    extracted_query = extracted_query.replace("<|im_start|>system", "").replace("<|im_end|>", "")
    return extracted_query.strip()


