import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES']="0"
import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json

from transformers import GenerationConfig
from transformers import AutoModelForCausalLM,AutoTokenizer


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    base_model: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 32,
    length_penalty:float = 0.0
):

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # # unwind broken decapoda-research config
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.
    model.config.use_cache=False
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=0.9,
        top_k=40,
        num_beams=5,
        length_penalty=length_penalty,
        max_new_tokens=128,
        **kwargs,
    ):
        # prompt = [generate_prompt(instruction,input) for instruction, input in zip(instructions, inputs)]
        prompt = [generate_prompt(instruction,input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty = length_penalty,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                min_new_tokens = 4,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs


    outputs = []
    from tqdm import tqdm
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        # instructions = [_['instruction'] for _ in test_data][:32]
        # inputs = [_['input'] for _ in test_data][:32]
        def batch(list, batch_size=batch_size):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions, inputs = batch
            output = evaluate(instructions, inputs)
            outputs = outputs + output
            
        for i, test in tqdm(enumerate(test_data)):
            # if i == len(instructions):
            #     break
            test_data[i]['predict'] = outputs[i]


    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)