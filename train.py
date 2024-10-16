import os
os.environ['LD_LIBRARY_PATH'] = 'YOUR_CONDA_ENV/lib'

import sys
from typing import List

import numpy as np 
import fire
import torch
torch.set_printoptions(profile="full")
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import transformers
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset 
from transformers import EarlyStoppingCallback,GenerationConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.trainer import *
from transformers import Trainer



# from transformers import AutoModel, AutoTokenizer
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM,AutoTokenizer  # noqa: F402

class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):                                
        x,_ = inputs["input_ids"].shape
        inputs_0,inputs_1=dict(),dict()
        values = list(inputs["input_ids"])
        maxlen = 0
        for i in range(x):
            id = torch.nonzero(torch.eq(values[i], self.model.config.eos_token_id))[-2] + 1
            l1= len(values[i][:id])
            maxlen = max(maxlen,l1)
            
        for i in range(x):
            id = torch.nonzero(torch.eq(values[i], self.model.config.eos_token_id))[-2] + 1
            l1,l2 = len(values[i][:id]),len(values[i][id:])
            for k,v in inputs.items():
                if k == "input_ids":
                    pad = self.model.config.eos_token_id
                elif k == "labels":
                    pad = -100
                else:
                    pad = 0
                prompt1 = torch.cat([torch.tensor([pad]*(maxlen-l1),dtype=torch.int64,device=v[i].device),v[i][:id]]).unsqueeze(0)
                prompt2 = torch.cat([torch.tensor([pad]*(maxlen-l2),dtype=torch.int64,device=v[i].device),v[i][id:]]).unsqueeze(0)
                inputs_0[k] = torch.cat([inputs_0.pop(k),prompt1],dim=0) if k in inputs_0.keys() else prompt1
                inputs_1[k] = torch.cat([inputs_1.pop(k),prompt2],dim=0) if k in inputs_1.keys() else prompt2
        
        # get logits
        # print("input:",inputs_1["input_ids"][0])
        labels = inputs_0.pop('labels')
        inputs_1.pop("labels")
        outputs_0 = model(**inputs_0)
        outputs_1 = model(**inputs_1)
        
        
        if self.args.past_index >= 0:
            self._past = outputs_0[self.args.past_index]
       
        logits_0 = outputs_0.get("logits")
        logits_1 = outputs_1.get("logits")
        
        shift_logits_0 = logits_0[:,:-1,:].contiguous()
        shift_logits_1 = logits_1[:,:-1,:].contiguous()
        shift_labels = labels[:,1:].contiguous()
        shift_logits_0 = shift_logits_0.view(-1, self.model.config.vocab_size).to(shift_labels.device)
        shift_logits_1 = shift_logits_1.view(-1, self.model.config.vocab_size).to(shift_labels.device)
        shift_logits = shift_logits_0 - shift_logits_1
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            beta = 0.25
            _flag = shift_labels > -100
            flag = _flag.float().to(shift_labels.device)
            flag_sum = torch.sum(flag,dim=-1).reshape(-1,1)
            _gap = (1-beta) / (flag_sum-1)
            _pos = torch.cumsum(flag,dim=-1)
            weight = 1 - (_pos-1) * _gap
            weight = torch.where(_flag,weight,torch.zeros_like(weight))
            weight = weight.view(-1)
        shift_labels = shift_labels.view(-1)
        loss = loss_fct(shift_logits_0, shift_labels)
        loss_diff = loss_fct(shift_logits, shift_labels)
        tot_weight = weight.sum()
        loss = (loss * weight).sum()/tot_weight
        loss_diff= (loss_diff * weight).sum()/tot_weight
        # loss_fct = torch.nn.CrossEntropyLoss()
        # shift_labels = shift_labels.view(-1)
        # loss = loss_fct(shift_logits_0, shift_labels)
        # loss_diff = loss_fct(shift_logits, shift_labels)
        if model.training:
            loss += 0.1 * loss_diff

 
        return (loss,outputs_0) if return_outputs else loss
     

        
        
def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: List[str] = [""],
    val_data_path: List[str] = [""],
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_DISABLED"] = "true"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    # model.set_tau(tau)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        data_point2 = {**data_point,"input":None}
        
        full_prompt1 = generate_prompt(data_point)
        tokenized_full_prompt1 = tokenize(full_prompt1)
        
        full_prompt2 = generate_prompt(data_point2)        
        tokenized_full_prompt2 = tokenize(full_prompt2)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt1["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt1["labels"][
                user_prompt_len:
            ]  
            
            user_prompt = generate_prompt({**data_point2, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt2["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt2["labels"][
                user_prompt_len:
            ]  
        tokenized_full_prompt = dict()
        for k,v1,v2 in zip(tokenized_full_prompt1.keys(),tokenized_full_prompt1.values(),tokenized_full_prompt2.values()):
            tokenized_full_prompt[k] = v1 + v2
        return tokenized_full_prompt

    train_data_list = []
    val_data_list = []

    for path in train_data_path:
        if path.endswith(".json"):
            train_data_list.append(load_dataset("json", data_files=path))
        else:
            train_data_list.append(load_dataset(path))

    for path in val_data_path:
        if path.endswith(".json"):
            val_data_list.append(load_dataset("json", data_files=path))
        else:
            val_data_list.append(load_dataset(path))

    for i in range(len(train_data_list)):
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    for i in range(len(val_data_list)):
        val_data_list[i] = val_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    val_data = concatenate_datasets([_["train"] for _ in val_data_list])




    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        if not os.path.exists(checkpoint_name):
            print(f"Checkpoint {checkpoint_name} not found")

    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    
    trainer = CustomTrainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        # train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=8,
            bf16=True,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point,is_his=True):

    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    {data_point["output"]}"""     


if __name__ == "__main__":
    fire.Fire(train)
