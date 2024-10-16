from transformers import GenerationConfig, AutoModelForCausalLM,AutoTokenizer
import transformers
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import fire
import math
import json
import argparse

from peft import PeftModel
def calc(path,result_path,base_model,embed_path):
    path = [path]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


    model.eval()
    


    f = open('YOUR_INFO_PATH', 'r')
    items = f.readlines()
    item_names = [_.split('\t')[0].strip(" ").strip("\"") for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    import pandas as pd


    result_dict = dict()
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.eval()
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        text = [_["predict"][0].strip().strip("\"").strip()  for _ in test_data]
        tokenizer.padding_side = "left"

        def batch(list, batch_size=1):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        predict_embeddings = []
        from tqdm import tqdm
        for i, batch_input in tqdm(enumerate(batch(text, 32))):
            input = tokenizer(batch_input, return_tensors="pt", padding=True)
            input_ids = input.input_ids.to("cuda:0")       
            attention_mask = input.attention_mask.to("cuda:0")     
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
        
        predict_embeddings = torch.cat(predict_embeddings, dim=0)     
        game_embedding = torch.load(embed_path)
        dist = torch.cdist(predict_embeddings, game_embedding, p=2)

            
        rank = dist
        rank = rank.argsort(dim = -1).argsort(dim = -1)
        topk_list = [1, 3, 5, 10, 20, 50]
        NDCG = []
        HR = []
        for topk in topk_list:
            S = 0
            SS = 0
            LL = len(test_data)
            for i in range(len(test_data)):
                target_item = test_data[i]['output'].strip("\"").strip(" ")
                minID = 1000000
                for _ in item_dict[target_item]:
                    if rank[i][_].item() < minID:
                        minID = rank[i][_].item()
                if minID < topk:
                    S= S+ (1 / math.log(minID + 2))
                    SS = SS + 1
            temp_NDCG = []
            temp_HR = []
            NDCG.append(S / LL / (1.0 / math.log(2)))
            HR.append(SS / LL) 

        print(NDCG)
        print(HR)
        print('_' * 100)
        result_dict[p]["NDCG"] = NDCG
        result_dict[p]["HR"] = HR
    f = open(result_path, 'w')    
    json.dump(result_dict, f, indent=4)

if __name__=='__main__':
    fire.Fire(calc)


