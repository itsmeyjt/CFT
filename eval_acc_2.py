from transformers import AutoModelForCausalLM,AutoTokenizer
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
        device_map="cuda:0",
    )


    model.eval()
    f = open('YOUR_INFO_PATH', 'r')
    items = f.readlines()
    item_names = [_.split('\t')[0].strip("\"\n").strip(" ") for _ in items]
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
        def batch(list, batch_size=1):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        text = [[_["predict"][k].strip().strip("\"").strip() for _ in test_data] for k in range(5)]
        tokenizer.padding_side = "left"
        from tqdm import tqdm
        game_embedding = torch.load(embed_path)
        for k in range(5):
            predict_embeddings = []
            for i, batch_input in tqdm(enumerate(batch(text[k], 32))):
                input = tokenizer(batch_input, return_tensors="pt", padding=True)
                input_ids = input.input_ids.long().to("cuda:0")
                attention_mask = input.attention_mask.long().to("cuda:0")
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                # print(hidden_states[-1].size())
                predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
            
            predict_embeddings = torch.cat(predict_embeddings, dim=0)     
            dist = torch.cdist(predict_embeddings, game_embedding, p=2)

            rank = dist
            rank = rank.argsort(dim = -1)
            most,next = rank[:,0].unsqueeze(-1),rank[:,1].unsqueeze(-1)
            if k == 0:
                res0 = most
                res1 = next
            else:
                res0 = torch.cat([res0,most],dim=-1)
                res1 = torch.cat([res1,next],dim=-1)
                # .argsort(dim = -1)
        topk_list = [1, 3, 5, 10]
        NDCG = []
        HR = []
        for topk in topk_list:
            S = 0
            SS = 0
            LL = len(test_data)
            for i in range(len(test_data)):
                target_item = test_data[i]['output'].strip("\"").strip(" ")
                rec_list = res0[i].tolist() + res1[i].tolist()
                rec_list = rec_list[:topk]
                for k in range(len(rec_list)):
                    for target_item_id in item_dict[target_item]:
                        if rec_list[k] == target_item_id:
                            S = S + (1 / math.log(k + 2))
                            SS = SS + 1
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