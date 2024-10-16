from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,AutoTokenizer
import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

import math
import json

import argparse
parse = argparse.ArgumentParser()

args = parse.parse_args()

import pandas as pd
# data = pd.read_csv("YOUR_TEST_CSV_PATH")
# index = data.index

path = ["YOUR_RESULT_PATH"]
# path.append(os.path.join('/data/test/result', "ml-1m_result.json"))
print(path)

# 读取模型
base_model = "YOUR_MODEL_PATH"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

# f = open('/data/ml-1m/ratings.dat', 'r')
# data = f.readlines()
# f.close()


f = open('YOUR_INFO_PATH', 'r')
item2id = dict()
with open("YOUR_INFO_PATH", "r") as f:
    lines = f.readlines()
    l = len(lines)
    for line in lines:
        line = line.split("\t")
        if line[0] not in item2id:
            item2id[line[0]] = [int(line[1])]
        else:
            item2id[line[0]].append(int(line[1]))
# movies = f.readlines()
# movie_names = [_.split('\t')[0].strip("\"") for _ in movies]
# movie_ids = [int(_.split('\t')[1]) for _ in movies]
# movie_dict = dict(zip(movie_names, movie_ids))
# id_mapping = dict(zip(movie_ids, range(len(movie_ids))))
# f.close()
id_mapping = dict()
for i,value in enumerate(item2id.values()):
    for v in value:
        id_mapping[v] = i


movie_genre = pd.read_csv("books/books_pop_count_5.csv")
genre_set = ["0", "1", "2", "3", "4"]

# 统计test set中的不同类别的比例
test_data = pd.read_csv("YOUR_TEST_CSV_PATH")
history_list = test_data["history_item_id"].to_list()
history_list = [eval(_) for _ in history_list]
# history_list = test_data["item_id"].to_list()
history_count = {_:0 for _ in genre_set}

for l in history_list:
    for id in l:
        index = id_mapping[id]
        pop_genre = movie_genre["Pop"].iloc[index]

        history_count["%d" % pop_genre] += 1

# for id in history_list:
#         index = id_mapping[id]
        
#         pop_genre = movie_genre["Pop"].iloc[index]

#         history_count["%d" % pop_genre] += 1
        
        

result_dict = dict()
for p in path:

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.eval()

    f = open(p, 'r')
    import json
    test_data = json.load(f)
    f.close()

    text = [_["predict"][0].strip().strip("\"").strip() for _ in test_data]
    
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
    # if p.find("des") == -1:
    movie_embedding = torch.load("item_embedding.pt")
    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)


    for gamma in [0]:

        # topk_list = [1, 3, 5, 10, 20, 50]

        # rank = torch.pow((1 + pop_rank), -gamma) * dist
        # # print(rank)
        rank = dist.argsort(dim = -1) # .argsort(dim = -1)

        topk_list = [1, 3, 5, 10, 20, 50]
        topk_count = {k: {_:0 for _ in genre_set} for k in topk_list}

        for topk in topk_list:
            for i in range(len(test_data)):
                for k in range(topk):
                    # for col in movie.index:
                    #     topk_count[topk][col] += int(movie[col])
                    pop_genre = movie_genre["Pop"].iloc[id_mapping[rank[i][k].item()]]
                    topk_count[topk]["%d" % pop_genre] += 1



        f = open('books/eval_all_pop.json', 'w') 
        history_count = {key: int(value) for key, value in history_count.items()}
        for tpok in topk_list:
            topk_count[topk] = {key: int(value) for key, value in topk_count[topk].items()}
        json.dump(history_count, f, indent=4)

        json.dump(topk_count, f, indent=4)
        f.close()
