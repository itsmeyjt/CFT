import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer
import transformers
import torch
import fire
def embed(base_model,embed_path,info_path = "YOUR_INFO_PATH"):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.eval()
    f = open(info_path, 'r')
    # the format of the item name file is 
    # item_name item_id
    # A 0
    # B 1
    lines = f.readlines()
    f.close()
    text = [_.split('\t')[0].strip(" ").strip('\"') for _ in lines] # remove the leading and trailing spaces and quotess make sure this preprocess is the same as the prediction
    tokenizer.padding_side = "left"
    from tqdm import tqdm
    device='cuda:0'
    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    item_embedding = []
    for i, batch_input in tqdm(enumerate(batch(text,32))):
        input = tokenizer(batch_input, return_tensors="pt", padding=True)
        input_ids = input.input_ids.to(device)
        attention_mask = input.attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        item_embedding.append(hidden_states[-1][:, -1, :].detach().cpu())
        # break
    item_embedding = torch.cat(item_embedding, dim=0)
    torch.save(item_embedding, embed_path)


if __name__ == "__main__":
    fire.Fire(embed)