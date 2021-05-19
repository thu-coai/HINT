from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import os
import sys
import numpy as np
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        TopKLogitsWarper,
        TemperatureLogitsWarper,
        BeamSearchScorer,
    )

def gather_nd(x, indices):
    newshape = list(indices.shape[:-1] + x.shape[indices.shape[-1]:]) + [1]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([torch.tensor([x.__getitem__(tuple(i))]) for i in indices]).reshape(tuple(newshape))
    return out

def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.size()
    sorted_logits, _ = torch.sort(logits, descending=True, axis=-1)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits), axis=-1)
    cumulative_position = torch.sum((cumulative_probs <= p).to(torch.int32), axis=-1) - 1
    indices = torch.stack([
        torch.arange(0, batch).to(device),
        # number of indices to include
        torch.max(cumulative_position, torch.zeros([batch], dtype=cumulative_position.dtype).to(device)),
    ], axis=-1)
    min_values = gather_nd(sorted_logits, indices).to(device)
    return torch.where(
        logits < min_values,
        torch.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(input_ids, model, max_length, vocab_size, sen_token_id, dis_token_id, temperature=0.7, top_p=0.9, no_sample=False):
    batch_size = input_ids.size()[0]
    decoder_input_ids = torch.tensor([2 for _ in range(batch_size)])[:, None].to(device)
    # tokens_embed = model.transformer.get_input_embeddings()
    for i in range(max_length):
        is_sen = torch.eq(decoder_input_ids[:, -1], tokenizer.mask_token_id)[:, None]
        tmpweight = [0 for _ in range(vocab_size)]
        tmpweight[dis_token_id] = 1e20
        # [batch_size, vocab_size]
        weight = is_sen * torch.tensor(tmpweight)[None, :].to(device)

        logits = model(input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)["logits"]
        logits = logits[:, -1, :] / temperature

        logits += weight

        if no_sample:
            prev = torch.topk(probs, 1)[:, 1]
        else:
            logits = top_p_logits(logits, p=top_p)
            probs = torch.nn.functional.softmax(logits)
            prev = torch.multinomial(probs, 1)
        decoder_input_ids = torch.cat([decoder_input_ids, prev], 1)
    return decoder_input_ids

device = sys.argv[1]
model_name_path = sys.argv[2]
data_dir = sys.argv[3]
with open("%s/test.source"%data_dir, "r") as fin:
    ipt = [line.strip() for line in fin]
with open("%s/test.target"%data_dir, "r") as fin:
    opt = [line.strip() for line in fin]

def pro(token_list, tokenizer):
    for i, t in enumerate(token_list):
        if t not in [0, 2]:
            break
    token_list = token_list[i:]
    string = tokenizer.decode(token_list, skip_special_tokens=False)
    string = string.replace("<mask><s>", " ")
    string = string[:string.find("</s>")].strip()
    return string

tokenizer = BartTokenizer.from_pretrained(model_name_path)

# model = BartModel.from_pretrained('./bart-base', return_dict=True)
model = BartForConditionalGeneration.from_pretrained(model_name_path, return_dict=True).to(device)
model.eval()

file_out_dir = "./result"
if not os.path.exists(file_out_dir):
    os.mkdir(file_out_dir)
file_out = "%s/generation.txt"%(file_out_dir)
print("write to %s"%file_out)
with open(file_out, "w") as fout:
    batch_size = 32
    st, ed = 0, 0
    all_loss = []
    with torch.no_grad():
        while ed < len(ipt):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
            input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
            gen = sample_sequence(input_ids=input_ids, model=model, max_length=500, vocab_size=len(tokenizer), sen_token_id=tokenizer.mask_token_id, dis_token_id=tokenizer.bos_token_id, temperature=0.7, top_p=0.9)
            for ip, op in zip(input_ids, gen):
                fout.write(pro(ip, tokenizer) + "|||" + pro(op, tokenizer)+"\n")