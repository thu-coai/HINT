from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import sys
import os
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

device = sys.argv[1]
model_name_path = sys.argv[2]
data_dir = sys.argv[3]
with open("%s/test.source"%data_dir, "r") as fin:
    ipt = [line.strip() for line in fin]
with open("%s/test.target"%data_dir, "r") as fin:
    opt = [line.strip() for line in fin]

batch_size = 10

print("loading model %s"%model_name_path)
tokenizer = BartTokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id
mask_token_id = tokenizer.mask_token_id

model = BartForConditionalGeneration.from_pretrained(model_name_path, return_dict=True).to(device)
model.eval()
st, ed = 0, 0
all_loss = []
while ed < len(ipt):
    st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
    input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
    with torch.no_grad():
        src_ids = input_ids
        tgt_ids = tokenizer(opt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
        decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
        outputs = model(src_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs["logits"]

        tmp_batch_size = lm_logits.size()[0]
        pad_pos = torch.eq(tgt_ids, pad_token_id).to(torch.float)
        sen_pos = torch.eq(tgt_ids, mask_token_id).to(torch.float)
        dis_pos = torch.cat([torch.zeros([tmp_batch_size, 1]).to(sen_pos.device), sen_pos[:, :-1]], 1)
        loss_mask = 1 - (pad_pos + sen_pos + dis_pos)
        ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        # assert lm_logits.shape[-1] == self.vocab_size
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        loss = torch.sum(loss * loss_mask.view(-1)) / (torch.sum(loss_mask) + 1e-20)
        all_loss.append(loss.cpu().numpy())

print("perplexity:", np.exp(np.mean(all_loss)))
