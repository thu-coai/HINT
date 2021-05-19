import sys
from tokenizer import SimpleTokenizer, PretrainedTokenizer
import nltk
import numpy as np
from nltk import ngrams
import os
tokenizer = SimpleTokenizer(method="nltk")

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

def learnable_metric(name, ipts, cands, model_path, type_, device):
    roberta_tokenizer = RobertaTokenizer.from_pretrained(model_path)
    roberta_model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
    all_pred = []
    batch_size = 32
    tgt_dir = "./%s_result/%s"%(type_, name.split("/")[0])
    if ("/" in name) and (not os.path.exists(tgt_dir)):
        os.mkdir(tgt_dir)    
    fout = open("./%s_result/%s.txt"%(type_, name), "w")
    with torch.no_grad():
        st, ed = 0, 0
        while ed < len(cands):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(cands)) else len(cands)
            tmp_ipts, tmp_cands = ipts[st:ed], cands[st:ed]

            new_ = []
            for i, c in zip(tmp_ipts, tmp_cands):
                tmp_sen = ("%s %s"%(i.strip(), c.strip())).strip()
                tmp_tokens = roberta_tokenizer([tmp_sen], return_tensors="pt", padding=True)
                if tmp_tokens["input_ids"].size()[1] < 512:
                    new_.append(tmp_sen)
            if not len(new_):
                continue

            inputs = roberta_tokenizer(["%s %s"%(i.strip(), c.strip()) for i, c in zip(tmp_ipts, tmp_cands)], return_tensors="pt", padding=True).to(device)
            outputs = roberta_model(**inputs)
            logits = outputs.logits
            prob = torch.nn.functional.softmax(logits).cpu().numpy()
            all_pred += prob[:, 1].tolist()
            for i, c, p in zip(tmp_ipts, tmp_cands, prob):
                fout.write("%.4f|||%s|||%s\n"%(p[1], i, c))
    return {type_: np.mean(all_pred)}

def sent_tokenize(s):
    sentokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle') 
    slist = sentokenizer.tokenize(s.strip())
    newslist = copy.deepcopy(slist)
    return [" ".join(s.strip().split()) for s in newslist if s.strip() != ""]

def bleu(refs, cands):
    result = {}
    for i in range(1, 5):
        result["bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu([[r] for r in refs], cands, weights=tuple([1./i for j in range(i)])))
    return result

def repetition_distinct(name, cands):
    result = {}
    tgt_dir = "./lex_rept/%s"%(name.split("/")[0])
    if ("/" in name) and (not os.path.exists(tgt_dir)):
        os.mkdir(tgt_dir)
    fout = open("./lex_rept/%s.txt"%name, "w")
    for i in range(1, 5):
        num, all_ngram, all_ngram_num = 0, {}, 0.
        for k, cand in enumerate(cands):
            ngs = ["_".join(c) for c in ngrams(cand, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
            for s in set(ngs):
                if ngs.count(s) > 1: # 4 for wp
                    if i == 4: fout.write("%d|||%s|||%s\n"%(k, s, " ".join(cand)))
                    num += 1
                    break
        result["repetition-%d"%i] = "%.4f"%(num / float(len(cands)))
        result["distinct-%d"%i] = "%.4f"%(len(all_ngram) / float(all_ngram_num))
    fout.close()
    return result

def length(cands, name):
    length = []
    tgt_dir = "./length/%s"%(name.split("/")[0])
    if ("/" in name) and (not os.path.exists(tgt_dir)):
        os.mkdir(tgt_dir)
    with open("length/%s.txt"%name, "w") as fout:
        for c in cands:
            fout.write("%d|||%s\n"%(len(c), str(c)))
            length.append(len(c))
    return {"length": "%.4f"%np.mean(length)}

def union(name):
    with open("./UNION/model/output/%s.txt"%name, "r") as fin:
        union_score = np.mean([float(line.strip()) for line in fin])
    return {"union": union_score}

import scipy
from transformers import AutoTokenizer, AutoModel
def sent_semantic_repetition(name, ipts, cands, device, model_type):
    sbert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    sbert_model = AutoModel.from_pretrained(model_type).to(device)
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    ss_max, ss_min, ss_mean = [], [], []
    tgt_dir = "./sent_reptscore/%s"%(name.split("/")[0])
    if ("/" in name) and (not os.path.exists(tgt_dir)):
        os.mkdir(tgt_dir)
    fout = open("./sent_reptscore/%s.txt"%name, "w")
    all_distance = []
    for k, (ip, c) in enumerate(zip(ipts, cands)):
        if k % 1000 == 0:
            print("processing %d lines"%k)
        sen_list = sent_tokenize("%s %s"%(ip.strip(), c.strip()))
        if len(sen_list) < 2:
            print(name, ip, c)
            continue

        with torch.no_grad():
            encoded_input = sbert_tokenizer(sen_list, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            model_output = sbert_model(**encoded_input)
            #Perform pooling. In this case, mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()

        max_dis, min_dis, all_dis = [-1, -1, -1.], [-1, -1, 1.], []
        for i, sen1 in enumerate(sentence_embeddings):
            for j, sen2 in enumerate(sentence_embeddings):
                if i < j:
                    distances = 1 - scipy.spatial.distance.cdist([sen1], [sen2], "cosine")[0][0]
                    all_dis.append(distances)
                    if distances > max_dis[2]:
                        max_dis = [i, j, distances]
                    if distances < min_dis[2]:
                        min_dis = [i, j, distances]
        sort_dis = sorted(all_dis)
        all_distance.append([])
        for i in range(1, 31):
            all_distance[-1].append(np.mean(sort_dis[-i:]))

        ss_max.append(max_dis[2])
        ss_min.append(min_dis[2])
        ss_mean.append(np.mean(all_dis))
        fout.write("%.4f|||%s|||%s\n"%(max_dis[2], sen_list[max_dis[0]], sen_list[max_dis[1]]))
        fout.write("%.4f|||%s|||%s\n"%(min_dis[2], sen_list[min_dis[0]], sen_list[min_dis[1]]))
    fout.close()
    all_distance = np.mean(all_distance, 0)
    return {
            "sent_rept_max": "%.4f"%(np.mean(ss_max)),
            "sent_rept_min": "%.4f"%(np.mean(ss_min)),
            "sent_rept_mean": "%.4f"%(np.mean(ss_mean)),
            "all_distance": " ".join(["%.4f"%tmpf for tmpf in all_distance]),
        }

def pro(s, name=""):
    s = s.strip()
    # for i in range(10):
    #     s = s.replace("[%d]"%i, "")
    s = s.replace("<mask><s>", " ")
    s = " ".join(s.strip().split())
    # s = roberta_tokenizer.decode(roberta_tokenizer.convert_tokens_to_ids(roberta_tokenizer.tokenize(s)))
    return s

device = "cuda:0"
result_list = [
    "roc_gpt",
]

data_dir=sys.argv[1]
with open("%s/test.source"%data_dir, "r") as fin:
    ipt = [pro(line) for line in fin]
with open("%s/test.target"%data_dir, "r") as fin:
    truth = [pro(line) for line in fin]

def mask(slist):
    return [s.replace("[MALE]", "MALE").replace("[FEMALE]", "FEMALE").replace("[NEUTRAL]", "NEUTRAL") for s in slist]

def get_result(name, ipt, truth, cand):
    result = {}

    result.update(learnable_metric(name, ipt, cand, model_path="your_model_path1", type_="order", device=device))
    result.update(learnable_metric(name, ipt, cand, model_path="your_model_path2", type_="relate", device=device))

    ipt, truth, cand = mask(ipt), mask(truth), mask(cand)
    result.update(sent_semantic_repetition(name, ipt, cand, device=device, model_type="your_sbert_model_path3"))
    ipt_token, truth_token, cand_token = [tokenizer.tokenize(i) for i in ipt], [tokenizer.tokenize(t) for t in truth], [tokenizer.tokenize(c) for c in cand]
    result.update(bleu(truth_token, cand_token))
    result.update(repetition_distinct(name, cand_token))
    result.update(length(cand_token, name))
    key = sorted(result.keys())
    for k in key:
        print(name, k, result[k])
    print("="*10)
    return result

for name in result_list:
    if os.path.isdir("./%s"%name):
        name_list = []
        for _, _, fl in os.walk("./%s"%name):
            for f in fl:
                name_list.append(os.path.join("%s"%name, f.split(".")[0]))
            break
        name_list = list(sorted(name_list))
    else:
        name_list = [name]
    for name in name_list:
        cand = []
        with open("./%s.txt"%name, "r") as fin:
            for line in fin:
                cand.append(pro(line.strip().split("|||")[-1], name))
        result = get_result(name, ipt, truth, cand)