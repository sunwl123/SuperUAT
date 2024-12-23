import argparse
import datetime
import json
import os
import gc
import random
import datetime

import datasets
import numpy as np
import pandas
import torch
import tqdm
import torch.nn as nn
from nltk.tokenize import sent_tokenize
from torch.nn import functional as F

from atop_utils import RobertaClassifier
from transformers import AutoTokenizer, RobertaForMaskedLM

datasets.set_caching_enabled(False)

exp_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

output_dir_root = "triggers/"


def dataset_mapping_wiki(item, tokenizer, trigger_pos_config):
    """Process wiki data. For each sentence, mask one word and insert <trigger> near the <mask>."""
    if item["text"].startswith("= ") or len(item["text"].split()) < 20:
        return None

    list_of_sents = sent_tokenize(item["text"])
    st = np.random.randint(len(list_of_sents))
    ed = st + 3 + np.random.randint(5)
    text = " ".join(list_of_sents[st:ed])

    toks = tokenizer.tokenize(text.strip())
    if len(toks) < 20:
        return None

    if trigger_pos_config == "all":
        trigger_pos_config = np.random.choice(["prefix", "suffix"])

    if trigger_pos_config == "prefix":
        toks = toks[:100]
        mask_pos = np.random.choice(int(0.1 * len(toks)))
        trigger_pos = min(mask_pos + np.random.randint(5) + 1, len(toks))
    elif trigger_pos_config == "suffix":
        toks = toks[-100:]
        mask_pos = np.random.choice(int(0.1 * len(toks)))
        mask_pos = len(toks) - mask_pos - 1
        trigger_pos = max(0, mask_pos - np.random.randint(5))
    else:
        assert 0

    label = tokenizer.vocab[toks[mask_pos]]
    toks[mask_pos] = "<mask>"
    toks = toks[:trigger_pos] + ["<trigger>"] + toks[trigger_pos:]
    return {
        "x": tokenizer.convert_tokens_to_string(toks),
        "y": label
    }


def my_search_triggers_on_pretrained_lm(victim, dataset, tokenizer, epoch, batch_size,
                                     trigger_len, used_tokens, topk=64, bin_id=0):
    word2id = victim.word2id
    embedding = victim.embedding
    id2word = {v: k for k,v in word2id.items()}
    def get_candidates(gradient, current_word_ids, pos):
        args = (embedding - embedding[current_word_ids[pos]]).dot(gradient.T).argsort()
        ret = []
        if pos == 0:
            for idx in args:
                if idx == current_word_ids[pos]:
                    continue
                if id2word[idx] in used_tokens:
                    continue
                if id2word[idx][0] == "Ġ":
                    tmp = [idx] + current_word_ids[pos + 1:]
                    tmp_detok = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tmp))
                    tmp_rec = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tmp_detok))
                    if len(tmp_rec) != len(tmp) or tmp[pos] != tmp_rec[pos]:
                        continue
                    ret.append(id2word[idx])
                    if len(ret) == topk:
                        break
        else:
            for idx in args:
                if idx == current_word_ids[pos]:
                    continue
                if id2word[idx] in used_tokens:
                    continue
                word = id2word[idx]
                # ignore special tokens
                if len(word) == 0 or (word[0] == "<" and word[-1] == ">"):
                    continue
                tmp = current_word_ids[:pos] + [idx] + current_word_ids[pos + 1:]
                tmp_detok = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tmp))
                tmp_rec = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tmp_detok))
                if len(tmp_rec) != len(tmp) or tmp[pos] != tmp_rec[pos]:
                    continue
                ret.append(id2word[idx])
                if len(ret) == topk:
                    break

        if len(ret) != topk:
            print("warning", current_word_ids)
        return ret

    trigger_tmp = []
    for i in range(trigger_len):
        tmp = np.random.choice(list(word2id.keys()))
        while len(tmp) == 0 or tmp[0] != "Ġ":
            tmp = np.random.choice(list(word2id.keys()))
        trigger_tmp.append(tmp)

    for epoch_idx in range(epoch):
        for num_iter in tqdm.tqdm(range((len(dataset) + batch_size - 1) // batch_size),
                                  desc="Trigger %d Epoch %d: " % (bin_id, epoch_idx)):
            cnt = num_iter * batch_size
            batch = dataset[cnt: cnt + batch_size]

            x = [tokenizer.tokenize(" " + sent) for sent in batch["x"]]
            y = batch["y"]
            
            victim.set_trigger(trigger_tmp)
            _,_,loss = victim.predict(x,labels=y,return_loss=True)
            loss_record = (trigger_tmp, loss)
            print("=======")
            print(f"loss record:{loss_record}")
            all_best = (trigger_tmp, loss)
            for idx in range(50):
                print(f"Begin trial {idx}")
                victim.set_trigger(all_best[0])
                grad = victim.get_grad(x, labels=y)[1]
                trigger_list = []
                for i in range(trigger_len):
                    candidate_tmp = get_candidates(grad[:, i, :].mean(axis=0),
                                                      tokenizer.convert_tokens_to_ids(all_best[0]), pos=i)
                    for candidate_choice in candidate_tmp:
                        trigger_choice = all_best[0][:i] + [candidate_choice] + all_best[0][i+1:]
                        trigger_list.append(trigger_choice)
                best_candidates = []
                random_candidates = random.sample(trigger_list, 96)
                for cw in random_candidates:
                    victim.set_trigger(cw)
                    _,_,loss = victim.predict(x,labels=y,return_loss=True)
                    best_candidates.append((cw, loss))
                all_best = sorted(best_candidates, key=lambda x:x[1])[0]
                trigger_tmp = all_best[0]
                print(all_best[0])
                print("\n")
                print(f"Idx:{idx}, trigger:{all_best[0]}, loss:{all_best[1]}")
                    
    return all_best[0]


def parse_args():
    parser = argparse.ArgumentParser("""Search for adversarial triggers on RoBERTa-large.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--trigger_len", type=int, default=3,
                        help="The length of the trigger.")
    parser.add_argument("--num_triggers", type=int, default=1, help="Number of triggers to be found.")
    parser.add_argument("--trigger_pos", choices=["prefix", "suffix", "all"], default="prefix",
                        help="The position of the trigger. \n"
                        "`prefix` means the trigger should be used by placing before the text, \n"
                        "`suffix` means the trigger should be used by placing after the text, \n"
                        "`both` means the trigger can be used either way.\n")

    parser.add_argument("--subsample_size", type=int, default=512,
                        help="Subsample the dataset. The sub-sampled dataset will be evenly splitted to search for "
                        "each trigger. \n"
                        "By default, a total of 1536 sentences will be splitted to three 512-sentence "
                        "subsets to find 3 triggers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Trigger search batch size.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to search each trigger.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")

    parser.add_argument("--loss_type", choices=["ce", "prob", "prob_sq", "mprob", "mprob_sq", "embdis"],
                        default="ce", help="Choose the objective to search for the trigger. \n"
                        "ce: maximize the cross entropy loss of the masked word. (Used in the paper).\n"
                        "prob, prob_sq: minimize the (square of) probability of correctly predict the masked word.\n"
                        "mprob, mprob_sq: minimize the (square of) probability of the maximum predicted probability "
                        "at the masked position.\n"
                        "embdis: maximize the embedding shift before and after inserting the trigger.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id.")
    return vars(parser.parse_args())


def main():
    starttime = datetime.datetime.now()
    meta = parse_args()

    print("Load roberta large.")
    np.random.seed(meta["seed"])
    torch.manual_seed(meta["seed"])
    victim = RobertaClassifier(torch.device("cuda:%d" % meta["gpu"]), loss_type=meta["loss_type"])

    # Load dataset =============================================
    print("Load dataset.")
    meta["dataset"] = dataset_name = "wikitext"
    dataset_subset = "wikitext-2-raw-v1"
    trainset_raw = list(datasets.load_dataset('/home/swl/wikitext-2-raw-v1', split="train"))
    np.random.shuffle(list(trainset_raw))
    trainset = []
    for item in trainset_raw:
        item_tmp = dataset_mapping_wiki(item, tokenizer=victim.tokenizer, trigger_pos_config=meta["trigger_pos"])
        if item_tmp is not None:
            trainset.append(item_tmp)
            if len(trainset) == meta["subsample_size"]:
                break

    assert len(trainset) == meta["subsample_size"]

    # Search for triggers ======================================
    used_tokens = []

    bin_size = len(trainset) // meta["num_triggers"]
    triggers = []
    for bin_id in range(meta["num_triggers"]):
        bin_data = trainset[bin_size * bin_id:bin_size * (bin_id + 1)]

        trigger_tmp = my_search_triggers_on_pretrained_lm(
            victim, datasets.Dataset.from_pandas(pandas.DataFrame(bin_data)), victim.tokenizer,
            epoch=meta["num_epochs"], batch_size=meta["batch_size"],
            trigger_len=meta["trigger_len"], used_tokens=used_tokens, bin_id=bin_id)
        triggers.append(trigger_tmp)
        used_tokens += trigger_tmp

    print(triggers)
    meta["triggers"] = triggers

    # Save results ==========================================
    os.makedirs(output_dir_root, exist_ok=True)
    with open(output_dir_root + "{pos}_len{len}_loss{loss}_seed{seed}_{exp_id}.json".format(
              pos=meta["trigger_pos"], len=meta["trigger_len"], loss=meta["loss_type"],
              seed=meta["seed"], exp_id=exp_id), "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    main()
