import os 
import csv
import torch
import numpy as np
import argparse
import pandas as pd 
import linecache
import json
from tqdm import tqdm
from openai import OpenAI
import random
import httpx

#os.environ["http_proxy"] = "http://localhost:7890"
#os.environ["https_proxy"] = "http://localhost:7890"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

def label_to_class(name,label): 
    label_id = int(label)
    return class_dict[name][label_id] 


def llama_prompt(name, prompt_type, template_id):
    prompt_path = os.path.join('/home/swl/prompt-universal-vulnerability/prompt_llama/prompt_llama', name, 'eval_prompt{}{}.txt'.format(prompt_type,template_id))
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt

def build_clean_eval_list(name, prompt_type, template_id):
    prompt_path = os.path.join('/home/swl/prompt-universal-vulnerability/prompt_llama/prompt_llama', name, 'manual_template.txt')
    prompt = linecache.getline(prompt_path, template_id).strip()
    dataset_path = os.path.join('/home/swl/prompt-universal-vulnerability/data', name, 'dev.tsv')
    test_set = pd.read_csv(dataset_path, sep='\t')
    with open(dataset_path, "r", encoding="utf-8", newline="") as tsvfile:
        test_set= csv.DictReader(tsvfile, delimiter="\t")
        test_input=[]
        test_label=[]
        for row in test_set:
            sentence = row["sentence"]
            test_input.append(sentence)
            label = row["label"]
            test_label.append(label)
        clean_input_sen = [llama_prompt(name, prompt_type, template_id) + "\n" + sentence + " " + prompt + " " + "=>" for sentence in test_input]
        test_label = [label_to_class(name,label) for label in test_label]
        if len(test_label) > 500:
            sample_indices = random.sample(range(len(clean_input_sen)), 500)
            clean_input_sen = [clean_input_sen[i] for i in sample_indices]
            test_label = [test_label[i] for i in sample_indices]
    return clean_input_sen, test_label

def build_trigger_eval_list(sentences, name, prompt_type, template_id, trigger, position):
    prompt_path = os.path.join('/home/swl/prompt-universal-vulnerability/prompt_llama', name, 'manual_template.txt')
    prompt = linecache.getline(prompt_path, template_id).strip()
    trigger_path = os.path.join('triggers', trigger)
    adv_input_li = []
    sentences = [sentence.replace(llama_prompt(name, prompt_type, template_id),"") for sentence in sentences]
    sentences = [sentence.replace(" =>","") for sentence in sentences]
    sentences = [sentence.replace(prompt,"") for sentence in sentences]
    for i in range(1):
        with open(trigger_path, "r") as json_file:
            trigger_file = json.load(json_file)
            trigger_words = trigger_file["triggers"][i]
            trigger_words = [trigger.replace("\u0120","") for trigger in trigger_words]

        if position == "suffix":
            adv_input = [llama_prompt(name, prompt_type, template_id) + sentence + " ".join(trigger_words) + " " + prompt + "=>" for sentence in sentences]
        elif position == "prefix":
            adv_input = [llama_prompt(name, prompt_type, template_id) + sentence + prompt + " " + " ".join(trigger_words) + "=>" for sentence in sentences]
        else:
            adv_input = []
            for j in range(int(len(sentences)/2)):
                adv_sentence = sentences[j] + " ".join(trigger_words) + " " + prompt + "=>"
                adv_input.append(adv_sentence)
            for j in range(int(len(sentences)/2), len(sentences)):
                adv_sentence = sentences[j] + prompt + " " + " ".join(trigger_words) + "=>"
                adv_input.append(adv_sentence)
        adv_input_li.append(adv_input)
    return adv_input_li

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate the ASR on GPT")
    parser.add_argument('--dataset', default='sst2',
                        choices=["sst2", "ag_news", "imdb", "twitter", "fakereview", "fakenews"])
    parser.add_argument('--trigger', default=None, help="the path to the trigger JSON file.")
    parser.add_argument('--template_id', type=int, default=1, choices=[1, 2], help="choose the template. 1 stands for manual template and 2 stands for null template. ")
    parser.add_argument('--prompt_type', type=int, default=1, choices=[0, 1], help="choose the prompt type for GPT. 0 stands for zero-shot and 1 stands for few-shot. ")
    parser.add_argument("--trigger_pos", choices=["prefix", "suffix", "all"], default="suffix",
                        help="The position of the trigger. \n"
                        "`prefix` means the trigger should be used by placing before the text, \n"
                        "`suffix` means the trigger should be used by placing after the text, \n"
                        "`both` means the trigger can be used either way.\n")
    params = parser.parse_args()
    name = params.dataset
    template_id = params.template_id
    position = params.trigger_pos
    prompt_type= params.prompt_type
    trigger = params.trigger

    class_dict = {'ag_news':["politics","sports","business","technology"],
                  'sst2':["bad","good"],
                  'imdb':["bad","good"],
                  'twitter':["harmless", "hate"],
                  'fakereview': ["real", "fake"],
                  'fakenews':["real", "fake"]}
                  
    result_dir = "GPT_result"
    os.makedirs(result_dir, exist_ok=True)
    result_file_name = "{}_{}.txt".format(params.trigger[13:21], position)
    result_file_path = os.path.join(result_dir, result_file_name)

    client = OpenAI(
    base_url="https://api.xty.app/v1", 
    api_key="sk-3ZDdZH3BV5PUddVm81D99e2843Ca4f93AfA369FfBdF184Cc",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    )
    )

    total_num = len(build_clean_eval_list(name, prompt_type, template_id)[1])
    print("=================================\nstart to calculate the accuracy")
    clear_count = 0

    sentences,labels = build_clean_eval_list(name, prompt_type, template_id)
    corr_sen=[]
    corr_label=[]
    answer_li=[]
    for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Processing"):
        label = labels[i]
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Response with 'harmless' or 'hate'."},
                    {"role": "user", "content": f"{sentence}"}
                
                ]
            )
        except Exception as e:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Response with 'harmless' or 'hate'."},
                    {"role": "user", "content": f"{sentence}"}
                
                ]
            )
            continue
        predict = completion.choices[0].message.content.lower()
        if predict == "political":
            predict = "politics"
        answer_li.append(predict)
        if label in predict:
            clear_count += 1
            corr_sen.append(sentence)
            corr_label.append(label)
        else: print(f"sentence: {sentence}\n true_label:{label} \n predict:{predict}")
    acc = 100 * clear_count/total_num
    with open(result_file_path, "a") as result_file:
            result_file.write("\ncurrent file:{}\n".format(params.trigger))
            result_file.write("para: dataset {}  template_id {} prompt type {}\n".format(params.dataset, params.template_id, prompt_type))
            result_file.write("ACC:{}\n".format(acc))
            result_file.write("Answer:{}\n".format(answer_li))
    print("==================================\nThe Acc is:{}. Start to calculate the ASR".format(acc))

    total_adv_num = len(corr_sen)
    indicator = np.zeros(total_adv_num)
    for i in range(1):
        adv_count = 0
        adv_sentences = build_trigger_eval_list(corr_sen, name, prompt_type, template_id, trigger, position)
        for j, sentence in tqdm(enumerate(adv_sentences[i]), total=len(adv_sentences[i]), desc=f"Processing iteration {i+1}"):
            adv_label = corr_label[j]
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Response with 'harmless' or 'hate'."},
                    {"role": "user", "content": f"{sentence}"}
                
                ]
            )
            predict = completion.choices[0].message.content.lower()
            if predict == "political":
                predict = "politics"
            if adv_label not in predict:
                indicator[j] = 1
            if j < 3:
                print(sentence)
        success_num = sum (1 for element in indicator if element != 0)
        print(success_num)
    
    success_num = sum (1 for element in indicator if element != 0)
    asr = 100 * success_num/total_adv_num
    print(success_num,total_adv_num,asr)

    with open(result_file_path, "a") as result_file:
        result_file.write("ASR:{}\n".format(asr))
    print("Results saved to:", result_file_path)
