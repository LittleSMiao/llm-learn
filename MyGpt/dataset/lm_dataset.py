from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 部分 tokenizer 没有 pad_token_id，兜底用 0（通常不影响，因为 padding 位置的 label 是 -100）
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = tokens + [pad_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100
        return input_ids, labels

def prob_remove_empty_think(prompt, prob=0.8):
    empty_pattern = '<think>\n\n</think>\n\n'
    if empty_pattern in prompt and random.random() < prob:
        prompt = prompt.replace(empty_pattern, "")
    return prompt

def SFTDataSet(Dataset):
    def __init__(self, max_length, tokenizer, json_file):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=json_file, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def get_fully_conversation_string(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools") is not None:
                tools = json.load(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            else if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message = json.load(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def mask_prompt_ids(self, prompt_ids):
        masked_prompt = [-100] * self.max_length
        index = 0

        while index < len(prompt_ids):
            if prompt_ids[index : index + len(bos_id)] == bos_id:
                start_pos = index + len(bos_id)
                index = start_pos:
                while prompt[index : index + len(eos_id)] != eos_id and index < len(prompt_ids):
                    index += 1
                end_pos = index + len(eos_id) if index + len(eos_id) < len(prompt_ids) else len(prompt_ids)
                for fill_index in range(start_pos, end_pos):
                    masked_prompt[fill_index] = prompt_ids[fill_index]
                index = end_pos
            else:
                index += 1

        return masked_prompt

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = get_fully_conversation_string(sample["conversations"])
        prompt = prob_remove_empty_think(prompt)
        prompt_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        prompt_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(prompt_ids))
        labels = self.mask_prompt_ids(prompt_ids)
        return torch.tensor(prompt_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

