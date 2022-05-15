from unicodedata import name
from retrieval import Retrieval
from typing import List, NoReturn, Optional, Tuple, Union
import numpy as np
import json
import random
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

class BertEncoder(BertPreTrainedModel):

    def __init__(self,
        config
    ):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output
    
class DenseTrainer():
    def __init__(
        self, 
        args,
        tokenizer,
        p_encoder,
        q_encoder,
        data_path: Optional[str] = "/opt/ml/input/data/train_dataset", 
    ):
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.args = args
        dataset = load_from_disk("/opt/ml/input/data/train_dataset")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        p_with_neg = self.prepare_in_batch_negative(train_dataset)
        q_seqs = tokenizer(
            train_dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, 16, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, 16, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, 16, max_len)
        self.train_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )
        p_with_neg = self.prepare_in_batch_negative(eval_dataset)
        q_seqs = tokenizer(
            eval_dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, 16, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, 16, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, 16, max_len)
        self.eval_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )
    def prepare_in_batch_negative(self,train_dataset):
        num_neg = 15
        corpus = np.array(train_dataset["context"])
        p_with_neg = []

        for c in train_dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)
                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break
        return p_with_neg
    def train(self):
        args = self.args
        train_dataset = self.train_dataset
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder
        num_neg = 15
        batch_size = args.per_device_train_batch_size
    
        # Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:

            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    p_encoder.train()
                    q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    del batch
                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = p_encoder(**p_inputs)
                    # (batch_size, emb_dim)  
                    q_outputs = q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    q_encoder.zero_grad()
                    p_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

        return p_encoder, q_encoder


def main():
    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=2, # 아슬아슬합니다. 작게 쓰세요 !
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
    )
    model_checkpoint = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()
    trainer = DenseTrainer(args= args, tokenizer= tokenizer, p_encoder= p_encoder, q_encoder= q_encoder)
    p_encoder, q_encoder = trainer.train()
    torch.save(p_encoder, "./dense_encoder")
    torch.save(q_encoder, "./dense_encoder")

if __name__ == "__main__":
    main()