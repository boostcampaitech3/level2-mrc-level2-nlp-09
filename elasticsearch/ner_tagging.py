import json
import pprint
import pandas as pd
from pororo import Pororo
from datasets import Dataset, concatenate_datasets, load_from_disk

# arrow 파일 읽어오기
train_dir = "../data/train_dataset"
test_dir = "../data/test_dataset"

train_org_dataset = load_from_disk(train_dir)
train_dataset = concatenate_datasets(
    [train_org_dataset["train"].flatten_indices(),
     train_org_dataset["validation"].flatten_indices(),]
)  

test_org_dataset = load_from_disk(test_dir)
test_dataset = concatenate_datasets(
    [test_org_dataset["validation"].flatten_indices(),]
)  
print("train:", len(train_dataset), "test:", len(test_dataset))


ner = Pororo(task="ner", lang="ko")

# train data tagging
train_tagging = [ner(train) for train in train_dataset["question"]]
train_dict = {
    "question": train_dataset["question"],
    "ner_tagging": train_tagging
}
train_df = pd.DataFrame(train_dict)
train_df.to_csv("train_tagging.csv", index=False)


# test data tagging
test_tagging = [ner(data) for data in test_dataset["question"]]
test_dict = {
    "question": test_dataset["question"],
    "ner_tagging": test_tagging
}
test_df = pd.DataFrame(test_dict)
test_df.to_csv("test_tagging.csv", index=False)
