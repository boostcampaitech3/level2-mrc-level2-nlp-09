# Open-Domain Question Anwering Competition - NLP 9조 (MnM)
> MRC(Machine Reading Comprehension) dataset으로 Open-Domain Question Answering을 수행하는 task입니다.
> 
> 질문에 관련된 문서를 찾는 Retriever와 찾아온 문서에서 질문에 대한 정답을 찾는 Reader로 구성됩니다. 

## [Wrap-up report](https://colorful-bug-b35.notion.site/MRC-Wrap-up-report-367aeebe548448ab8d157fd86aa0196d)

## [Solution Presentation](https://github.com/boostcampaitech3/level2-mrc-level2-nlp-09/blob/develop/assets/MRC_%E1%84%89%E1%85%A9%E1%86%AF%E1%84%85%E1%85%AE%E1%84%89%E1%85%A7%E1%86%AB%E1%84%87%E1%85%A1%E1%86%AF%E1%84%91%E1%85%AD_NLP9%E1%84%8C%E1%85%A9.pdf)

## MnM Team

|[김태일](https://github.com/detailTales)|[문찬국](https://github.com/nonegom)|[이재학](https://github.com/wogkr810)|[하성진](https://github.com/maxha97)|[한나연](https://github.com/HanNayeoniee)|
|:-:|:-:|:-:|:-:|:-:|
|<a href="https://github.com/detailTales"><img src="assets/profile/ty.png" width='300px'></a>|<a href="https://github.com/nonegom"><img src="assets/profile/cg.png" width='300px'></a>|<a href="https://github.com/wogkr810"><img src="assets/profile/jh.png" width='300px'></a>|<a href="https://github.com/maxha97"><img src="assets/profile/sj.png" width='300px'></a>|<a href="https://github.com/HanNayeoniee"><img src="assets/profile/ny.png" width='300px'></a>|

## Members' Role
| Member | Role | 
| --- | --- |
| 김태일 | 실험 세팅, BM25, rerank, DPR 구현 및 실험 |
| 문찬국 | 협업 툴 관리, 실험 세팅, KorQuAD Fine-tuning, Curriculum Learning, Hyperparameter Tuning |
| 이재학 | EDA, Scheduler 실험, 성능 검증 코드 제작, Data Length 관련 실험, Inference 후처리, Ensemble |
| 하성진 | Pre-trained 모델 실험, KorQuAD Pre-training  |
| 한나연 | EDA, Pre-trained 모델 실험, Elasticsearch, NER tagging |

## Score
▶️ Public Leaderboard: 1st / 11
<img width="1277" alt="image" src="https://user-images.githubusercontent.com/33839093/169644650-3d21da45-4e06-46a8-879f-80479fd93f3e.png">

▶️ Private Leaderboard: 1st / 11
<img width="1273" alt="image" src="https://user-images.githubusercontent.com/33839093/169644744-434d9dcf-4734-4340-8a9f-df382861550c.png">

## Our Solution
- Data Processing
    - Remove punctuation marks
- Reader Model
    - klue\roberta-large + fine-tuning on KorQuAD + fine-tuning on our own train dataset + curriculum learning
- Retriever Model
    - BM25
    - Elasticsearch BM25
- Ensemble
    - Soft voting
    - Hard voting
    - Post processing


## Dataset
아래는 제공하는 데이터셋의 분포를 보여줍니다.

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 

![데이터 분포](./assets/dataset.png)

## Evalutaion Metric
EM(Exact Match)와 F1 score 두 개의 평가지표를 사용하지만, EM기준으로 리더보드 등수가 반영되고, F1은 참고용으로만 활용됩니다.

`Exact Match`: 모델의 예측과 실제 답이 정확하게 일치할 때만 점수가 주어지고, 띄어쓰기나 문장부호를 제외한 후 정답에 대해서만 일치하는지 확인합니다. 또한 답이 하나가 아닐 수 있는데, 이런 경우는 하나라도 일치하면 정답으로 간주합니다.

![EM](https://user-images.githubusercontent.com/46811558/169698749-b5c5cec0-1ae8-4260-a406-01a805268604.png)

`F1 Score`: EM과 다르게 부분 점수를 제공합니다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 받을 수 있습니다.

![F1](https://user-images.githubusercontent.com/46811558/169698799-e6000a79-6dd0-435f-9a95-5293bdbc6953.png)

## Usage

### 1) Installation
- 저장소를 clone한 후에 필요한 파이썬 패키지를 설치합니다.
- Elasticsearch 설치와 사용법은 [여기](https://www.notion.so/Elasticsearch-e7901787b60b4024959e5b009d5d8594)를 참고해주세요.
```
git clone https://github.com/boostcampaitech3/level2-mrc-level2-nlp-09.git
cd level2-mrc-level2-nlp-09
bash ./install/install_requirements.sh
```

### 2) Train model
arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 에서 확인 가능합니다. 

```bash
python train.py \
--output_dir ./models/train_dataset \
--learning_rate 3e-5 \
--per_device_train_batch_size 16 \
--eval_steps 500 \
--do_train \
--save_steps 500 \
--logging_steps 500 \
--do_eval \
--evaluation_strategy "steps" \
--num_train_epochs 3 
```


### 3) Inference

- retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행해 `predictions.json` 이라는 파일이 생성됩니다. 

- 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다. 

- 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`--do_eval`)를 진행하면 됩니다.

- wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다.

```bash
python inference.py \
--output_dir ./outputs/test_dataset/ \
--dataset_name ../data/test_dataset/ \
--model_name_or_path "Nonegom/roberta_finetune_twice" \
--index_name "origin-wiki" \
--top_k_retrieval 40 \
--do_predict
```


## Code Structure
```bash
level2-mrc-level2-nlp-09
├── EDA
│   ├── EDA.ipynb
│   ├── EDA_tokenizer.ipynb
├── assets
├── elasticsearch
├── install
├── utils
│   ├── SOTA_comparison.ipynb
│   ├── ensemble_hardvoting.ipynb
│   ├── ensemble_softvoting.ipynb
│   ├── post_process.ipynb
├── arguments.py
├── elastic_setting.py
├── inference.py
├── retrieval.py
├── setting.json
├── sweep.yaml
├── train.py
├── train_sweep.py
├── trainer_qa.py
└── utils_qa.py
```

## Git Commit Rule
```
- feat      : 새로운 기능 추가
- debug     : 버그 수정
- docs      : 문서 수정
- style     : 코드 formatting, 세미콜론(;) 누락, 코드 변경이 없는 경우
- refactor  : 코드 리팩토링
- test      : 테스트 코드, 리팩토링 테스트 코드 추가
- chore     : 빌드 업무 수정, 패키지 매니저 수정
- exp       : 실험 진행
- merge     : 코드 합칠 경우
- anno      : 주석 작업
- etc       : 기타
```
