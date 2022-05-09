import json
import pprint
import warnings
from tqdm import tqdm
from elasticsearch import Elasticsearch

warnings.filterwarnings('ignore')


# elasticsearch 서버 세팅 
def es_setting(index_name="origin-wiki"):
    es = Elasticsearch('http://localhost:9200', tineout=30, max_retries=10, retry_on_timeout=True)
    print("Ping Elasticsearch :", es.ping())
    # print('Elastic search info:')
    # print(es.info())

    return es, index_name

# 인덱스 생성
def set_index(es, index_name, setting_path):
    # 이미 인덱스가 존재하는 경우 삭제
    if es.indices.exists(index_name):
        print("Index already exists. Creating a new one after deleting it...")
        es.indices.delete(index=index_name)

    with open(setting_path, "r") as f:
        setting = json.load(f)
    es.indices.create(index=index_name, body=setting)
    print("Index creation has been completed")

# 위키피디아 데이터 로드
def load_data(dataset_path):
    # dataset_path = "../data/wikipedia_documents.json"
    with open(dataset_path, "r") as f:
        wiki = json.load(f)

    wiki_contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    wiki_articles = [
        {"document_text": wiki_contexts[i]} for i in range(len(wiki_contexts))
    ]
    return wiki_articles

# 인덱스에 데이터 삽입
def insert_data(es, index_name, dataset_path):
    wiki_articles = load_data(dataset_path)
    for i, text in enumerate(tqdm(wiki_articles)):
        try:
            es.index(index=index_name, id=i, body=text)
        except:
            print(f"Unable to load document {i}.")

    n_records = es.count(index=index_name)["count"]
    print(f"Succesfully loaded {n_records} into {index_name}")
    print("@@@@@@@ 데이터 삽입 완료 @@@@@@@")

# 삽입한 데이터 확인
def check_data(es, index_name, doc_id=1):
    print('샘플 데이터:')
    doc = es.get(index=index_name, id=doc_id)
    pprint.pprint(doc)


def es_search(es, index_name, question, topk):
    # question = "주기표는 무엇인가?"
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"document_text": question}}
                ]
            }
        }
    }

    res = es.search(index=index_name, body=query, size=topk)
    return res

if __name__ == "__main__":
    INDEX_NAME = "origin-wiki"
    setting_path = "./setting.json"
    dataset_path="../data/wikipedia_documents.json"

    es, index_name = es_setting(index_name=INDEX_NAME)
    set_index(es, index_name, setting_path)  # 이미 인덱스가 존재하면 주석처리하기
    insert_data(es, index_name, dataset_path)  # 이미 인덱스 안에 데이터가 존재하면 주석처리하기
    check_data(es, index_name, doc_id=1)


    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    res = es_search(es, index_name, query, 10)
    print("========== RETRIEVE RESULTS ==========")
    pprint.pprint(res)


    print('\n=========== RETRIEVE SCORES ==========\n')
    for hit in res['hits']['hits']:
        print("Doc ID: %3r  Score: %5.2f" % (hit['_id'], hit['_score']))