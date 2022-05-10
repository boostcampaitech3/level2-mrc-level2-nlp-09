# Elasticsearch 사용하는 경우
--index_name "origin-wiki"


# 기존 BM25 사용하는 경우
--elastic False \ 
--bm25 True (default 설정)



# TF-IDF 사용하는 경우
--elastic False \
--bm25 False


default 설정은 Elasticsearch 사용
elastic = False로 하는 경우 default는 BM25