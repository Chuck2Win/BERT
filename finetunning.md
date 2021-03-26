# BERT model 직접 만들고 pretrain 시켜보기
- 처음에 나무위키 덤프 데이터를 토대로 하려고 했지만.. 1억2000만개의 문장인지라.. 졸업 전에 학습이 불가능할 것으로 생각되서
- NSMC 데이터만을 가지고 pretrain시키고, 이를 이용해서 classification 진행해볼 예정


## BERT Model

## Pretrain dataset

- 나무위키

### reference

#### 나무위키 덤프 데이터 활용법

- 크롤링을 하게 되면 서버에 많은 부담을 준다고 하네.

https://heegyukim.medium.com/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4-json-%EB%8D%A4%ED%94%84-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%8C%8C%EC%8B%B1%ED%95%98%EA%B8%B0-8f41cee1e155

http://www.engear.net/wp/tag/%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4/

https://velog.io/@nawnoes/%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%A0%84%EC%B2%98%EB%A6%AC-%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4-%EB%8D%A4%ED%94%84-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%96%BB%EA%B8%B0



```python
# 필요 패키지
pip install ijson
pip install namu-wiki-extractor
# 필요 자료
# 나무위키 데이터 베이스 덤프 파일(다운로드를 위해선 Tor 브라우저 설치 요)
# Tor 브라우저를 활용해서 데이터 다운로드 완료
```

## 

#### ijson

json 파일을 읽으면서 parsing (쉽게 이야기해서 텍스트 문서를 한방에 처리하는 것이 아니라 순차적으로 처리)

## Pretrain Dataset

#### 고민

Mask를 씌울때, pickle 형태로 저장?? - 2021-03-23 ▶️ 옙





## Fine tunning dataset

- NSMC
- KoSquad
