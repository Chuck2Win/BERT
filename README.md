# BERT

## Pretrain - MLM + NSP (data는 NSMC로 pretrain시킴 + NSMC를 finetunning까지 할 예정)
## Two Track으로 진행 중 - 연구실 컴퓨터 - 나무위키 100만건 sentence pretrain 
## Colab - NSMC 15만건 pretrain

### comment

BERT 논문에서 말하는 sequence는 single sentence or two sentences packed together

Token embedding : `WordPiece` 활용

![bert](https://github.com/Chuck2Win/BERT/blob/main/img/bert.png)

<CLS> sentence <SEP> sentence 로 구성

![embedding](https://github.com/Chuck2Win/BERT/blob/main/img/embedding.png)

Embedding은 Token embedding + Segment embedding + Position Embedding으로 구성

Segment embedding : pair로 된 sentence의 경우 적용 가능(QnA도 가능)



## MLM

sequence에서 15%에 `[MASK]`를 위한 후보로 선택 :heavy_check_mark: 15%만을 loss로 활용하기에 LM에 비해 많은 step 요구.

80% :  `[MASK]` 

```
my dog is hairy -> my dog is [MASK]
```

10% : 그대로 :ballot_box_with_check: representation이 실제를 반영할 수 있게끔(bias) 

```
my dog is hairy -> my dog is hairy
```

10% : random

```
my dog is hairy -> my dog is fat
```

목적 

Model의 `[MASK]` 예측, Model이 어떤 부분이 random으로 바뀌었는 지 예측.

:arrow_right: keep a distributional contextual representation (즉, 모든 token을 학습하게끔.)



## NSP

A :arrow_forward: B

50% : IsNext , 50% : NotNext



## 구현 - 모델 구현 자체는 어렵지 않다.

데이터 전처리+Embedding + Model 부분만 구현

(Word Piece는 huggingface 그것을 가져올 것)

:heavy_plus_sign: nn.multiheadattention 등은 활용 안하고, 복습 차원에서 nn.Linear로 구현



![transformer](https://github.com/Chuck2Win/BERT/blob/main/img/transformer.png)

input으로는, input ids(token)+segment_ids(segment)+masks(mask면 0, 아니면 1)


## HyperParameters

| Hyperparameter |             Original                    |           MINE                                          |
| -------------- | ------------------------------- | ----------------------------------------- |
| n_layers       | 12                              | 2                                         |
| n_head         | 12                              | 2                                         |
| d_model        | 768                             | 128                                       |
| d_ff           | 768*4                           | 128*4                                     |
|                |                                 |                                           |
| n_vocab        | 32000                           | 32000                                      |
| batch size     | 256                             | 32                                        |
| seq_len        | 512                             | 128                                       |
| epochs         | 40                              | 100                                       |
| Adam           | b1 0.9, b2=0.999, L2 decay 0.01 | b1 0.9, b2=0.999, L2 decay 0.01           |
| **lr**         | 1e-4                            | 1e-5,warm up 10,00steps and linear decay |
| **dropout**    | 0.1                             | 0.1                                       |
| activation     | gelu                            | gelu                                      |

## Status  
현재 학습이 안되고 있음
- TensorBoard로 Tracking(https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/tensorboard_with_pytorch.ipynb)
- Data의 수를 줄임(DN layer도 축소, no dropout and ReLU)  
- 학습은 잘 진행됨
- 현재, ReLU / GeLU로 나눠서 진행 중(dropout on/off)  

## 참고 공부

### Layer Normalization

Neuron의 Output을 normalize 시킴.

https://yonghyuc.wordpress.com/2020/03/04/batch-norm-vs-layer-norm/

https://jeongukjae.github.io/posts/layer-normalization/

![ln](https://github.com/Chuck2Win/BERT/blob/main/img/ln.png)

![bnln](https://github.com/Chuck2Win/BERT/blob/main/img/bnln.png)

Batch normalization - data preprocessing에서 했던,  `feature의 mean, std` 구하기

- batch size와 관련이 깊음

$$
\mu_{i} = \frac{1}{BS}\sum^{BS}_{n=1}x_{n,i}\\
    \sigma_{i}^2 = \frac{1}{BS}\sum^{BS}_{n=1}[x_{n,i}-\mu_i]^2
$$

- BS : Batch size, $\mu_i$ 에서 i는 i 번째 feature



Layer normalization - 각 `sample의 mean, std` 구하기

- batch size와 관련이 없음.
  $$
  \mu_{n} = \frac{1}{K}\sum^{K}_{i=1}x_{n,i}\\
  \sigma_{n}^2 = \frac{1}{K}\sum^{K}_{i=1}[x_{n,i}-\mu_n]^2
  $$

- K : feature의 개수

`Layer Normalization` 

n번째 sample의 k번째 feature
$$
\hat{x}_{n,k} = \frac{x_{n,k}-\mu_n}{\sqrt{\sigma_{n}^2+\epsilon}}
$$
scaling($\boldsymbol{\gamma}$) and shifting step($\boldsymbol{\beta}$)
$$
y_n = \boldsymbol{\gamma}\hat{\boldsymbol{x}}_{n}+\boldsymbol{\beta} \equiv LN_{\boldsymbol{\gamma},\boldsymbol{\beta}}(\boldsymbol{x}_{n})
$$


#### LN의 효과 - 직관적으로 정규화 시키니깐, 학습 속도 향상될 것으로 보임

1) Smoother gradients

2) Faster training

3) better generalization accuracy

Layer normalization은 CNN에선 잘 발휘가 안된다고 함.

`추가 공부` : Understanding and Improving Layer Normalization



## GeLU(Gaussian Error Linear Unit Activation)

![img](https://blog.kakaocdn.net/dn/baz54S/btqAVIiRUqA/42LPxGbfBgoBdIKuCVxE90/img.png)

실험 결과

MNIST - 8 layer, 128 hidden layer, 128 batch size

- error rate

| GeLU  | ReLU  | eLU   |
| ----- | ----- | ----- |
| 7.89% | 8.16% | 8.41% |

:arrow_forward: GeLU가 좋음.
$$
GeLU(x)=0.5x(1+\tanh(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)))
$$
![gelu](https://github.com/Chuck2Win/BERT/blob/main/img/gelu.png)

## Transformer encoder

![transformerencoder](https://github.com/Chuck2Win/BERT/blob/main/img/transformerencoder.png)



### BertTokenizer

https://keep-steady.tistory.com/37

https://paul-hyun.github.io/implement-paper/
