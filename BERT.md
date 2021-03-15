# BERT

## MLM + NSP

### comment

BERT 논문에서 말하는 sequence는 single sentence or two sentences packed together

Token embedding : `WordPiece` 활용

<img src="C:\Users\admin\Desktop\bert.png" alt="bert"  />

<CLS> sentence <SEP> sentence 로 구성

![embedding](C:\Users\admin\Desktop\embedding.png)

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



## 구현

데이터 전처리+Embedding + Model 부분만 구현

(Word Piece는 huggingface 그것을 가져올 것)

:heavy_plus_sign: nn.multiheadattention 등은 활용 안하고, 복습 차원에서 nn.Linear로 구현 예정

![transformer](C:\Users\admin\Desktop\transformer.png)

## HyperParameters

| Hyperparameter |                                 |                                      |
| -------------- | ------------------------------- | ------------------------------------ |
| n_layers       | 12                              |                                      |
| n_head         | 12                              |                                      |
| d_model        | 768                             |                                      |
| d_ff           | 768*4                           |                                      |
| dropout        | 0.1                             |                                      |
| n_vocab        | 32000                           |                                      |
| batch size     | 256                             |                                      |
| seq_len        | 512                             |                                      |
| epochs         | 40                              |                                      |
| Adam           | b1 0.9, b2=0.999, L2 decay 0.01 |                                      |
| lr             | 1e-4                            | warm up 10,000steps and linear decay |
| dropout        | 0.1                             |                                      |
| activation     | gelu                            |                                      |



## 참고 공부

### Layer Normalization

Neuron의 Output을 normalize 시킴.

https://yonghyuc.wordpress.com/2020/03/04/batch-norm-vs-layer-norm/

https://jeongukjae.github.io/posts/layer-normalization/

![ln](C:\Users\admin\Desktop\ln.png)

![bnln](C:\Users\admin\Desktop\bnln.png)

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
![gelu](C:\Users\admin\Desktop\gelu.png)

## Transformer encoder

![transformerencoder](C:\Users\admin\Desktop\transformerencoder.png)