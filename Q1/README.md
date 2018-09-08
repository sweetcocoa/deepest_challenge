# Q1 Model Compression #

## 코딩비화 ##
- Squeezenet Pretrained model로 도저히 훈련이 안 돼서(뭔지 아마 버그같은데)
그냥 결국 Resnet 씀
- 목표했던 Squeezenet보다 Param 개수는 2.5배정도 많음.

## 모델 설명 ##
- Pretrained resnet18에서 Layer를 4개 중 3개만 씀
- 빠진 레이어 하나는 가장 parameter가 많은 마지막 레이어

## 데이터셋 ##
Pytorch ImageFolder 구조 다들 알잖아요
```
Images/train/각 클래스명/~~.jpg
Images/test/각 클래스명/~~.jpg
```
로 분할해서 train만 학습에 사용함
- 지금 README 쓰다 생각났는데
- test set에다 대고 early stopping을 한게 좀 문제이긴 하네요


## To Reproduce ##

- (Best Accuracy Model Download)[]
- (Best Score Model Download)[]

```commandline
python main.py
```

** 실행을 위해서는 소스를 참조하세요 (argparse 안하고 소스에서 hyperparam 하드코딩됨) **

### Best Accuracy 모델 : ### 
- model : "big_resnet"

### Best Score 모델  ###
- model : "splicedresnet" ### 

### 그 외 설정사항 ### 
- default_args의 train, test dataset 위치를 지정해줘야 함 (ImageFolder 구조)
- batch_size를 실행 환경에 맞게 적당히 조절
- pretrained model 위치도 조절
- CUDA_VISIBLE_DIVICES 도 수정

## Best Score : 47.085157492 ##

- Precision @ Top1 test : 52.796
- Precision @ Top5 test : 80.652
- "#" of Params : 2834184
```python
# param 개수 구했던 함수인데 만약 이게 아니라면...
def get_n_params(model):
    return sum(p.numel() for p in model.parameters())
```
##### TMI #####
- Precision @ Top1 train : 100.000
- Precision @ Top5 train : 100.000



## Best Top1 Accuracy : 63.307 ##
- Precision @ Top1 test : 63.307
- "#' of Params : 11689512 ~ 1e7
 

