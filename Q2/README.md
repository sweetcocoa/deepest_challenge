# Q2 요약 : 이름 기반 성별 추측 사용 #
---
## 접근 방법 ##
- No Neural Net
- Machine Learning
1. 논문 제목의 단어 갯수
2. 논문 저자의 수
3. 논문 제목의 길이
4. **논문 제 1저자의 성별**

## 접근 절차 ##
1. 단어갯수나 저자수 등등 구함
2. 논문 제 1저자 성별은 논문 제1저자의 First name을 가지고 판단함
3. 이름갖고 성별을 구해주는 API가 존재함(코드참고)

## 결과 ##
![각 값 관계성 plotting](https://github.com/sweetcocoa/deepest_challenge/raw/master/Q2/img/corr.jpg)

### ** Summary 참조 ** ###  
- 성별을 참조하면 될 거라고 생각했지만 안 됨
- 그보다 Accept된 논문이 평균적으로 논문제목의 길이가 짧고 저자가 많다.
- 근데 다 Reject때리는게 제일 잘됨 (논문제목길이, 저자수 모두 Correlation 0.1)
- 아무튼 안 됨
- 국적까지 보면 될 수도 있을 것 같다는 생각이 들지만.. 누가 했겠죠?

 
