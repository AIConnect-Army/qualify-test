


# 국방 AI 경진대회 (본선) 코드 사용법  
> 팀명 : 중요한 건 꺾는 마음 ~~하지만 꺾여버린..~~

정시현, 이준형, 김태민, 임서현   
sh2298, jjuun, taemin6697, 임리둥절

  
  
  
  
# 핵심 파일 설명 * 
 - 학습 데이터 경로: `./data~~~`  
 - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 3개  (그냥 잘나온 거 세개)
 - `./!!!.ckpt`  
 - `./~~ckpt`  
 - `./~~.ckpt`  
 - 학습 메인 코드:  `./~~.py`  
 - 테스트 메인 코드: `./~~.py`  
 - 테스트 이미지, 마스크 경로: `./data/~~ `
 - 테스트 결과 이미지 경로: `./final_result/~~~~~`  
  
  
  
  
  
## 코드 구조 설명  *

 - **전처리**
  
	mixup :  `./~/~.py ~line` 
  
    Brightness :  `./~~.py ~line` 
  
    최종 사용 데이터 :  `./data/train # data + mixup + brightness  `
  
  
  
  
  
 - **모델  
   MPR NET ( Multi-Stage Progressive Image Restoration)** 로 사용하여 학습 및 테스트함.  
  
   parameter : 18000iter
  
    tuning : batch , lr 
  
   최종 사용 모델 : 
   MPR NET ( Multi-Stage Progressive Image Restoration) (~iter lr ~)
  
  
 - **predict**
   `./~~.py` 실행.

  
 - **최종 제출 파일 : submitted.zip**  
 - **학습된 가중치 파일 : ~~**  
  
  
  
  
  
## 주요 설치 library   
 - transformers (~)
 - torch (~)
 - torchvision (~)
 
 
 
 
  
# How to use  해야댐?*

  ## SERVER P40 (CUDA 11.3)
 - 가상환경 셋팅 후   
 ```  
pip install -r requirements.txt  
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

 ```

- 학습 실행 방법  
```
- 학습 데이터 경로 설정  
  `./configs/train.yaml` # ~~ 수정
  `./train.py` #~~ 수정	
  # 학습 데이터 절대경로명
  # data_dir: ./data/train
``` 
 - 학습 실행  
 ```
  ./train.ipynb
 ``` 
 - 실행 내용  
첫번째 Model 설정 기반 학습: ~~.ckpt 획득     python bin/train.py -cn big-lama-aigp-1 location=my_dataset data.batch_size=5      
두번째 Model 설정 기반 학습: last_v11__cdb2dc80b605a5e59d234f2721ff80ea.ckpt 획득   
그래서 어디에 저장됨.~ 정도





# Predict *
 - predict 
  ```  
 python predict.py   
 #어디에 저장됨..
  ```
   - predict 결과 
   사진과 같이 결과가 나와서 ~ val miou 이렇게 좋아서 ~ 이거 사용 ~
   
   
최종 결과 파라메타 @ 사용함.

## Architecture *
폴더 구조 그거 ㅇㅇ
