
# 국방 AI 경진대회 코드 사용법  
> 팀명 : 중요한 건 꺾는 마음

정시현, 이준형, 김태민, 임서현   
sh2298, jjuun, taemin6697, 임리둥절

  
  
# 핵심 파일 설명 * 
 - 학습 데이터 경로: `./data`  
 - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 3개  
 - `./mymodel/models/last_v7.ckpt`  
 - `./mymodel/models/last_v10.ckpt`  
 - `./mymodel/models/last_v11.ckpt`  
 - 학습 메인 코드: `./train.ipynb`, `./train.py`  
 - 테스트 메인 코드: `./predict.py`  
 - 테스트 이미지, 마스크 경로: `./data/result/pred/mask `
 - 테스트 결과 이미지 경로: `./final_result/output_aipg`  
  
## 코드 구조 설명  *

 - **전처리**
  
    upsampling :  `./preprocess/augmentation.py` 
  
    cutmix :  `./preprocess/cutmix.py` 
  
    최종 사용 데이터 :  `./data/train # data + upsamping + cutmix  `
  
  
 - **모델  
   nvidia/segformer-b4-finetuned-cityscapes-1024-1024**로 사용하여 학습 및 테스트함.  
  
   parameter : iter 체크한거 첨부하면 좋을 듯. 수정 comment line으로 기재
  
    tuning : (batch, lr, 등..)  
  
   최종 사용 모델 : segformer-b4-  ( b4 18000iter lr 0.00005   )+ (...)
  
  
 - **predict**
   `./predict.py` 실행.

  
 - **최종 제출 파일 : submitted.zip**  
 - **학습된 가중치 파일 : training_results/submitted_model/iter_10000.pth**  
  
## 주요 설치 library   
 - transformers (4.24.0)
 - torch (1.12.0+cu113)
 - torchvision (0.13.0+cu113)
 
  
# How to use *
  
 - 가상환경 셋팅 후   
 ```  
pip install -r requirements.txt  
 ```

- 학습 실행 방법  
```
- 학습 데이터 경로 설정  
  `./configs/train.yaml` # ~~ 수정
  `./train.py` #~~ 수정	
  # 학습 데이터 절대경로명 설정 예시
  # data_dir: /home/user/detection/my_dataset/  
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
  
 - 실행  
  ```  
  ./predict.ipynb
 ```  
 - predict 
  ```  
 install -d ./final_result/output_aipg  
 python bin/predict.py   
 #어디에 저장됨..
  ```
   - predict 결과 
   
파라메타 @,@,@ 사용함.
상기의 3가지 결과를 ensemble 처리하여 최종 detection 결과 생성  
 ```
code(ensemble)
 ```  
 - plot
 (결과plot첨부? 있나)

## Architecture *
폴더 구조 그거 ㅇㅇ
