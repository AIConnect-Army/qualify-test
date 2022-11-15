# 국방 AI 경진대회 코드 사용법
- 중요한 건 꺾는 마음팀, 정시현, 이준형, 김태민, 임서현
- 닉네임 : sh2298, jjuun, taemin6697, 임리둥절


# 핵심 파일 설명
  - 학습 데이터 경로: `./data`
  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터: `./LaMa_models/big-lama-with-discr/models/best.ckpt`
  - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 3개
    - `./mymodel/models/last_v7.ckpt`
    - `./mymodel/models/last_v10.ckpt`
    - `./mymodel/models/last_v11.ckpt`
  - 학습 실행 스크립트: `./train.sh`
  - 학습 메인 코드: `./DataBuilder.ipynb`, `./bin/train.py`
  - 테스트 실행 스크립트: `./inference.sh`
  - 테스트 메인 코드: `./bin/predict.py`
  - 테스트 이미지, 마스크 경로: `./Inpainting_Test_Raw_Mask`
  - 테스트 결과 이미지 경로: `./final_result/output_aipg`

## 코드 구조 설명 Architecture
- mmsegmentation을 backend로 사용하여 학습 및 테스트
    - 최종 사용 모델 : mmsegmentation에서 제공하는 xxx 모델
    - custom mIoU loss 추가
    ```
   mmsegmentation/core/models/losses/custom_loss.py
   mmsegmentation/core/models/losses/__init__.py
    ```
    - model에 loss 추가
    ```
   mmsegmentation/core/models/detector/one_stage.py (line 58~59, 72~73, 101~106)
   mmsegmentation/core/models/detector/detector.py (line 97 ~ 105)
-------------------------------------
- 전처리 
    - upsampling : 기존 data 증강
    - cutmix : 
    - 최종 사용 데이터 : 기존 data + upsamping + cutmix

- 모델
    - nvidia/segformer-b4-finetuned-cityscapes-1024-1024로 사용하여 학습 및 테스트함.
    - parameter : (iter 체크한거 첨부하면 좋을 듯. 수정 comment line으로 기재)
    - tuning : (batch, lr, 등..)
    - ensemble : 
    - 최종 사용 모델 : segformer-b4-
    - b4 18000iter lr 0.00005 

- predict : (./config/predict/yaml, 수정하고 predict.ipynb..  순서 정도만 ! (뒤에 방법 기재란 따로 있음))

    ```

- **최종 제출 파일 : submitted.zip**
- **학습된 가중치 파일 : training_results/submitted_model/iter_10000.pth**

## 주요 설치 library 
- huggingface-hub==0.10.1
- torch==1.12.0+cu113
- torchaudio==0.12.0+cu113
- torchvision==0.13.0+cu113
- mmcv-full==1.6.0

# 실행 환경 설정 (How to use)

  - 소스 코드 및 conda 환경 설치 
    ```
    cd.. (본인 저장소 맞추기)
    git clone https://github.com/AIConnect-Army/qualify-test.git
    conda env create -n "envname"
    conda activate "envname"
    pip install -r requirements.txt

    '''
# 학습 실행 방법

  - 학습 데이터 경로 설정
    - `./configs/training/location/my_dataset.yaml` 내의 경로명을 실제 학습 환경에 맞게 수정
      ```
      data_root_dir: /home/user/detection/my_dataset/  # 학습 데이터 절대경로명
      out_root_dir: /home/user/detection/experiments/  # 학습 결과물 절대경로명
      tb_dir: /home/user/detection/tb_logs/  # 학습 로그 절대경로명
      ```

  - 학습 스크립트 실행
    ```
    ./train.sh
    ```
    
  - 학습 스크립트 내용
    ```
    #!/bin/bash

    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    # 첫번째 Model 설정 기반 학습: last_v7__0daee5c4615df5fc17fb1a2f6733dfc1.ckpt, last_v10__dfcb68d46a9604de3147f9ead394f179.ckpt 획득
    python bin/train.py -cn big-lama-aigp-1 location=my_dataset data.batch_size=5

    # 두번째 Model 설정 기반 학습: last_v11__cdb2dc80b605a5e59d234f2721ff80ea.ckpt 획득
    python bin/train.py -cn big-lama-aigp-2 location=my_dataset data.batch_size=5
    ```

# 테스트 실행 방법

  - 테스트 스크립트 실행
    ```
    ./inference.sh
    ```

  - 테스트 스크립트 내용
    ```
    #!/bin/bash

    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    install -d ./final_result/output_aipg

    python bin/predict.py 


    # 상기의 3가지 추론 결과를 Pixel-wise Averaging 처리하여 최종 detection 결과 생성
    python ensemble_avg.py
    (pixel-wise averaging 모든 이미지에 대한 모든 픽셀 의 표준 편차로 나누는 것)
    ```

├── app.py
├── main.py
├── model
│ ├── dataloader.py
│ ├── dataset.py
│ ├── inference.py
│ ├── tag2id.pkl
│ ├── train.py
│ └── utilities.py
├── calculate
│ ├── metric.py
│ └── pickle.py
├── extraction
│ ├── detection.py
│ ├── serialization.py
│ └── valid_info.py
├── preprocess
│ ├── augmentation.py
│ └── cutmix.py
│
├── util
│ ├── log_and_config.py
│ ├── ocr_api.py
│ └── translation.py
└── config
│ ├── predict.yaml
│ └── train.yaml
├── log
│ └── info.log
├── README.md
├── requirements.txt