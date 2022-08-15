# Desc

- Classification 모델 데이터 생성 및 모델 학습

# Prerequsition

- Python 3.9 이상
- 아래 명령어로 dependency 설치
  ```
  >>> pip install -r requirements.txt
  ```

# Modules

- video2images: 비디오 파일에서 프레임단위로 이미지 추출
- concatsubdirs: 카테고리별로 분류된 사진들을 하나의 폴더에 모으고, 모인 이미지들에 대해 레이블 생성
- dataset2tfrecord: Dataset(이미지들, 레이블 파일)을 TFRecord 파일로 변환
- train: MobileNet에 기반하여 classification 모델 학습 및 savedmodel 추출
- inference: 학습된 모델 테스트
- savedmodel2tfjs: 학습된 모델을 tfjs로 추출. 웹브라우저에서 이용
- savedmodel2tflite: 학습된 모델을 tflite로 추출. Edge device에서 이용

# Steps

아래의 3단계로 구성

1. 데이터 준비
2. 학습 및 모델추출
3. 모델변환

# How To Use

## 1. 데이터 준비

1. 비디오 파일에서 종류별로 각각의 폴더에 이미지들을 추출
2. 각 폴더내 이미지들을 하나의 폴더에 합하면서, 각 이미지들의 레이블을 나타내는 파일 생성
3. 이미지들과 레이블 파일을 이용하여 TFRecord 파일들 생성

### 1-1. Video to Images

- 비디오 파일에서 프레임단위로 이미지 추출. 아래의 명령어로 실행.
  ```Python
  >>> python -m scripts.video2images dataset/myvideo.mp4 -o dataset/myimages
  >>> python -m scripts.video2images myvideo.mp4 -o my_images
  >>> python -m scripts.video2images myvideo.mp4 -o my_images --add  # add images to the folder
  ```

### 1-2. Make Dataset

- 카테고리별로 분류된 사진들을 하나의 폴더에 모으고, 모인 이미지들에 대한 레이블 csv 파일생성
- 아래의 명령어로 실행
  ```Python
  >>> python -m scripts.concatsubdirs dataset/rawmeter -o dataset/meter \
      --labelfname meter --imgfnamelength 3
  >>> python -m scripts.cocatsubdirs imgrootdir
  >>> python -m scripts.cocatsubdirs imgrootdir -o dataset
  ```

### 1-3. Make TFRecords

- Dataset(이미지들, 레이블 파일)을 TFRecord 파일로 변환
- 아래의 명령어로 실행
  ```Python
  >>> python -m scripts.dataset2tfrecord \
      -o dataset -i dataset/meter/images -l dataset/meter/meter.csv \
      -n mydata --valid --test
  >>> python -m scripts.dataset2tfrecord -o dataset -i train -l label.csv
  >>> python -m scripts.dataset2tfrecord -o dataset -i train -l label.csv
  ```

---

## 2. 학습 및 모델추출

1. TFRecord 파일들을 이용하여 모델 학습 및 saved model 추출
2. 테스트 이미지를 이용하여 결과 확인

### 2-1. Train & Export SavedModel

- `models` 폴더 밑의 하위폴더 `ckpt`와 `export`에 설정한 이름으로 학습된 모델 파일 저장
- 아래의 명령어로 학습 및 파일 생성
  ```Python
  >>> python -m scripts.train ./dataset/mydata.train.tfrecord
  >>> python -m scripts.train ./dataset/mydata.train.tfrecord \
      -n mymodel -v ./dataset/mydata.valid.tfrecord \
      -b 16 -e 1
  >>> python -m scripts.train ./dataset/my.train.tfrecord \
      -n meter -v ./dataset/my.valid.tfrecord --ckpt ./models/ckpt/my_model
  >>> python -m scripts.train ./dataset/my.train.tfrecord \
      -n meter -v ./dataset/my.valid.tfrecord --ckpt ./models/ckpt/my_model \
      -b 1024 -e 10
  ```

## 2-2. Inference

```Python
>>> python -m scripts.inference ./models/export/mymodel -i ./images/1.png ./images/2.png
```

---

## 3. 모델변환

1. Javascript에서 사용가능하도록 모델 변환
2. Edge Device(Android, IOS)에서 사용가능하도록 모델 변환

### 3-1. TFjs Export

- 학습된 모델을 tfjs로 추출
- 웹브라우저에서 이용
- 아래 명령어로 추출
  ```Python
  >>> python -m scripts.savedmodel2tfjs ./models/export/mymodel -o ./models/tfjs
  ```

### 3-2. TFLite Export

- 학습된 모델을 tflite로 추출
- Edge device에서 이용
- 아래 명령어로 추출
  ```Python
  >>> python -m scripts.savedmodel2tflite ./models/export/mymodel -o ./models/tflite
  ```
