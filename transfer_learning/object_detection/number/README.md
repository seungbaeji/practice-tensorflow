# Desc
- SVHN (Street View House Numbers) Dataset으로 학습된 object detection 모델을 이용하여 number detection (ref. 1)
- object detection 모델은 mobile net이 사용 (ref. 2)
- 결과값은 숫자에 해당하는 id로 반환. 0을 제외한 숫자들은 id와 숫자값이 동일
- id 10은 숫자 0을 나타냄. id 0은 Tensorflow 내에서 사용하는 값

# Prerequsition
- Python 3.9 이상
- 아래 명령어로 dependency 설치
    ```
    >>> pip install -r requirements.txt
    ```

# Modules
- odjson2pbtxt: object detection 모델 양식에 맞는 `label.json` 파일을 읽어서 `label.pbtxt` 파일 생성
- dataset2tfrecord: Dataset(images, annotations, label.pbtxt)을 TFRecord 파일로 변환
- train: pre-trained 모델에 기반하여, TFRecord 파일로 number detection 모델 학습
- inference: 학습된 모델 테스트
- freezemodel: 학습된 모델을 saved model로 추출
- savedmodel2tfjs: 학습된 모델을 tfjs로 추출. 웹브라우저에서 이용
- savedmodel2tflite: 학습된 모델을 tflite로 추출. Edge device에서 이용

# Steps
아래의 3단계로 구성되며, 순서대로 실행한다
1. 데이터 준비
2. 학습 및 모델추출
3. 모델변환

# How To Use

## 1. 데이터 준비
- label.pbtxt 파일 생성
- images, annotations, and `label.pbtxt` 파일을 이용하여 TFRecord 파일 생성

### 1-1. Make Label pbtxt
- Tensorflow object detection 모델에서 사용하는 label 파일 생성 
- object detection 모델의 양식을 따르는 json파일을 입력으로 받음
- json 파일은 아래의 양식을 따름
    ```json
    [
        {
            "name": "1",
            "id": 1
        },
        {
            "name": "2",
            "id": 2
        },
        ...
        {
            "name": "10",
            "id": 10
        }
    ]
    ```
- Tensorflow object detection 모델에서 사용하는 `label.pbtxt` 파일은 아래의 명령어로 생성.
    ```Python
    >>> python -m scripts.odjson2pbtxt data/label.json -o data
    ```

### 1-2. Make TFRecord
- image 파일들과 annotation 파일들을 이용하여 TFRecord 파일 생성
- image들 annotation들은 하나의 폴더에 있어야 함
- train과 test는 각각 만들어 줘야 함
- 아래의 명령어로 train dataset 생성
    ```Python
    >>> python -m scripts.dataset2tfrecord \
        -x data/trainset \
        -l data/label_map.pbtxt \
        -o data/train.tfrecord
    ```
- 아래의 명령어로 test dataset 생성
    ```Python
    >>> python -m scripts.dataset2tfrecord \
        -x data/testset \
        -l data/label_map.pbtxt \
        -o data/test.tfrecord
    ```

--------------

## 2. 학습 및 모델추출
- TFRecord 파일들을 이용하여 모델 학습
- 학습된 모델을 saved model로 추출
- 테스트 이미지를 이용하여 결과 확인

### 2-1. Train Model
- 학습된 모델에 기반하여 새로운 데이터셋으로 모델 학습
- `ref.3`를 참고하여, 모델 폴더내 `pipeline.config`파일의 data augumentation 옵션 조절
- 아래의 명령어로 모델학습
    ```Python
    >>> python -m scripts.train models/pretrained/housenum \
        -l data/label_map.pbtxt -train data/train.tfrecord -o models/my_number
    >>> python -m scripts.train models/pretrained/housenum \
        -l data/label_map.pbtxt -train data/train.tfrecord -o models/my_number \
        -test data/test.tfrecord -b 2 -e 2
    ```

### 2-2. Export Saved Model
- 학습된 모델을 saved model로 추출
- 아래의 명령어로 모델 추출
    ```Python
    >>> python -m scripts.freezemodel models/my_number -o export
    ```

### 2-3. Inference
```Python
>>> python -m scripts.inference export/first_number/saved_model \
    -i data/samples/1.png data/samples/2.png data/samples/3.png \
    -o data
```
--------------

## 3. 모델변환
1. Javascript에서 사용가능하도록 모델 변환
2. Edge Device(Android, IOS)에서 사용가능하도록 모델 변환

### 3-1. TFjs Export
- 추출된 saved model을 tfjs로 변환
- 웹브라우저에서 이용
- 아래 명령어로 추출
    ```Python
    >>> python -m scripts.savedmodel2tfjs export/my_number/saved_model \
        -o export/my_number/tfjs
    ```

### 3-2. TFLite Export
- 추출된 saved model을 tflite로 변환
- Edge device에서 이용
- 아래 명령어로 추출
    ```Python
    >>> python -m scripts.savedmodel2tflite export/my_number/saved_model \
        -e export/my_number/my_number.tflite
    ```

# Reference
1. http://ufldl.stanford.edu/housenumbers/
2. https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1
3. https://github.com/tensorflow/models/blob/master/research/object_detection/builders/preprocessor_builder_test.py
