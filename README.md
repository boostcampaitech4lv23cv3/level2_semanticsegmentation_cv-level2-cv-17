# 🌠 Sixth Sense 팀의 Semantic Segmentation 프로젝트

쓰레기가 찍힌 사진에서 쓰레기를 Segmentation 하는 모델을 통한 분리수거 인공지능 만들기 대회

> 기간 : 2022.12.21 ~ 2023.01.05

![부스트 캠프 AI Tech 4기](https://img.shields.io/badge/%EB%B6%80%EC%8A%A4%ED%8A%B8%EC%BA%A0%ED%94%84%20AI%20Tech-4%EA%B8%B0-red)
![Level 2](https://img.shields.io/badge/Level-2-yellow)
![CV 17조](https://img.shields.io/badge/CV-17%EC%A1%B0-brightgreen)
![Semantic Segmentation 대회](https://img.shields.io/badge/%EB%8C%80%ED%9A%8C-Semantic%20Segmentation-blue)

![stages ai_competitions_227_overview_description](https://user-images.githubusercontent.com/9074297/208830640-df24aeaa-fc33-40ee-8756-1beca7a5f678.png)


## 😎 Members

<table>
    <thead>
        <tr>
            <th>박선규</th>
            <th>박세준</th>
            <th>서장원</th>
            <th>이광민</th>
            <th>장국빈</th>
            <th>조태환</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td width="16%"><img src="https://user-images.githubusercontent.com/9074297/207550543-a4a35f97-c647-4013-b440-dbfec61b01d7.png" width="100%"/></td>
            <td width="16%"><img src="https://user-images.githubusercontent.com/9074297/207550381-3f2deddb-ffef-4249-8738-66d27c83ea79.png" width="100%"/></td>
            <td width="16%"><img src="https://user-images.githubusercontent.com/9074297/207550023-28ad4754-e60b-4a0c-835e-ea3c32108703.png" width="100%"/></td>
            <td width="16%"><img src="https://user-images.githubusercontent.com/9074297/207551768-ca68e744-70bf-452d-bd61-f4db912e59ee.png" width="100%"/></td>
            <td width="16%"><img src="https://user-images.githubusercontent.com/9074297/207583484-e4cff046-7656-4c27-90c9-0ce116418e70.png" width="100%"/></td>
            <td width="16%"><img src="https://user-images.githubusercontent.com/9074297/207550298-4dd75fe8-137d-4bad-accf-d56363c01895.png" width="100%"/></td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/Sungyu-Park"><sub><sup>@Sungyu-Park</sup></sub></a></td>
            <td align="center"><a href="https://github.com/sjleo1"><sub><sup>@sjleo1</sup></sub></a></td>
            <td align="center"><a href="https://github.com/nanpuhaha"><sub><sup>@nanpuhaha</sup></sub></a></td>
            <td align="center"><a href="https://github.com/lkm6871"><sub><sup>@lkm6871</sup></sub></a></td>
            <td align="center"><a href="https://github.com/JKbin"><sub><sup>@JKbin</sup></sub></a></td>
            <td align="center"><a href="https://github.com/OMMANT"><sub><sup>@OMMANT</sup></sub></a></td>
        </tr>
    </tbody>
</table>

## 🧑‍💻 Contributions

- 박선규 :
- 박세준 :
- 서장원 :
- 이광민 :
- 장국빈 :
- 조태환 :

<br>

## :earth_asia: Project Overview

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

<img src="https://s3-ap-northeast-2.amazonaws.com/prod-aistages-public/app/Users/00000274/files/7645ad37-9853-4a85-b0a8-f0f151ef05be..png" height="200px"/>

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Segmentation하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 배경, 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요!

<br>

- Input : 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 COCO format으로 제공됩니다.
- Output : 모델은 pixel 좌표에 따라 카테고리 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다.

<br>

## 🚨 Competition Rules

<br>

## 💾 Dataset

### 기본 제공된 데이터셋

-

### 추가로 활용한 외부 데이터셋

-

<br>

## 💻 Develop Environment

- OS : Ubuntu 18.04.5 LTS (bionic)
- CPU : Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz (8 Cores, 8 Threads)
- RAM : 90GB
- GPU : Tesla V100-PCIE-32GB (NVIDIA : 450.80.02)
- Storage : 100GB
- Python : 3.8.5 (conda)
- CUDA : 11.0.221
- PyTorch : 1.7.1
- Segmentation Models PyTorch : 0.3.1
- MMSegmentation : 0.29.1 ~~1.0.0rc2~~

<br>

## 📂 Project Structure

<details>
<summary>프로젝트 구조 살펴보기</summary>

```bash
.
├──📁input
│   ├──📁data
│   │   ├──📁batch_01_vt
│   │   │   ├──🖼️0002.jpg
│   │   │   └──🖼️...
│   │   ├──📁batch_02_vt
│   │   │   ├──🖼️0001.jpg
│   │   │   └──🖼️...
│   │   ├──📁batch_03
│   │   │   ├──🖼️0001.jpg
│   │   │   └──🖼️...
│   │   ├──📄test.json
│   │   ├──📄train.json
│   │   ├──📄val.json
│   └──📁mmseg
│       └──📁trash
│           ├──📁ann_dir
│           │   ├──📁train
│           │   │   ├──🖼️0001.png
│           │   │   ├──🖼️0001_color.png
│           │   │   └──🖼️...
│           │   └──📁val
│           │       ├──🖼️0001.png
│           │       ├──🖼️0001_color.png
│           │       └──🖼️...
│           └──📁img_dir
│               ├──📁test
│               │   ├──🖼️0000.jpg
│               │   └──🖼️...
│               ├──📁train
│               │   ├──🖼️0000.jpg
│               │   └──🖼️...
│               └──📁val
│                   ├──🖼️0000.jpg
│                   └──🖼️...
└──📁level2_semanticsegmentation_cv-level2-cv-17
    ├──📁mmsegmentation
    ├──📁src
    └──📄environment.yml
```

</details>

<br>

## 👨‍🏫 Evaluation Methods

Semantic Segmentation에서 사용되는 대표적인 성능 측정 방법인 mIoU로 평가합니다.

<details>
<summary>평가 방법 자세히 살펴보기</summary>

### Test set의 mIoU (Mean Intersection over Union)로 평가

- Semantic Segmentation에서 사용되는 대표적인 성능 측정 방법
- IoU
  $$
  \mathrm{IoU}=\frac{|X \cap Y|}{|X \cup Y|}=\frac{|X \cap Y|}{|X|+|Y|-|X \cap Y|}
  $$

### Example of IoU

![image](https://user-images.githubusercontent.com/9074297/208380726-3bf69d83-4ec6-4e6b-bf94-994a51e5be75.png)

![image](https://user-images.githubusercontent.com/9074297/208380793-39cf224d-dcb3-4472-afd5-8c983b69e28a.png)

### [참고사항]

model로부터 예측된 mask의 size는 512 x 512 지만, 대회의 원활한 운영을 위해 output을 일괄적으로 256 x 256 으로 변경하여 score를 반영하게 되었습니다.

### 제출 방법

1. 베이스라인 코드 실행
2. submission.csv 제출

</details>

<br>

## 👀 How to Start

```bash
# clone repository
git clone https://github.com/boostcampaitech4lv23cv3/level2_semanticsegmentation_cv-level2-cv-17.git

# change directory
cd level2_semanticsegmentation_cv-level2-cv-17

# create conda environment
conda env create -f environment.yml

# activate conda environment
conda activate "segmentation"

# install mmcv-full using mim or pip
mim install mmcv-full==1.7.0
# pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html

# install mmsegmentation as editable mode
pip install -e mmsegmentation

# install pre-commit hook
pre-commit install
```
