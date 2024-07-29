# データセットの構成

データセットは、以下のような構成になっていることを想定しています。
transcriptionには、漢字仮名交じり文が記載されています。（whisper-v3の結果など）

```bash
- dataset
    - user1
        - wav
            - 1.wav
            - 2.wav
            - ...
        - transcription
            - 1.txt
            - 2.txt
            - ...
    - user2
        - wav
            - 1.wav
            - 2.wav
            - ...
        - transcription
            - 1.txt
            - 2.txt
            - ...

```

# ユーザーリストの作成

```bash
python ./src/dataset/utils/create_userlist.py --dataset_dir sv_dataset1/  --output_dir data/userlist/ --train_ratio 0.975
```


# 訓練

```bash
python ./src/experiments/ecapa2/exp00001/train00001.py
```