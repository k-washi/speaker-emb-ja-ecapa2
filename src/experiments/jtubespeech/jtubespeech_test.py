import torch
import numpy as np
from scipy.io import wavfile
from torchaudio.compliance import kaldi
from tqdm import tqdm
from pathlib import Path

from src.utils.audio import load_wave
from src.experiments.utils.utils import get_audiofp_and_label_list_from_userlist_file
from src.metrics.metrics import accuracy, ComputeErrorRates, ComputeMinDcf
from src.metrics.utils import tuneThresholdfromScore
from src.experiments.utils.plot_emb import umap_show


def extract_xvector(
  model, # xvector model
  wav   # 16kHz mono
):
  # extract mfcc
  wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
  mfcc = kaldi.mfcc(wav, num_ceps=24, num_mel_bins=24) # [1, T, 24]
  mfcc = mfcc.unsqueeze(0)

  # extract xvector
  xvector = model.vectorize(mfcc) # (1, 512)
  xvector = xvector.to("cpu").detach().numpy().copy()[0]  

  return xvector

def test_jtubexvector(
    test_userlist_fp: str,
    output_dir: str = "data/test/jtubexvector"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    audiofp_list, label_list = get_audiofp_and_label_list_from_userlist_file(test_userlist_fp)
    label_map = {label: i for i, label in enumerate(set(label_list))}
    
    model = torch.hub.load("sarulab-speech/xvector_jtubespeech", "xvector", trust_repo=True)
    
    
    embedding_dict = {}
    for audio_fp, label in tqdm(zip(audiofp_list, label_list), total=len(audiofp_list)):
        if label not in embedding_dict:
            embedding_dict[label] = []
        wav, _ = load_wave(audio_fp, sample_rate=16000, is_torch=False, mono=True)
        xvecter = extract_xvector(model, wav)
        embedding_dict[label].append(xvecter)
    
    embedding_list = []
    label_umap_list = []
    for label in embedding_dict.keys():
        label = label_map[label]
        embedding_list.extend(embedding_dict[label])
        label_umap_list.extend([label] * len(embedding_dict[label]))
    umap_show(embedding_list, label_umap_list, output_dir / "umap.png")
    
    speaker_id_list = sorted(list(embedding_dict.keys()))
    # スコア計算
    score_list, label_list = [], []
    # 同じ話者のspeaker embeddingをまとめる
    embedding_error_num = 0
    score_error_num = 0
    all_data_num = 0
    
    for label_idx in tqdm(speaker_id_list, desc="Calc same speaker score", total=len(speaker_id_list)):
        embedding_list = embedding_dict[label_idx]
        for i, (emb1, emb2) in enumerate(zip(embedding_list[0::2], embedding_list[1::2])):
            emb1 = torch.from_numpy(emb1)
            emb2 = torch.from_numpy(emb2)
            emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(0)
            all_data_num += 1
            if not (emb1.abs().max() >= 0 and emb2.abs().max() >= 0):
                embedding_error_num += 1
            score = torch.mean(torch.matmul(emb1, emb2.mT))
            if not score.abs().max() >= 0:
                score_error_num += 1
                score = torch.tensor([1.0])
            score_list.append(score.item())
            label_list.append(1)
    if embedding_error_num > 0:
        print(f"Embedding error: {embedding_error_num}/{all_data_num}")
    if score_error_num > 0:
        print(f"Score error: {score_error_num}/{all_data_num}")
    # 異なる話者のspeaker embeddingを比較
    for speaker_id1, speaker_id2 in tqdm(zip(speaker_id_list[0::2], speaker_id_list[1::2]),
                                         desc="Calc diff speaker score",
                                         total=len(speaker_id_list)//2
                                        ):
        spkemb_list1 = embedding_dict[speaker_id1]
        spkemb_list2 = embedding_dict[speaker_id2]
        for i, (emb1, emb2) in enumerate(zip(spkemb_list1, spkemb_list2)):
            emb1 = torch.from_numpy(emb1)
            emb2 = torch.from_numpy(emb2)
            emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(0)
            score = torch.mean(torch.matmul(emb1, emb2.T))
            if not score.abs().max() >= 0:
                score = torch.tensor([1.0])
            
            score_list.append(score.item())
            label_list.append(0)
    score_list = np.array(score_list)
    label_list = np.array(label_list)
    try:
        eer = tuneThresholdfromScore(score_list, label_list, [1, 0.1])[1]
    except Exception as e:
        print(f"Error in tuneThresholdfromScore: {e}")
        eer = 1.0
    try:
        fnrs, fprs, thresholds = ComputeErrorRates(score_list, label_list)
        minDCF, _  = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    except Exception as e:
        print(f"Error in ComputeErrorRates: {e}")
        minDCF = 1.0
    print(f"EER: {eer}, minDCF: {minDCF}")
    with open(output_dir / "result.txt", "w") as f:
        f.write(f"EER: {eer}, minDCF: {minDCF}")
    

test_jtubexvector(
    test_userlist_fp = "data/users/test_userlist.txt",
    output_dir = "data/test/jtubexvector"
)