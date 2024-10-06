import torch
import numpy as np
from scipy.io import wavfile
from torchaudio.compliance import kaldi
from tqdm import tqdm
from pathlib import Path

import torch.nn.functional as F

from src.utils.audio import load_wave
from src.experiments.utils.utils import get_audiofp_and_label_list_from_userlist_file
from src.metrics.metrics import accuracy, ComputeErrorRates, ComputeMinDcf
from src.metrics.utils import tuneThresholdfromScore
from src.experiments.utils.plot_emb import umap_show

import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.metric_stats import EER, minDCF

def test_jtubexvector(
    test_userlist_fp: str,
    output_dir: str = "data/test/jtubexvector"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    audiofp_list, label_list = get_audiofp_and_label_list_from_userlist_file(test_userlist_fp)
    label_map = {label: i for i, label in enumerate(set(label_list))}
    
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    embedding_dict = {}
    for audio_fp, label in tqdm(zip(audiofp_list, label_list), total=len(audiofp_list)):
        if label not in embedding_dict:
            embedding_dict[label] = []
        wav, _ = load_wave(audio_fp, sample_rate=16000, is_torch=True, mono=True)
        xvecter = model.encode_batch(wav)
        xvecter = xvecter.squeeze(0).squeeze(0).detach().cpu().numpy()
        embedding_dict[label].append(xvecter)
    
    embedding_list = []
    label_umap_list = []
    for label in embedding_dict.keys():
        label = label_map[label]
        embedding_list.extend(embedding_dict[label])
        label_umap_list.extend([label] * len(embedding_dict[label]))
    #umap_show(embedding_list, label_umap_list, output_dir / "umap.png")
    
    speaker_id_list = sorted(list(embedding_dict.keys()))
    # スコア計算
    score_list, label_list = [], []
    # 同じ話者のspeaker embeddingをまとめる
    embedding_error_num = 0
    score_error_num = 0
    all_data_num = 0
    cos_sim = torch.nn.CosineSimilarity()
    for label_idx in tqdm(speaker_id_list, desc="Calc same speaker score", total=len(speaker_id_list)):
        embedding_list = embedding_dict[label_idx]
        for i, (emb1, emb2) in enumerate(zip(embedding_list[0::2], embedding_list[1::2])):
            emb1 = torch.from_numpy(emb1)
            emb2 = torch.from_numpy(emb2)
            emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(0)
            print(emb1.shape, emb2.shape)
            #emb1 = F.normalize(emb1)
            #emb2 = F.normalize(emb2)
            all_data_num += 1
            if not (emb1.abs().max() >= 0 and emb2.abs().max() >= 0):
                embedding_error_num += 1
            score = cos_sim(emb1, emb2)
            if not score.abs().max() >= 0:
                score_error_num += 1
                score = torch.tensor([1.0])
            score_list.append(score.item())
            label_list.append(1)
    if embedding_error_num > 0:
        print(f"Embedding error: {embedding_error_num}/{all_data_num}")
    if score_error_num > 0:
        print(f"Score error: {score_error_num}/{all_data_num}")
    score_mean = np.mean(score_list)
    print(f"Score mean: {score_mean}")
    
    # 異なる話者のspeaker embeddingを比較
    diff_score_list, diff_label_list = [], []
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
            score = cos_sim(emb1, emb2)
            if not score.abs().max() >= 0:
                score = torch.tensor([1.0])
            
            diff_score_list.append(score.item())
            diff_label_list.append(0)
    score_mean = np.mean(diff_score_list)
    print(f"Diff Score mean: {score_mean}")
    
    eer, _ = EER(torch.FloatTensor(score_list), torch.FloatTensor(diff_score_list))
    if np.isnan(eer):
        eer = 1.
    eer *= 100
    print(eer)
    
    min_dcf, _ = minDCF(torch.FloatTensor(score_list), torch.FloatTensor(diff_score_list))
    if np.isnan(min_dcf):
        min_dcf = 1.
   
    print(f"EER: {eer}, minDCF: {min_dcf}")
    with open(output_dir / "result.txt", "w") as f:
        f.write(f"EER: {eer}, minDCF: {minDCF}")
    

test_jtubexvector(
    test_userlist_fp = "data/users/test_userlist.txt",
    output_dir = "data/test/speechbrain"
)