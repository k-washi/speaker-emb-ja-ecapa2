import torch
import numpy as np
from scipy.io import wavfile
from torchaudio.compliance import kaldi
from tqdm import tqdm
from pathlib import Path

import itertools
import random
from src.utils.audio import load_wave
from src.experiments.utils.utils import get_audiofp_and_label_list_from_userlist_file
from speechbrain.utils.metric_stats import EER, minDCF
from src.metrics.utils import tuneThresholdfromScore
from src.experiments.utils.plot_emb import umap_show

random.seed(0)

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
	voc_dir: str,
	output_dir: str = "data/test_voc/jtubexvector"
):
	output_dir = Path(output_dir)
	output_dir.mkdir(exist_ok=True, parents=True)
	user_list = sorted(list(Path(voc_dir).glob("*")))
	print(f"Found {len(user_list)} users")
	audiofp_list, label_list = [], []
	for label_index, user_fp in enumerate(user_list):
		audio_dir = user_fp / "wav"
		_audio_fp_list = list(audio_dir.glob("*.wav"))
		audiofp_list.extend(_audio_fp_list)
		label_list.extend([label_index] * len(_audio_fp_list))
		print(f"Found {len(_audio_fp_list)} audio files for {user_fp}")

	model = torch.hub.load("sarulab-speech/xvector_jtubespeech", "xvector", trust_repo=True)
    
	embedding_dict = {}
	for audio_fp, label in tqdm(zip(audiofp_list, label_list), total=len(audiofp_list)):
		if label not in embedding_dict:
			embedding_dict[label] = []
		wav, _ = load_wave(audio_fp, sample_rate=16000, is_torch=False, mono=True)
		xvecter = extract_xvector(model, wav)
		embedding_dict[label].append(xvecter)
	label_map = {label: i for i, label in enumerate(set(label_list))}

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
	diff_score_list = []
	diff_label_list = []
    # 同じ話者のspeaker embeddingをまとめる
	embedding_error_num = 0
	score_error_num = 0
	all_data_num = 0
    
	for label_idx in tqdm(speaker_id_list, desc="Calc same speaker score", total=len(speaker_id_list)):
		embedding_list = embedding_dict[label_idx]
		for i, (emb1, emb2) in enumerate(itertools.combinations(embedding_list, 2)):
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
		for i in range(len(embedding_list)):
			emb1 = embedding_list[i]
			other_speaker_id = random.choice(list(set(speaker_id_list) - set([label_idx])))
			emb2 = random.choice(embedding_dict[other_speaker_id])
			emb1 = torch.from_numpy(emb1)
			emb2 = torch.from_numpy(emb2)
			emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(0)
			score = torch.mean(torch.matmul(emb1, emb2.T))
			if not score.abs().max() >= 0:
				score = torch.tensor([1.0])
			diff_score_list.append(score.item())
			diff_label_list.append(0)
	if embedding_error_num > 0:
		print(f"Embedding error: {embedding_error_num}/{all_data_num}")
	if score_error_num > 0:
		print(f"Score error: {score_error_num}/{all_data_num}")
	# 異なる話者のspeaker embeddingを比較
	
	eer, _ = EER(torch.FloatTensor(score_list), torch.FloatTensor(diff_score_list[:len(score_list)]))
	min_dcf, _ = minDCF(torch.FloatTensor(score_list), torch.FloatTensor(diff_score_list[:len(score_list)]))
	print(f"EER: {eer}, minDCF: {min_dcf}")
	with open(output_dir / "result.txt", "w") as f:
		f.write(f"EER: {eer}, minDCF: {min_dcf}")
if __name__ == "__main__":
	test_jtubexvector(
		voc_dir="/data/voxceleb2_test"
	)