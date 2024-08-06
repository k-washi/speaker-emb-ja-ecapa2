from pathlib import Path
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

def summary_dataset_info(
    dataset_dir: str,
    output_dir: str
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) / dataset_dir.stem
    output_dir.mkdir(exist_ok=True, parents=True)
    
    assert dataset_dir.exists(), f"Directory not found: {dataset_dir}"
    audio_fp_list = [s for s in list(dataset_dir.glob("**/*")) if s.is_file() and s.suffix in [".wav", ".flac", ".mp3", ".m4a", ".aac", "ogg"]]
    audio_time_list = []
    for audio_fp in tqdm(audio_fp_list):
        try:
            audio_info = torchaudio.info(str(audio_fp))
        except Exception as e:
            print(f"Error: {e}")
            continue
        audio_time_list.append(audio_info.num_frames / audio_info.sample_rate)
    print(f"Total time: {sum(audio_time_list)} sec")
    print(f"Max time: {max(audio_time_list)} sec")
    print(f"Min time: {min(audio_time_list)} sec")
    print(f"Mean time: {sum(audio_time_list) / len(audio_time_list)} sec")
    print(f"Median time: {sorted(audio_time_list)[len(audio_time_list) // 2]} sec")
    
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Number of audio files: {len(audio_fp_list)}\n")
        f.write(f"Total time: {sum(audio_time_list)} sec\n")
        f.write(f"Total time hours: {sum(audio_time_list) / 3600} hours\n")
        f.write(f"Max time: {max(audio_time_list)} sec\n")
        f.write(f"Min time: {min(audio_time_list)} sec\n")
        f.write(f"Mean time: {sum(audio_time_list) / len(audio_time_list)} sec\n")
        f.write(f"Median time: {sorted(audio_time_list)[len(audio_time_list) // 2]} sec\n")
    
    plt.hist(audio_time_list, bins=100)
    plt.xlabel("Time [sec]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "audio_time_hist.png")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()
    summary_dataset_info(args.dataset_dir, args.output_dir)