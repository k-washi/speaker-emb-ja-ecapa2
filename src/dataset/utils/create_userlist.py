# ディレクトリ内のユーザのリストを作成
# 特定の割合で訓練データとテストデータに分割


from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchaudio

def parser_user_list(
    dataset_dir_list: list[str],
    output_dir: str,
    train_ratio: float=0.975
):
    user_list = []
    for dataset_dir in dataset_dir_list:
        dataset_dir = Path(dataset_dir)
        user_list += list(dataset_dir.glob("*"))
    tmp_user_list = []
    for user_dir in tqdm(user_list):
        wav_list = list(user_dir.glob("wav/*"))
        if len(wav_list) == 0:
            continue
        tmp_user_list.append(user_dir)
    user_list = tmp_user_list
    user_list = sorted(user_list, key=lambda x: x.name)
    print(f"Total users: {len(user_list)}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    otput_userlist_fp = output_dir / "userlist.txt"
    with open(otput_userlist_fp, "w") as f:
        for user in user_list:
            f.write(f"{user}\n")
    
    # split train and test dataset
    train_user_list, test_user_list = train_test_split(
        user_list,
        train_size=train_ratio,
        random_state=42
    )
    output_train_userlist_fp = output_dir / "train_userlist.txt"
    with open(output_train_userlist_fp, "w") as f:
        for user in train_user_list:
            f.write(f"{user}\n")
    output_test_userlist_fp = output_dir / "test_userlist.txt"
    with open(output_test_userlist_fp, "w") as f:
        for user in test_user_list:
            f.write(f"{user}\n")
    print(f"Train users: {len(train_user_list)}")
    print(f"Test users: {len(test_user_list)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_dir", type=str, required=True, nargs="+")
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--train_ratio", type=float, default=0.975)
    
    args = parser.parse_args()
    parser_user_list(
        args.dataset_dir,
        args.output_dir,
        args.train_ratio
    )