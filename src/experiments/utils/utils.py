import pandas as pd
from pathlib import Path

from pathlib import Path

def get_audiofp_and_label_list_from_userlist_file(userlist_fp: str):
    """
    Retrieves a list of audio file paths and corresponding labels from a userlist file.

    Args:
        userlist_fp (str): The file path to the userlist file.

    Returns:
        audio_fp_list (list): A list of audio file paths.
        label_list (list): A list of corresponding labels.

    Raises:
        AssertionError: If the userlist file does not exist or if a user directory does not exist.

    """
    userlist_fp = Path(userlist_fp)
    assert userlist_fp.exists(), f"File not found: {userlist_fp}"
    
    user_list = []
    with open(userlist_fp, "r") as f:
        for line in f:
            user_dir = Path(line.strip())
            assert user_dir.exists(), f"Directory not found: {user_dir}"
            user_list.append(user_dir)
    audio_fp_list, label_list = [], []
    for i, user_dir in enumerate(user_list):
        _audio_fp_list = list((user_dir / "wav").glob("*"))
        assert len(_audio_fp_list) > 0, f"No audio found in {user_dir}"
        audio_fp_list += _audio_fp_list
        label_list += [i] * len(_audio_fp_list)
    label_num = len(set(label_list))
    print(f"Total audio: {len(audio_fp_list)}")
    assert label_num == max(label_list) + 1, f"Label num mismatch: {label_num} != {max(label_list) + 1}"
    return audio_fp_list, label_list