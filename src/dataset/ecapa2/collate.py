import torch
import random

def collate_fn(batch):
    """
    Collate function for the dataloader.
    
    dataset output: first index is batch
    mixed_spec, label1, label2, mixup_lambda, (audio1, audio2)
    torch.Size([2, 1, 257, 201]) torch.Size([2, 4]) torch.Size([2, 1]) torch.Size([2, 1, 32000]) torch.Size([2, 1, 32000])
    """
    spec_time_length_list = [x[0].size(-1) for x in batch]
    max_time_length = max(spec_time_length_list)
    min_time_length = min(spec_time_length_list)
    
    # 出力作成用のサイズを決定
    batch_size = len(batch)
    freq_bins = batch[0][0].size(1) # (channel, freq_bins, time_steps)
    if min_time_length == max_time_length:
        time_length = max_time_length
    else:
        random_value_in_range = random.randint(min_time_length, max_time_length)
        time_length = random.choice([min_time_length, max_time_length+1, random_value_in_range])
    assert time_length > 0, f"Time length should be greater than 0, but got {time_length}"
    
    # create output tensor
    spec_padded = torch.zeros(batch_size, 1, freq_bins, time_length)
    label1_padded = torch.zeros(batch_size)
    label2_padded = torch.zeros(batch_size)
    mixup_lambda_padded = torch.zeros(batch_size)
    
    for i, (spec, label1, label2, mixup_lambda, _) in enumerate(batch):
        spec_time_length = spec.size(-1)
        if spec_time_length >= time_length:
            spec_padded[i, :, :, :time_length] = spec[:, :, :time_length]
        else:
            spec_padded[i, :, :, :spec_time_length] = spec[:, :, :spec_time_length]
        label1_padded[i] = label1
        label2_padded[i] = label2
        mixup_lambda_padded[i] = mixup_lambda
    return (spec_padded, label1_padded, label2_padded, mixup_lambda_padded)