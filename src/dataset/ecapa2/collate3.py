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
    
    # 出力作成用のサイズを決定
    batch_size = len(batch)
    freq_bins = batch[0][0].size(1) # (channel, freq_bins, time_steps)
    
    # create output tensor
    spec_padded = torch.zeros(batch_size, 1, freq_bins, max_time_length)
    time_lengths = torch.zeros(batch_size).long()
    label1_padded = torch.zeros(batch_size).long()

    for i, (spec, label1, label2, mixup_lambda, _) in enumerate(batch):
        spec_time_length = spec.size(-1)
        spec_padded[i, :, :, :spec_time_length] = spec[:, :, :spec_time_length]
        time_lengths[i] = spec_time_length
        label1_padded[i] = label1
    return (spec_padded, time_lengths, label1_padded)