import torch
import random

def collate_fn(batch):
    """
    Collate function for the dataloader.
    
    dataset output: first index is batch
    spec1, spec2, label, audio1, audio2
    torch.Size([2, 1, 257, 201]) torch.Size([2, 1, 257, 201]) torch.Size([2, 4]) torch.Size([2, 1, 32000]) torch.Size([2, 1, 32000])
    """
    spec1_time_length_list = [x[0].size(-1) for x in batch]
    spec2_time_length_list = [x[1].size(-1) for x in batch]
    max_time_length1 = max(spec1_time_length_list)
    max_time_length2 = max(spec2_time_length_list)
    max_time_length = max(max_time_length1, max_time_length2)
    
    # 出力作成用のサイズを決定
    batch_size = len(batch)
    freq_bins = batch[0][0].size(1) # (channel, freq_bins, time_steps)
    
    # create output tensor
    spec1_padded = torch.zeros(batch_size, 1, max_time_length, freq_bins)
    spec2_padded = torch.zeros(batch_size, 1, max_time_length, freq_bins)
    time_lengths1 = torch.zeros(batch_size).long()
    time_lengths2 = torch.zeros(batch_size).long()
    label1_padded = torch.zeros(batch_size).long()

    for i, (spec1, spec2, label1, audio1, audio2) in enumerate(batch):
        spec_time_length1 = spec1.size(-1)
        spec1 = spec1.transpose(1, 2)
        spec1_padded[i, :, :spec_time_length1, :] = spec1[:, :spec_time_length1, :]
        time_lengths1[i] = spec_time_length1
        
        spec_time_length2 = spec2.size(-1)
        spec2 = spec2.transpose(1, 2)
        spec2_padded[i, :, :spec_time_length2, :] = spec2[:, :spec_time_length2, :]
        time_lengths2[i] = spec_time_length2
        
        label1_padded[i] = label1
    return (spec1_padded, spec2_padded, time_lengths1, time_lengths2, label1_padded)