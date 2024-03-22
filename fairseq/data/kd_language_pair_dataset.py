# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from fairseq.data import LanguagePairDataset

class KDLanguagePairDataset(LanguagePairDataset):
    def __getitem__(self, index):
        example = super().__getitem__(index)
        src_tokens, src_lengths = example['net_input']['src_tokens'], example['net_input']['src_lengths']
        tgt_tokens = example['target']
        
        # Load attention weights from file
        attention_weights = torch.load(f'attention_weights{index}.pt')
        
        # Return input sentence and attention weights
        return {
            'id': index,
            'source': src_tokens,
            'target': tgt_tokens,
            'attention_weights': attention_weights,
            'source_lengths': src_lengths,
        }