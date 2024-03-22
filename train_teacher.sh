# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash
base_dir=$path_to_fairseq
export PYTHONPATH="${PYTHONPATH}:$base_dir"
data_dir=$base_dir/data-bin
data=iwslt14.tokenized.de-en
custom_model_dir=$base_dir/custom/
teacher_model=transformer_teacher

mkdir -p $data_dir/$teacher_model/$data/
touch $data_dir/$teacher_model/$data/train.log

CUDA_VISIBLE_DEVICES=$GPU_NUM fairseq-train $data_dir/$data \
    --arch $teacher_model --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $data_dir/$teacher_model/$data --user-dir $custom_model_dir | tee -a $data_dir/$teacher_model/$data/train.log