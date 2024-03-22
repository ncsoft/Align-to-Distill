# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash
base_dir=$path_to_fairseq
export PYTHONPATH="${PYTHONPATH}:$base_dir"
data_dir=$base_dir/data-bin
data=iwslt14.tokenized.de-en
student_model=transformer_student_4heads_A2D
custom_model_dir=$base_dir/custom

mkdir -p $data_dir/$student_model/$data/
touch $data_dir/$student_model/$data/test.log

CUDA_VISIBLE_DEVICES=$GPU_NUM fairseq-generate $data_dir/$data/ \
    --path $data_dir/$student_model/$data/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe --skip-invalid-size-inputs-valid-test \
    --scoring sacrebleu \
    --user-dir ./custom --log-format json 2>&1 | tee $data_dir/$student_model/$data/test.log