# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture

@register_model_architecture("transformer", "transformer_wmt_en_de_teacher")
def transformer_wmt_en_de_alignment_module(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    return base_architecture(args)

@register_model_architecture("transformer", "transformer_wmt_en_de_student")
def transformer_wmt_en_de_alignment_module(args):
    args.alignment_module = getattr(args, "alignment_module", True)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    return base_architecture(args)

@register_model_architecture("transformer", "transformer_teacher")
def transformer_alignment_module(args): 
    args.alignment_module = getattr(args, "alignment_module", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

@register_model_architecture("transformer", "transformer_student_4heads")
def transformer_student_4heads(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)

@register_model_architecture("transformer", "transformer_student_4heads_A2D")
def transformer_student_4heads(args):
    args.alignment_module = getattr(args, "alignment_module", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)

@register_model_architecture("transformer", "transformer_student_8heads")
def transformer_student_8heads(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)

@register_model_architecture("transformer", "transformer_student_8heads_A2D")
def transformer_student_8heads(args):
    args.alignment_module = getattr(args, "alignment_module", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)