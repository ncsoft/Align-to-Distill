# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from typing import Optional
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from einops import rearrange

@dataclass
class KDLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    decoder_kd: bool = field(
        default=False, metadata={"help": "decoder attention distillation"}
    )
    self_kd: bool = field(
        default=True, metadata={"help": "decoder attention distillation"}
    )
    cross_kd: bool = field(
        default=True, metadata={"help": "decoder attention distillation"}
    )
    beta: float = field(
        default=1,
        metadata={"help": "attn_loss weight"},
    )
    decay: float = field(
        default=0.9,
        metadata={"help": "decay value for beta"}
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    student_temp: float = field(
        default=1,
        metadata={"help": "student model temperature for distillation"}
    )
    teacher_temp: float = field(
        default=1,
        metadata={"help": "teacher model emperature for distillation"}
    )
    alpha: Optional[float] = field(
        default=0.5,
        metadata={"help": "KD loss weightage, 0 means pure training without KD"}
    )
    adaptive_smoothing: Optional[float] = field(
        default=None,
        metadata={"help": "beta for smoothing factor in the sigmoid function"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "kd_label_smoothed_cross_entropy", dataclass=KDLabelSmoothedCrossEntropyCriterionConfig
)
class KDLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        student_temp,
        teacher_temp,
        alpha,
        beta,
        decay,
        decoder_kd,
        self_kd,
        cross_kd,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        # new parameters
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.decoder_kd = decoder_kd
        self.self_kd = self_kd
        self.cross_kd = cross_kd
    
    def forward(self, model, sample, epoch=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, (attn_output, decoder_self_attn_output, decoder_cross_attn_output, encoder_states) = model(**sample["net_input"])
        decoder_states=net_output[1]['inner_states']

        teacher_output = sample.get("teacher_output", None)
        teacher_attn_output = sample.get("teacher_attn_output", None)
        teacher_decoder_self_attn_output = sample.get("teacher_decoder_self_attn_output", None)
        teacher_decoder_cross_attn_output = sample.get("teacher_decoder_cross_attn_output", None)
        teacher_decoder_states = sample.get('teacher_decoder_states', None)
                
        loss, extra = self.compute_loss(
            model, 
            net_output, 
            sample,
            epoch,
            teacher_output=teacher_output,
            attn=attn_output,
            decoder_self_attn=decoder_self_attn_output,
            decoder_cross_attn=decoder_cross_attn_output,
            teacher_attn=teacher_attn_output,
            teacher_decoder_self_attn=teacher_decoder_self_attn_output,
            teacher_decoder_cross_attn=teacher_decoder_cross_attn_output,
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'kd_loss': extra['kd_loss'].data if extra.get('kd_loss', None) is not None else 0,
            'nll_loss_student': extra['nll_loss_student'].data if extra.get('nll_loss_student', None) is not None else loss.data,
            'nll_loss_teacher': extra['nll_loss_teacher'].data if extra.get('nll_loss_teacher', None) is not None else 0,
            'attn_loss' : extra['attn_loss'].data if extra.get('attn_loss', None) is not None else 0,
            'decoder_self_attn_loss': extra['decoder_self_attn_loss'].data if extra.get('decoder_self_attn_loss', None) is not None else 0,
            'decoder_cross_attn_loss': extra['decoder_cross_attn_loss'].data if extra.get('decoder_cross_attn_loss', None) is not None else 0,
            'golden_loss': extra['golden_loss'].data if extra.get('golden_loss', None) is not None else 0,
        }
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # import torch.nn.functional as F
        # logits = net_output.float()
        # lprobs = F.log_softmax(logits, dim=-1)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    def compute_loss(self, model, net_output, sample, epoch=None, teacher_output=None, attn=None, decoder_self_attn=None, decoder_cross_attn=None,
                    teacher_attn=None, teacher_decoder_self_attn=None, teacher_decoder_cross_attn=None):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        pad_mask = target.eq(self.padding_idx).view(-1)
        extra = dict()

        # get student logits
        student_logits = net_output[0]
        student_logits = student_logits.view(-1, student_logits.size(-1))
        student_logits_T = student_logits/self.student_temp

        # get teacher probs
        teacher_logits = teacher_output[0]
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        teacher_probs_T = F.softmax(teacher_logits/self.teacher_temp, dim=-1, dtype=torch.float32)

        # compute teacher log-probs to get teacher loss value
        teacher_lprobs = sample.get("teacher_lprobs", None)

        # compute preliminary loss and nll_loss of student_model
        golden_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, 
            target, 
            self.eps, 
            ignore_index=self.padding_idx, 
            reduce=False
        )

        if teacher_lprobs is not None:
            # compute preliminary lprobs, loss, nll_loss of teacher_model
            teacher_lprobs = teacher_lprobs.view(-1, teacher_lprobs.size(-1))
            _, nll_loss_teacher = label_smoothed_nll_loss(
                teacher_lprobs, 
                target, 
                self.eps, 
                ignore_index=self.padding_idx, 
                reduce=False
            )

        nll_loss = nll_loss.view(-1)
        nll_loss_teacher = nll_loss_teacher.view(-1)
        golden_loss = golden_loss.view(-1)
        extra['golden_loss'] = golden_loss.mean()

        if teacher_output is None:
            loss = golden_loss
            
        kd_loss = F.cross_entropy(
            student_logits_T,
            teacher_probs_T,
            reduction='none'
        )
        kd_loss.masked_fill_(pad_mask, 0)
        extra['kd_loss'] = kd_loss.mean()
        extra['nll_loss_student'] = nll_loss.mean()
        extra['nll_loss_teacher'] = nll_loss_teacher.mean()

        loss = ((1.0 - self.alpha) * golden_loss).mean() + (self.alpha * kd_loss).mean()

        attn_loss = None
        decoder_self_attn_loss = None
        decoder_cross_attn_loss = None

        if attn is not None and teacher_attn is not None and epoch is not None:
            attn_loss =F.kl_div(F.log_softmax(rearrange(attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.beta
            if self.decoder_kd:
                if self.self_kd and self.cross_kd:
                    decoder_self_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) / 2 * self.beta
                    decoder_cross_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) / 2 * self.beta
                elif self.self_kd:
                    decoder_self_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.beta
                elif self.cross_kd:
                    decoder_cross_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.beta

        if attn_loss:
            extra['attn_loss'] = attn_loss.sum()
            loss += attn_loss
        if decoder_self_attn_loss:
            extra['decoder_self_attn_loss'] = decoder_self_attn_loss.sum()
            loss += decoder_self_attn_loss
        if decoder_cross_attn_loss:
            extra['decoder_cross_attn_loss'] = decoder_cross_attn_loss.sum()
            loss += decoder_cross_attn_loss
        return loss, extra


    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)  
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # sum metrics
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_student = sum(log.get('nll_loss_student', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nll_loss_teacher = sum(log.get('nll_loss_teacher', 0) for log in logging_outputs)
        kd_loss = sum(log.get('kd_loss', 0) for log in logging_outputs)
        attn_loss = sum(log.get('attn_loss', 0) for log in logging_outputs)
        decoder_self_attn_loss = sum(log.get('decoder_self_attn_loss', 0) for log in logging_outputs)
        decoder_cross_attn_loss = sum(log.get('decoder_cross_attn_loss', 0) for log in logging_outputs)
        golden_loss = sum(log.get('golden_loss', 0) for log in logging_outputs)
        # log metrics
        metrics.log_scalar(
            'loss', 
            loss, 
            round=3
        )
        metrics.log_scalar(
            'attn_loss', 
            attn_loss,
            round=3
        )
        metrics.log_scalar(
            'golden_loss', 
            golden_loss,
            round=3
        )
        metrics.log_scalar(
            'decoder_self_attn_loss', 
            decoder_self_attn_loss,
            round=3
        )
        metrics.log_scalar(
            'decoder_cross_attn_loss', 
            decoder_cross_attn_loss,
            round=3
        )
        metrics.log_scalar(
            'nll_loss', 
            nll_loss_student,
            round=3)
        metrics.log_scalar(
            'nll_loss_teacher', 
            nll_loss_teacher,
            round=3)
        metrics.log_scalar(
            'kd_loss', 
            kd_loss,
            round=3)
        metrics.log_derived(
            'ppl', 
            lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
