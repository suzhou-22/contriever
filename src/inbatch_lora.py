# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import numpy as np
import math
import random
import transformers
import logging
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from src import contriever, dist_utils, utils

logger = logging.getLogger(__name__)


lora_config = LoraConfig(
    r=16,
    init_lora_weights="gaussian",
    target_modules=["query", "value"],
    task_type=TaskType.FEATURE_EXTRACTION,
    lora_alpha=32,
    lora_dropout=0.05
)

class InBatchLora(nn.Module):
    def __init__(self, opt, encoder_query=None, encoder_doc=None, tokenizer=None):
        super(InBatchLora, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        self.tokenizer = tokenizer
        self.encoder_query = encoder_query
        self.encoder_doc = encoder_doc
        if encoder_query == None or encoder_doc == None:
            self.encoder_query, _ = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
            self.encoder_doc, _ = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
            
        self.encoder_doc = get_peft_model(self.encoder_doc, lora_config)
        self.encoder_doc.print_trainable_parameters()
        for param in self.encoder_query.parameters():
            param.requires_grad = False

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)
            print(model_id)
        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self, encoder_type = None):
        if encoder_type  == 'doc':
            return self.encoder_doc
        else:
            return self.encoder_query

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder_query(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder_doc(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats
