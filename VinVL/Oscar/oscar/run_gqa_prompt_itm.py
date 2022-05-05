# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import copy, time, json

import sys
sys.path.insert(0, '.')
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import _pickle as cPickle

from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import WEIGHTS_NAME, BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from transformers.pytorch_transformers.modeling_bert import BertPooler

from oscar.utils.misc import set_seed
from oscar.utils.task_utils import (_truncate_seq_pair, convert_examples_to_features_vqa,
                        output_modes, processors)

logger = logging.getLogger(__name__)

class ImageBertForSequenceClassificationPrompt(ImageBertForSequenceClassification):
    def __init__(self, config):
        super(ImageBertForSequenceClassificationPrompt, self).__init__(config)

        self.mask_pooler = BertPooler(config)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'):
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size * 2,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size * 2, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)  # original
        self.apply(self.init_weights)

        for p in self.parameters():
            p.requires_grad = False

        self.matching_cls_pooler = BertPooler(config)
        self.matching_msk_pooler = BertPooler(config)
        self.dropout = nn.Dropout(0.2)
        self.matcher = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
            position_ids=None, head_mask=None, img_feats=None, mask_index=None, predict_labels=None):
        with torch.no_grad():
            # (batch_size, sequence_length, hidden_dim), (batch_size, hidden_dim), ...
            outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

            # (batch_size, hidden_dim)
            # pooled_output = outputs[1]
            seq_length, hidden_dim = outputs[0].shape[1:]
            pooled_output = torch.gather(
                input=outputs[0],
                dim=1,
                index=mask_index.unsqueeze(-1).expand((-1, -1, hidden_dim))
            )   # [batch_size, 1, hidden_dim]

        pooled_output = self.matching_msk_pooler(pooled_output)
        pooled_output_cls = self.matching_cls_pooler(outputs[0])

        pooled_output = torch.cat([pooled_output, pooled_output_cls], dim=-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.matcher(pooled_output)   # [B, 1]

        outputs = (logits, ) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            target = (labels.view(-1) == predict_labels.view(-1)).float()
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), target)
            outputs = (loss, ) + outputs

        return outputs

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassificationPrompt, BertTokenizer),
}

log_json = []

def _load_dataset(args, name):
    processor = processors[args.task_name]()

    labels = processor.get_labels(args.label_file)

    if name == 'train':
        if args.train_data_type == 'bal':
            examples = processor.get_train_examples(args.data_dir, 'gqa_bal_qla_train.json') #[0: debug_size]

            with open(os.path.join(args.data_dir, 'gqa_bal_qla_train_declarative.json')) as fp:
                lines = fp.read().split("\n")
            assert len(examples) == len(lines)
            for example, line in zip(examples, lines):
                example.text_c = line
        else:
            examples = processor.get_train_examples(args.data_dir, 'gqa_all_qla_train.json') #[0: debug_size]
    elif name == 'val':
        if args.eval_data_type == 'bal':
            examples = processor.get_dev_examples(args.data_dir, 'gqa_bal_qla_val.json') #[0: debug_size]

            with open(os.path.join(args.data_dir, 'gqa_bal_qla_val_declarative.json')) as fp:
                lines = fp.read().split("\n")
            assert len(examples) == len(lines)
            for example, line in zip(examples, lines):
                example.text_c = line
        else:
            examples = processor.get_dev_examples(args.data_dir, 'gqa_all_qla_val.json') #[0: debug_size]
    elif name == 'train+val': # depreciated
        if args.data_label_type == 'mask':
            examples = processor.get_train_examples(args.data_dir, 'train+val2014_qla_mrcnn.json')
        else:
            examples = processor.get_train_examples(args.data_dir, 'train+val2014_qla.json')
    elif name == 'test': # test-submission
        if args.data_label_type == 'bal':
            examples = processor.get_test_examples(args.data_dir, 'gqa_all_qla_submission.json')
        else:
            examples = processor.get_test_examples(args.data_dir, 'gqa_all_qla_submission.json')

        with open(os.path.join(args.data_dir, 'gqa_all_qla_submission_declarative.json')) as fp:
            lines = fp.read().split("\n")
        assert len(examples) == len(lines)
        for example, line in zip(examples, lines):
            example.text_c = line
    elif name == 'test-dev': # test-dev set
        if args.data_label_type == 'bal':
            examples = processor.get_dev_examples(args.data_dir, 'gqa_bal_qla_testdev.json')
        else:
            examples = processor.get_dev_examples(args.data_dir, 'gqa_all_qla_testdev.json')

    return examples, labels

def _load_img_features(args):
    t_start = time.time()
    if args.img_feature_type == 'faster_r-cnn':
        if args.img_feature_dim == 2048:  # object features
            feat_file_name = 'gqa_img_frcnn_feats_obj.pt'
        else:  # object + spatial features
            feat_file_name = 'gqa_img_frcnn_feats.pt'
    else:
        feat_file_name = 'gqa_img_frcnn_feats.pt'
    img_features = torch.load(os.path.join(args.data_dir, feat_file_name))
    t_end = time.time()
    logger.info('Info: loading {0:s} features using {1:.2f} secs'.format(feat_file_name, (t_end - t_start)))

    return img_features

class GQADataset(Dataset):
    """ GQA Dataset """

    def __init__(self, args, name, img_features, tokenizer, label_pos_feats=None):
        super(GQADataset, self).__init__()
        assert name in ['train', 'val', 'test-dev', 'test', 'train+val']

        self.img_features = img_features
        self.label_pos_feats = label_pos_feats   # None
        self.output_mode = output_modes[args.task_name]   # classification
        self.tokenizer = tokenizer
        self.args = args
        self.name = name

        self.examples, self.labels = _load_dataset(args, name)
        self.label_map = {label: i for i, label in enumerate(self.labels)}

        # Record the topk results of MLM task
        self.idx2label = {k:v for v,k in self.label_map.items()}   # classification idx -> label idx
        self.label2ans = cPickle.load(open(args.label2ans_file, "rb"))  # label idx -> answer
        self.stage1_dict = dict()  # q_id -> (topk_logits, topk_scores)

        # False
        if self.args.load_fast:
            self.features = self.tensorize(args, cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        else:
            pass

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))

    def tensorize(self, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):

        # debug:
        debug_size = 500
        features = []

        for (ex_index, example) in enumerate(self.examples[0: ]):
            if len(example.label) == 0: continue
            if ex_index % 10000 == 0: logger.info("Tensorizing example %d of %d" % (ex_index, len(self.examples)))

            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.args.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.args.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length

            # image features
            img_feat = self.img_features[example.img_key] # torch
            #img_feat = self.img_features.item().get(example.img_key)  # numpy
            if img_feat.shape[0] > self.args.max_img_seq_length:
                img_feat = img_feat[0:self.args.max_img_seq_length, ]
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

            if self.args.output_mode == "classification":
                label_id = [self.label_map[l] for l in example.label]
                score = example.score
            elif self.args.output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(self.args.output_mode)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %s)" % (example.label, label_id))
                logger.info("score: %s (score = %s)" % (example.score, score))

            new_scores = target_tensor(len(self.labels), label_id, score)
            #features.append(InputFeat(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id, score=score, img_feat=img_feat))
            features.append((torch.tensor(input_ids, dtype=torch.long),
                            torch.tensor(input_mask, dtype=torch.long),
                            torch.tensor(segment_ids, dtype=torch.long),
                            torch.tensor([label_id[0]], dtype=torch.long),
                            torch.tensor(new_scores, dtype=torch.float), img_feat))

        return features

    def generate_sample(self, input_ids, input_mask, segment_ids, label_idx, mask_index):
        template = copy.copy(input_ids)
        template_mask = copy.copy(input_mask)
        template_segment = copy.copy(segment_ids)

        # Transform to answer string
        template_label = self.idx2label[label_idx]
        template_answer = self.label2ans[template_label]
        tokens_ans = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(template_answer))
        ans_length = len(tokens_ans)
        #
        template = (template[:mask_index] + tokens_ans + template[mask_index + 1:])[:len(input_ids)]
        template_mask = template_mask[:mask_index] + [1] * ans_length + template_mask[mask_index + 1:len(
            input_mask) - ans_length + 1] + \
                        template_mask[len(input_mask):]
        template_segment = (template_segment[:mask_index] + [
            template_segment[mask_index]] * ans_length + template_segment[mask_index + 1:])[:len(segment_ids)]

        assert len(template) == len(input_ids)
        assert len(template_mask) == len(input_mask)
        assert len(template_segment) == len(segment_ids)

        return template, template_mask, template_segment

    def tensorize_example(self, example, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):
        tokens_a = self.tokenizer.tokenize(example.text_a)

        tokens_c = self.tokenizer.tokenize("Answer: " + example.text_c)
        tokens_a = tokens_a + tokens_c

        tokens_b = None
        if example.text_b:
            txt_b_arr = example.text_b.split(';')
            txt_label_ixs = []
            for txt_b_ix, txt_b_ele in enumerate(txt_b_arr):
                tokens_b_ele = self.tokenizer.tokenize(txt_b_ele)
                txt_label_ixs.extend([txt_b_ix] * len(tokens_b_ele))
            txt_b = example.text_b.replace(';', ' ').strip()
            tokens_b = self.tokenizer.tokenize(txt_b)
            assert len(tokens_b) == len(txt_label_ixs)

            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
            txt_label_ixs = txt_label_ixs[0:len(tokens_b)]

        # original
        #if example.text_b:
        #    txt_b = example.text_b.replace(';', ' ').strip()
        #    tokens_b = self.tokenizer.tokenize(txt_b)
        #    _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
        else: # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        # question + [SEP]
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            # question + [SEP] + tags + [SEP]
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        # False
        if cls_token_at_end:
            raise ValueError("No CLS at end.")
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            # [CLS] + question + [SEP] + tags + [SEP]
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        if "[MASK]" in tokens:
            mask_index = tokens.index("[MASK]")
        else:
            mask_index = 0
            print("qid {} has no [MASK]".format(example.q_id))
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        # False
        if pad_on_left:
            raise ValueError("No pad on left.")
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            # [..., 0, 0, 0, ...]
            input_ids = input_ids + ([pad_token] * padding_length)
            # [1, 1, ..., 0, 0, 0, ...]
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            # [1, 0, 0, ..., 1, 1, ..., 0, 0, 0, ...]
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        # image features
        if self.args.img_feature_type.startswith('dis_code'):
            img_feat = self.img_features[example.img_key]

            if self.args.img_feature_type == 'dis_code_ln': # for discrete code image representation
                img_feat = img_feat.reshape(-1, img_feat.shape[0])

            if self.args.img_feature_type == 'dis_code_t': # transposed
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * 64
            else:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
        else:
            img_feat = self.img_features[example.img_key] #[:, 0:self.args.img_feature_dim]  # torch

            if img_feat.shape[0] > self.args.max_img_seq_length:
                img_feat = img_feat[0:self.args.max_img_seq_length, ]
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        if self.args.output_mode == "classification":
            if (example.label is None):
                label_id = [0]
                score = [0]
            elif len(example.label) == 0:
                label_id = [0]
                score = [0]
            else:
                label_id = [self.label_map[l] for l in example.label]
                score = example.score
        elif self.args.output_mode == "regression":
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
        else:
            raise KeyError(self.args.output_mode)

        if self.args.img_feature_type in ['dis_code', 'dis_code_t']:
            img_feat = img_feat.type(torch.long)
        elif self.args.img_feature_type in ['dis_code_ln']:
            #img_feat = img_feat.reshape(-1, img_feat.shape[0])
            img_feat = img_feat.type(torch.float)

        templates = []
        templates_mask = []
        templates_segment = []
        if self.name == "train":
            if example.q_id not in self.stage1_dict:
                if random.random() < 0.5:
                    template_idx = random.randint(0, len(self.label_map)-1)
                else:
                    template_idx = label_id[0]
            else:
                indexes, scores = self.stage1_dict[example.q_id]
                template_idx = int(np.random.choice(a=indexes))
            sample = self.generate_sample(input_ids, input_mask, segment_ids, template_idx, mask_index)
            templates.append(sample[0])
            templates_mask.append(sample[1])
            templates_segment.append(sample[2])
        else:
            template_idx = 0
            if len(self.stage1_dict) > 0:
                indexes, scores = self.stage1_dict[example.q_id]
                for ind, sco in zip(indexes, scores):
                    ind = int(ind)
                    sample = self.generate_sample(input_ids, input_mask, segment_ids, ind, mask_index)
                    templates.append(sample[0])
                    templates_mask.append(sample[1])
                    templates_segment.append(sample[2])

        return (torch.tensor(input_ids, dtype=torch.long),           # 0
                    torch.tensor(input_mask, dtype=torch.long),      # 1
                    torch.tensor(segment_ids, dtype=torch.long),     # 2
                    torch.tensor([label_id[0]], dtype=torch.long),   # 3
                    torch.tensor([label_id[0]], dtype=torch.long),   # 4
                    img_feat,                                        # 5
                    torch.tensor([example.q_id], dtype=torch.long),  # 6
                    torch.tensor([mask_index], dtype=torch.long),    # 7
                    torch.tensor(templates, dtype=torch.long),       # 8
                    torch.tensor(templates_mask, dtype=torch.long),  # 9
                    torch.tensor(templates_segment, dtype=torch.long),# 10
                    torch.tensor([template_idx], dtype=torch.long)  # 11
                    )

    def __getitem__(self, index):
        if self.args.load_fast:
            example = self.features[index]
        else:
            entry = self.examples[index]
            example = self.tensorize_example(entry,
                cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return example

    def __len__(self):
        return len(self.examples)

def trim_batch(batch):
    """ new batch func
    :param batch:
    :return:
    """
    print('batch size', len(batch))

    batch_size = len(batch)
    batch_tensors = []
    for ele in batch[0]:
        print(ele.shape, ele.size())
        zero_tensor = torch.zeros(([batch_size] + list(ele.size())))
        batch_tensors.append(zero_tensor)

    for b_id, b in enumerate(batch):
        print(b_id, len(b))
        for ele_id, ele in enumerate(b):
            print(ele_id, ele.shape)
            batch_tensors[ele_id][b_id] = ele
    return batch_tensors

def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  num_workers=args.workers,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size) #, collate_fn=trim_batch)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.fp16:
        scaler = GradScaler()

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    #train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    # Predict labels from stage1
    stage1_file = os.path.join(args.model_name_or_path, "stage1.pkl")
    assert os.path.exists(stage1_file)
    print("loading labels from stage1 using top-{}".format(args.k))
    train_dataset.stage1_dict = dict([(k,(v[0][:args.k], v[1][:args.k])) for k,v in cPickle.load(open(stage1_file, "rb")).items()])

    for epoch in range(int(args.num_train_epochs)):
    #for epoch in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        total_loss = 0
        total_norm = 0
        count_norm = 0

        t_start = time.time()

        for step, batch in enumerate(train_dataloader):
        #for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[8].squeeze(1),
                      'attention_mask': batch[9].squeeze(1),
                      'token_type_ids': batch[10].squeeze(1) if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3],
                      'img_feats':      None if args.img_feature_dim == -1 else batch[5],
                      'mask_index':     batch[7],
                      'predict_labels': batch[11]}

            with autocast(enabled=args.fp16):
                outputs = model(**inputs)

                #loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                loss, logits = outputs[:2]

                if args.n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu parallel training

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                count_norm += 1

            #batch_score = compute_score_with_logits(logits, batch[4]).sum()
            #train_score += batch_score.item()

            tr_loss += loss.item()
            total_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:# Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
                        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
                        if eval_score > best_score:
                            best_score = eval_score
                            best_model['epoch'] = epoch
                            best_model['model'] = copy.deepcopy(model)

                        logger.info("EVALERR: {}%".format(100 * best_score))
                    logging_loss = tr_loss

            #if args.max_steps > 0 and global_step > args.max_steps:
            #    epoch_iterator.close()
            #    break

        t_end = time.time()
        logger.info('Train Time Cost: %.3f' % (t_end-t_start))

        # evaluation
        logger.info("Epoch: %d" % (epoch))
        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)
            #best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())

        # save checkpoints
        if args.local_rank in [-1, 0] and args.save_epoch > 0 and epoch % args.save_epoch == 0: # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

            save_num = 0
            while (save_num < 10):
                try:
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    break
                except:
                    save_num += 1
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))

        epoch_log = {'epoch': epoch, 'eval_score': eval_score, 'best_score':best_score}
        log_json.append(epoch_log)

        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
            json.dump(log_json, fp)

        logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))
        logger.info("EVALERR: {}%".format(100*best_score))
        logger.info("LOSS: {}%".format(total_loss / len(train_dataset)))

    with open(args.output_dir + '/eval_logs.json', 'w') as fp:
        json.dump(log_json, fp)

    if args.local_rank in [-1, 0]: # Save the final model checkpoint
        output_dir = os.path.join(args.output_dir, 'best-{}'.format(best_model['epoch']))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

        save_num = 0
        while (save_num < 10):
            try:
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                tokenizer.save_pretrained(output_dir)
                break
            except:
                save_num += 1
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    #if args.n_gpu > 1: model = torch.nn.DataParallel(model) # debug: single-gpu or multi-gpus

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, num_workers=args.workers, sampler=eval_sampler, batch_size=args.eval_batch_size)

        stage1_file = os.path.join(args.model_name_or_path, "stage1_eval.pkl")
        assert os.path.exists(stage1_file)
        print("loading stage1 file and use top-{}.".format(args.k))
        eval_dataset.stage1_dict = dict([(k,(v[0][:args.k], v[1][:args.k])) for k,v in cPickle.load(open(stage1_file, "rb")).items()])

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        num_data = 0
        correct_num = 0
        correct_num_mat = 0
        correct_num_all = 0

        for batch in eval_dataloader:
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            templates = batch[8]
            templates_mask = batch[9]
            templates_segment = batch[10]
            mat_logits = []
            for i in range(templates.shape[1]):
                temp = templates[:, i]
                temp_mask = templates_mask[:, i]
                temp_segment = templates_segment[:, i]

                with torch.no_grad():
                    inputs = {'input_ids':      temp,
                              'attention_mask': temp_mask,
                              'token_type_ids': temp_segment if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                              'labels':         batch[3],
                              'img_feats':      None if args.img_feature_dim == -1 else batch[5],
                              'mask_index':     batch[7],
                              'predict_labels': batch[11]}
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    mat_logits.append(logits.view(-1).detach())

                    eval_loss += tmp_eval_loss.mean().item()

                #logger.info('logits: %s, batch[3]: %s' % (str(logits.shape), str(batch[3].shape)))
                #logger.info('correct: %s' % (str(logits.argmax(1) == batch[3].view(-1))))

            mat_logits = torch.stack(mat_logits, dim=-1)   # [B, K]
            mat_logits = torch.sigmoid(mat_logits)
            mat_logits = mat_logits / (mat_logits.sum(-1, keepdim=True) + 1e-6)

            q_ids = batch[6].squeeze(-1).detach().cpu().numpy()
            topk_indexes, topk_scores = [], []
            for _i, q_id in enumerate(q_ids):
                temp = eval_dataset.stage1_dict[int(q_id)]
                topk_indexes.append(temp[0])
                topk_scores.append(temp[1])

            topk_indexes = torch.from_numpy(np.stack(topk_indexes, axis=0)).to("cuda")  # [B, K]
            cls_logits = torch.from_numpy(np.stack(topk_scores, axis=0)).to("cuda")     # [B, K], normalized
            cls_logits = cls_logits.softmax(dim=-1)

            preds_cls = torch.gather(
                input=topk_indexes,
                dim=1,
                index=cls_logits.argmax(1).unsqueeze(-1)
            )
            correct_num += (preds_cls.view(-1) == batch[3].view(-1)).sum().item()

            preds_mat = torch.gather(
                input=topk_indexes,
                dim=1,
                index=mat_logits.argmax(1).unsqueeze(-1)
            )   # [B, 1]
            correct_num_mat += (preds_mat.view(-1) == batch[3].view(-1)).sum().item()

            preds_all = torch.gather(
                input=topk_indexes,
                dim=1,
                index=(mat_logits + cls_logits).argmax(1).unsqueeze(-1)
            )

            correct_num_all += (preds_all.view(-1) == batch[3].view(-1)).sum().item()
            num_data += logits.size(0)

            # debug
            #val, idx = logits.max(1)
            #logger.info('idx: %s, batch[4]: %s' % (str(idx.shape), str(batch[3].shape)))
            #for i in range(idx.size(0)):
            #    logger.info('idx: %d, pred: %d, real: %d' % (idx[i].item(), eval_dataset.labels[idx[i].item()], batch[3][i].item()))

            nb_eval_steps += 1

        # with open(os.path.join(args.output_dir, "stage2_eval.pkl"), "wb") as fp:
        #     cPickle.dump(stage2_dict, fp)

        acc = float(correct_num) / len(eval_dataloader.dataset)
        acc_mat = float(correct_num_mat) / len(eval_dataloader.dataset)
        acc_all = float(correct_num_all) / len(eval_dataloader.dataset)

        logger.info("Eval Results:")
        logger.info("Eval Accuracy: %.3f" % (100*acc))
        logger.info("Eval MAT Accuracy: %.3f" % (100*acc_mat))
        logger.info("Eval ALL Accuracy: %.3f" % (100*acc_all))
        logger.info("Eval Loss: %.3f" % (eval_loss))

    t_end = time.time()
    logger.info('Eva Time Cost: %.3f' % (t_end - t_start))

    return results, max(acc, acc_mat, acc_all)

def test_stage2(args, model, eval_dataset=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))
    logger.info('label2ans: %d' % (len(label2ans)))

    stage1_file = os.path.join(args.model_name_or_path, "stage1_submission.pkl")
    assert os.path.exists(stage1_file)
    print("loading stage1 file ...")
    eval_dataset.stage1_dict = dict([(k,(v[0][:args.k], v[1][:args.k])) for k,v in cPickle.load(open(stage1_file, "rb")).items()])

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval
        logger.info("***** Running Test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        for batch in tqdm(eval_dataloader):
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            templates = batch[8]
            templates_mask = batch[9]
            templates_segment = batch[10]
            mat_logits = []
            for i in range(templates.shape[1]):
                temp = templates[:, i]
                temp_mask = templates_mask[:, i]
                temp_segment = templates_segment[:, i]

                with torch.no_grad():
                    inputs = {'input_ids': temp,
                              'attention_mask': temp_mask,
                              'token_type_ids': temp_segment if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'labels': batch[3],
                              'img_feats': None if args.img_feature_dim == -1 else batch[5],
                              'mask_index': batch[7],
                              'predict_labels': batch[11]}
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    mat_logits.append(logits.view(-1).detach())

            mat_logits = torch.stack(mat_logits, dim=-1)  # [B, K]
            mat_logits = torch.sigmoid(mat_logits)
            mat_logits = mat_logits / (mat_logits.sum(-1, keepdim=True) + 1e-6)

            q_ids = batch[6].squeeze(-1).detach().cpu().numpy()
            topk_indexes, topk_scores = [], []
            for _i, q_id in enumerate(q_ids):
                temp = eval_dataset.stage1_dict[int(q_id)]
                topk_indexes.append(temp[0])
                topk_scores.append(temp[1])
            topk_indexes = torch.from_numpy(np.stack(topk_indexes, axis=0)).to("cuda")  # [B, K]
            cls_logits = torch.from_numpy(np.stack(topk_scores, axis=0)).to("cuda")
            cls_logits = cls_logits.softmax(-1)

            preds_all = torch.gather(
                input=topk_indexes,
                dim=1,
                index=(mat_logits + cls_logits).argmax(1).unsqueeze(-1)
            ).squeeze(-1)  # [B, ]

            for i in range(preds_all.size(0)):
                result = {}
                result['questionId'] = str(batch[6][i].item())
                result['prediction'] = label2ans[eval_dataset.labels[preds_all[i].item()]]
                results.append(result)

    with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:
        json.dump(results, fp)

    t_end = time.time()
    logger.info('# questions: %d' % (len(results)))
    logger.info('Test Time Cost: %.3f' % (t_end - t_start))

def test(args, model, eval_dataset=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))
    logger.info('label2ans: %d' % (len(label2ans)))

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval
        logger.info("***** Running Test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        for batch in eval_dataloader:
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         None,
                          'img_feats':      None if args.img_feature_dim == -1 else batch[5],
                          'mask_index':     batch[7]}
                outputs = model(**inputs)
                logits = outputs[0]

                val, idx = logits.max(1)
                #logger.info('idx: %s, batch[6]: %s' % (str(idx.shape), str(batch[6].shape)))

                for i in range(idx.size(0)):
                    result = {}
                    result['questionId'] = str(batch[6][i].item())
                    result['prediction'] = label2ans[eval_dataset.labels[idx[i].item()]]
                    results.append(result)

                    #logger.info('q_id: {0}, answer: {1}'.format(result['question_id'], result['answer']))

    with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:
        json.dump(results, fp)

    t_end = time.time()
    logger.info('# questions: %d' % (len(results)))
    logger.info('Test Time Cost: %.3f' % (t_end - t_start))

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    label_list = processor.get_labels(args.label_file)

    t_start = time.time()
    examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

    #img_features = torch.load(os.path.join(args.data_dir, 'val_img_feats.pt' if evaluate else 'train_img_feats.pt'))
    #img_features = torch.load(os.path.join(args.data_dir, 'val_img_frcnn_feats.pt' if evaluate else 'train_img_frcnn_feats.pt'))
    img_features = np.load(os.path.join(args.data_dir, 'val_img_frcnn_feats.npy' if evaluate else 'train_img_frcnn_feats.npy'))

    features = convert_examples_to_features_vqa(examples, img_features, label_list, args.max_img_seq_length, args.max_seq_length,
            tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    #if args.local_rank in [-1, 0]:
    #    logger.info("Saving features into cached file %s", cached_features_file)
    #    torch.save(features, cached_features_file)
    t_end = time.time()
    logger.info('Info: loading features using %.5f secs' % (t_end-t_start))


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long) # batch*max_seq_len
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        labels = torch.tensor([f.label_id[0] for f in features], dtype=torch.long)
        targets = torch.tensor([target_tensor(len(label_list), f.label_id, f.score) for f in features], dtype=torch.float)

        if args.img_feature_dim > 0: # change here
            t_start = time.time()
            img_feat_np = np.zeros((labels.shape[0], args.max_img_seq_length, args.img_feature_dim))
            for f_id, f in enumerate(features):
                img_feat_np[f_id] = f.img_feat

            img_feats = torch.from_numpy(img_feat_np)

            #img_feats = torch.empty((labels.shape[0], args.max_img_seq_length, args.img_feature_dim))
            #for f_id, f in enumerate(features):
            #   img_feats[f_id] = f.img_feat

            t_end = time.time()
            logger.info('Info: convert image tensor features using %.5f secs' % (t_end - t_start))

            #img_feats = torch.stack([f.img_feat[:,-args.img_feature_dim:] for f in features])
            #img_feats = torch.stack([f.img_feat for f in features])
        #img_feats = img_feats.type(torch.long)

        #print('targets:', targets.shape)
        print('img_feats:', img_feats.shape)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    if args.img_feature_dim == -1:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels, targets)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels, targets, img_feats)
    return dataset

def target_tensor(len, labels, scores):
    """ create the target by labels and scores """
    target = [0]*len
    for id, l in enumerate(labels):
        target[l] = scores[id]

    return target

"""
主函数入口；
"""
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--label_file", type=str, default=None, help="Label Dictionary")
    parser.add_argument("--label2ans_file", type=str, default=None, help="Label to Answer Dictionary")

    parser.add_argument("--data_label_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--train_data_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--eval_data_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--loss_type", default='kl', type=str, help="kl or xe")

    parser.add_argument("--spatial_dim", default=6, type=int, help="spatial_dim")

    parser.add_argument("--max_label_pos_length", default=45, type=int, help="The maximum total input label position sequence length.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--do_test_dev", action='store_true', help="Whether to run test on the test-dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")

    parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    parser.add_argument("--code_level", default='top', type=str, help="code level: top, botttom, both")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=5, help="Save checkpoint every X epochs.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")
    parser.add_argument("--load_fast", action='store_true', help="Load Tensor Fast")
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--k', default=10, type=int, help="Number of answers for ITM.")

    #args = '--data_dir ../vqa/ban-vqa/data/qal_pairs --model_type bert --model_name_or_path bert-base-uncased --task_name vqa_text ' \
    #       '--do_train --do_eval --do_lower_case --max_seq_length 40 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --learning_rate 2e-5 ' \
    #       '--num_train_epochs 20.0 --output_dir ./results/vqa_text --label_file ../vqa/ban-vqa/data/cache/trainval_ans2label.pkl ' \
    #       '--save_steps 5000 --overwrite_output_dir --max_img_seq_length 1 --img_feature_dim 565 --img_feature_type dis_code '

    #args = '--data_dir ../vqa/ban-vqa/data/qal_pairs --model_type bert --model_name_or_path bert-base-uncased --task_name vqa_text ' \
    #       '--do_train --do_eval --do_lower_case --max_seq_length 40 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --learning_rate 2e-5 ' \
    #       '--num_train_epochs 20.0 --output_dir ./results/vqa_text --label_file ../vqa/ban-vqa/data/cache/trainval_ans2label.pkl ' \
    #       '--save_steps 5000 --overwrite_output_dir --max_img_seq_length 10 --img_feature_dim 565 --img_feature_type other '

    #args = parser.parse_args(args.split())

    args = parser.parse_args()

    if args.philly:  # use philly
        logger.info('Info: Use Philly, all the output folders are reset.')
        args.output_dir = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.output_dir)
        logger.info('OUTPUT_DIR:', args.output_dir)

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("Output Directory Exists.")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]

    label_list = processor.get_labels(args.label_file)
    num_labels = len(label_list)
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # discrete code
    config.img_feature_dim = args.img_feature_dim     # 2054
    config.img_feature_type = args.img_feature_type   # faster_r-cnn
    config.code_voc = args.code_voc                   # 512
    config.hidden_dropout_prob = args.drop_out        # 0.3
    config.loss_type = args.loss_type                 # xe
    config.classifier = args.classifier               # linear
    config.cls_hidden_scale = args.cls_hidden_scale   # 2
    config.spatial_dim = args.spatial_dim             # 6

    # load discrete code
    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Load discrete code from: {}'.format(args.data_dir))
        t_start = time.time()
        train_code = torch.load(os.path.join(args.data_dir, 'vqvae', 'train.pt'))
        t_end = time.time()
        logger.info('Load time: %.3f' % (t_end - t_start))

        if args.code_level == 'top':
            config.code_dim = train_code['embeddings_t'].shape[0]
            config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
        elif args.code_level == 'bottom':
            config.code_dim = train_code['embeddings_b'].shape[0]
            config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
        elif args.code_level == 'both':
            config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]

    # 初始化模型
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Initializing the code embedding with {}'.format(args.code_level))
        if args.code_level == 'top':
            model.init_code_embedding(train_code['embeddings_t'].t())
        elif args.code_level == 'bottom':
            model.init_code_embedding(train_code['embeddings_b'].t())

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # load image features
    img_features = _load_img_features(args)
    label_pos_feats = None

    #if args.do_eval:
    eval_dataset = GQADataset(args, 'val', img_features, tokenizer, label_pos_feats)
    #eval_dataset = GQADataset(args, 'test-dev', img_features, tokenizer) # test-dev as val
    if args.do_val:
        evaluate(args, model, eval_dataset)
        exit(0)

    if args.do_test:
        test_dataset = GQADataset(args, 'test', img_features, tokenizer, label_pos_feats)

    if args.do_test_dev:
        test_dev_dataset = GQADataset(args, 'test-dev', img_features, tokenizer, label_pos_feats)

    # Training
    if args.do_train:
        train_dataset = GQADataset(args, 'train', img_features, tokenizer, label_pos_feats)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Training on train+val
    if args.do_train_val: # depreciated
        train_dataset = GQADataset(args, 'train+val', img_features, tokenizer)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]: os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`. They can then be reloaded using `from_pretrained()`
        #model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        #model_to_save.save_pretrained(args.output_dir)

        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        #model = model_class.from_pretrained(args.output_dir)
        #tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        #model.to(args.device)

    # Evaluation
    #results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result, score = evaluate(args, model, eval_dataset, prefix=global_step)
            #result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            #results.update(result)

    # Test-Dev
    if args.do_test_dev and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result, score = evaluate(args, model, test_dev_dataset, prefix=global_step)
            #test(args, model, test_dev_dataset, prefix=global_step)

    # Test-Submission
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            global_step = "submission"
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            test_stage2(args, model, test_dataset, prefix=global_step)


if __name__ == "__main__":
    main()
