# README

___

This directory contains the files for declaration generation. [T5-small](https://arxiv.org/pdf/1910.10683.pdf) is 
exploited as the encoder-decoder model for training and evaluation.

## INSTALL

+ Follow the [documentation](https://huggingface.co/docs/transformers/model_doc/t5) for installation.

## Data

Assume the root data dir is `[ROOT_DATA_DIR]`, then the declaration dataset for training and 
validation is placed in `[ROOT_DATA_DIR]/declaration/*`:
```
|- [ROOT_DATA_DIR]
    |- declaration/
        |- question_to_declarative_train.json
        |- question_to_declarative_val.json
```

## Model Training

Run the script for training:
```bash
bash run_train.sh
```

or the fine-tuned T5-small model (`model/declaration/checkpoint-480000`) can be 
downloaded from [Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA). The fine-tuned or downloaded checkpoint is placed in 
`data/model/declaration/checkpoint-480000`.

## Declaration Generation

Once T5 is trained on the declaration dataset, the model can be used to generate 
declarative sentences for GQA and VQA datasets. Just follow the steps as bellow:
1. Transform the questions (from GQA and VQA v2.0 datasets) into the translation 
   format (one sample per line), where `en_q` denotes the source question string 
   and `en_a` denotes the target declarative sentence we want to generate. The file 
   is named `source_file.txt`:
   ```text
   {"translation": {"en_q": "Is the sky dark?", "en_a": ""}}
   {"translation": {"en_q": "What is on the white wall?", "en_a": ""}}
   {"translation": {"en_q": "Is that pipe red?", "en_a": ""}}
   ...
   ```
2. Assume the path of `source_file.txt` is `[SOURCE_FILE_DIR]/source_file.txt`, then
   run the script:
   ```bash
   bash run_predict.sh
   ```
   Finally, there will be a `.txt` file in `output` dir, _i.e._, `generated_predictions.txt`.
   This file contains one sentence per line, representing the declarative 
   sentence of the corresponding question in `source_file.txt`.
   The format of `generated_predictions.txt` is shown as follows:
   ```text
   [MASK], the sky [BE] dark.
   the [MASK] is on the wall.
   [MASK], that pipe [BE] red.
   ```
3. We provide the pre-generated declaration files (from the questions of GQA and VQA v2.0 datasets) 
   for easy-to-use. The files can be downloaded from [Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA), the files are arranged as 
   follows:
   ```
   |- [ROOT_DATA_DIR]
       |- declaration/
           |- gqa/
               |- gqa_all_submission_declaration.json
               |- gqa_all_train_declaration.json
               |- gqa_all_val_declaration.json
               |- gqa_bal_train_declaration.json
               |- gqa_bal_val_declaration.json
           |- vqa/
               |- test2015_declarative.json
               |- test-dev2015_declarative.json
               |- train2014_declarative.json
               |- val2014_declarative.json
               |- vqa_vg_declarative.json
   ```