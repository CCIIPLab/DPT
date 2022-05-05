# README
___
This directory contains the files for training and evaluating vision-language 
pre-trained model, i.e., VinVL. Most of the files are copied from VinVL 
[github repo](https://github.com/microsoft/Oscar/tree/vinvl). We mainly modified or 
added the following files:
```
|- Oscar/
   |- oscar/
      |- run_gqa_prompt_mlm.py
      |- run_gqa_prompt_itm.py
      |- run_gqa_prompt_zero_few.py
      |- run_vqa_prompt_mlm.py
      |- run_vqa_prompt_itm.py
      |- utils/
         |- task_utils.py
```

## INSTALL

+ Refer to [INSTALL](https://github.com/microsoft/Oscar/tree/vinvl) for installation.

## Data Preparation

### GQA
Please follow the steps bellow to configure data:
1. Refer to [DOWNLOAD](https://github.com/microsoft/Oscar/blob/vinvl/VinVL_DOWNLOAD.md) 
   to download the pre-processed GQA dataset.
   The downloaded data should contain following files:
   ```
   |- [DATA_ROOT]/gqa/
       |- gqa_bal_qla_train.json
       |- gqa_bal_qla_val.json
       |- gqa_all_qla_train.json
       |- gqa_all_qla_val.json
       |- gqa_all_qla_submission.json
       ...
   ```
2. Download the corresponding declaration files and put them in the `gqa/` directory. 
   The declaration files are downloaded from [Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA) (`data/vinvl/gqa/*_declarative.json`). 
   These files contain declarative sentence per line, which can used for later data-loading
   and processing. Please put these `*_declarative.json` files into the `gqa/` directory, 
   resulting in following directory tree:
   ```text
   |- [DATA_ROOT]/gqa/
       |- gqa_bal_qla_train.json
       |- gqa_bal_qla_val.json
       |- gqa_all_qla_train.json
       |- gqa_all_qla_val.json
       |- gqa_all_qla_submission.json
       |- gqa_bal_qla_train_declarative.json  # newly added
       |- gqa_bal_qla_val_declarative.json  # newly added
       |- gqa_all_qla_train_declarative.json  # newly added
       |- gqa_all_qla_val_declarative.json  # newly added
       |- gqa_all_qla_submission_declarative.json  # newly added
       ...
   ```

### VQA v2.0

Please follow the steps bellow to configure data:
1. Refer to [DOWNLOAD](https://github.com/microsoft/Oscar/blob/vinvl/VinVL_DOWNLOAD.md) 
   to download the pre-processed VQA v2.0 dataset.
   The downloaded data should contain following files:
   ```
   |- [DATA_ROOT]/vqa/
       |- train2014_qla_mrcnn.json
       |- val2014_qla_mrcnn.json
       ...
   ```
2. Download the corresponding declaration files and put them in the `vqa/` directory. 
   The declaration files are downloaded from [Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA) (`data/vinvl/vqa/*_declarative.json`). 
   Please put these `*_declarative.json` files into the `vqa/` directory,
   resulting in following directory tree:
   ```text
   |- [DATA_ROOT]/vqa/
       |- train2014_qla_mrcnn.json
       |- val2014_qla_mrcnn.json
       |- train2014_declarative.json  # newly added
       |- val2014_declarative.json  # newly added
       ...
   ```

## Pre-trained Model

Please refer to [DOWNLOAD](https://github.com/microsoft/Oscar/blob/vinvl/VinVL_DOWNLOAD.md)
to download the pre-trained VinVL base model (`checkpoint-2000000`). We also provide the 
model checkpoint in [Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA) (`data/model/vinvl/checkpoint-2000000`). Assume that the 
`checkpoint-2000000` is placed in directory `[MODEL_ROOT]`, resulting in `[MODEL_ROOT]/checkpoint-2000000/`

## Training and Validation

### GQA
Please follow the steps bellow to reproduce the results (we take the balanced 
split for example). 

We first utilize the adapted **masked language model (MLM) task** for GQA fine-tuning:

1. **Training(MLM)**: Run the following code to train VinVL-DPT(MLM) on the balanced split:
   ```bash
   python oscar/run_gqa_prompt_mlm.py \
    -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
    --data_dir [DATA_ROOT]/gqa/ \
    --model_type bert \
    --model_name_or_path [MODEL_ROOT]/checkpoint-2000000/ \
    --task_name gqa --do_lower_case --max_seq_length 165 \
    --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
    --learning_rate 5e-05 --num_train_epochs 6 --output_dir gqa_mlm \
    --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
    --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
    --eval_data_type bal \
    --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
    --loss_type xe --save_epoch 1 --seed 88 --evaluate_during_training \
    --logging_steps 4000 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps 0 \
    --gradient_accumulation_steps 2
   ```
   If successful, the _overall_ accuracy will reach up to ~62.7%. We also 
   provide the fine-tuned model in [Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA) (`data/model/vinvl/vinvl_bal_mlm`).
2. **Validation(MLM)**: Evaluate the fine-tuned model on the GQA validation set using the fine-tuned model we 
   provide (or the model in the output_dir `gqa_mlm`).
   ```bash
   python oscar/run_gqa_prompt_mlm.py \
    -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
    --data_dir [DATA_ROOT]/gqa/ \
    --model_type bert \
    --model_name_or_path data/model/vinvl/vinvl_bal_mlm \
    --task_name gqa --do_lower_case --max_seq_length 165 \
    --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
    --learning_rate 5e-05 --num_train_epochs 6 --output_dir gqa_mlm \
    --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
    --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
    --eval_data_type bal \
    --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
    --loss_type xe --save_epoch 1 --seed 88 --evaluate_during_training \
    --logging_steps 4000 --drop_out 0.3 --do_val --weight_decay 0.05 --warmup_steps 0 \
    --gradient_accumulation_steps 2
   ```
   Note that the `model_name_or_path` and `--do_val` arguments have been changed compared to 
   training stage.
3. **Testing and Submission(MLM)**: Test the fine-tuned model and submit the result file
   to the [online evaluation website](https://eval.ai/web/challenges/challenge-page/225/leaderboard/733):
   Run the following code:
   ```bash
   python oscar/run_gqa_prompt_mlm.py \
    -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
    --data_dir [DATA_ROOT]/gqa/ \
    --model_type bert \
    --model_name_or_path data/model/vinvl/vinvl_bal_mlm \
    --task_name gqa --do_lower_case --max_seq_length 165 \
    --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
    --learning_rate 5e-05 --num_train_epochs 6 --output_dir gqa_mlm \
    --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
    --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
    --eval_data_type bal \
    --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
    --loss_type xe --save_epoch 1 --seed 88 --evaluate_during_training \
    --logging_steps 4000 --drop_out 0.3 --do_test --weight_decay 0.05 --warmup_steps 0 \
    --gradient_accumulation_steps 2
   ```
   Note that the `--do_test` argument has been changed compared to 
   validation stage.
   
Then, we apply the adapated **image-text matching (ITM) task** to solve VQA problem. To 
achieve this, we need to obtain the top-k candidate answer predicted by MLM tasks. Specifically,
we pre-generate the prediction results of MLM task:
+ Pre-generate topk results for training and validation.
  ```bash
   python oscar/run_gqa_prompt_mlm.py \
       -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
       --data_dir [DATA_ROOT]/gqa/ \
       --model_type bert \
       --model_name_or_path data/model/vinvl/vinvl_bal_mlm/ \
       --task_name gqa --do_lower_case --max_seq_length 165 \
       --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
       --learning_rate 5e-05 --num_train_epochs 6 --output_dir gqa_mlm \
       --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
       --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
       --eval_data_type bal \
       --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
       --loss_type xe --save_epoch 1 --seed 88 --evaluate_during_training \
       --logging_steps 4000 --drop_out 0.3 --do_train --do_generate --weight_decay 0.05 --warmup_steps 0 \
       --gradient_accumulation_steps 2
   ```
+ Pre-generate topk results for submission.
   ```bash
   python oscar/run_gqa_prompt_mlm.py \
       -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
       --data_dir [DATA_ROOT]/gqa/ \
       --model_type bert \
       --model_name_or_path data/model/vinvl/vinvl_bal_mlm/ \
       --task_name gqa --do_lower_case --max_seq_length 165 \
       --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
       --learning_rate 5e-05 --num_train_epochs 6 --output_dir gqa_mlm \
       --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
       --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
       --eval_data_type bal \
       --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
       --loss_type xe --save_epoch 1 --seed 88 --evaluate_during_training \
       --logging_steps 4000 --drop_out 0.3 --do_test --do_generate --weight_decay 0.05 --warmup_steps 0 \
       --gradient_accumulation_steps 2
   ```


Note that the `--do_generate` argument has been added. In this way, there will be two result files 
saved in `model_name_or_path`, i.e., `stage1.pkl`, `stage1_eval.pkl`, and `stage1_submission.pkl`. The files have following 
data format:
```text
{
    "[QID]": (np.ndarray([topk, ], np.int16),     # Topk answer indices
              np.ndarray([topk, ], np.float16),), # Topk answer scores
    ...
}
```
> We also provide the result files in the fine-tuned checkpoint `vinvl_bal_mlm`.

4. **Training(ITM)**: Equipped with the pre-generated topk answers, we can apply ITM by running following
code:
   ```bash
   python oscar/run_gqa_prompt_itm.py \
       -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
       --data_dir [DATA_ROOT]/gqa/ \
       --model_type bert \
       --model_name_or_path data/model/vinvl/vinvl_bal_mlm/ \
       --task_name gqa --do_lower_case --max_seq_length 165 \
       --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
       --learning_rate 5e-05 --num_train_epochs 2 --output_dir gqa_itm \
       --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
       --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
       --eval_data_type bal \
       --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
       --loss_type xe --save_epoch 1 --seed 88 --evaluate_during_training \
       --logging_steps 4000 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps 0 \
       --gradient_accumulation_steps 2
   ```
Note that we need to load the checkpoint from MLM task. We also provide the checkpoint in 
[Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA) (`data/model/vinvl/vinvl_bal_itm/`).
5. **Validation(ITM)**: Once the model is fine-tuned via ITM, we can validate the model 
through following code:
   ```bash
   python oscar/run_gqa_prompt_itm.py \
       -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
       --data_dir [DATA_ROOT]/gqa/ \
       --model_type bert \
       --model_name_or_path data/model/vinvl/vinvl_bal_itm/ \
       --task_name gqa --do_lower_case --max_seq_length 165 \
       --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
       --learning_rate 5e-05 --num_train_epochs 2 --output_dir gqa_itm \
       --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
       --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
       --eval_data_type bal \
       --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
       --loss_type xe --save_epoch 1 --seed 88 --evaluate_during_training \
       --logging_steps 4000 --drop_out 0.3 --do_val --weight_decay 0.05 --warmup_steps 0 \
       --gradient_accumulation_steps 2
   ```
   Note that the pre-generate result files, i.e., `stage1.pkl`, `stage1_eval.pkl`, and `stage1_submission.pkl`
   should be copied to `data/model/vinvl/vinvl_bal_itm/` so that the code has the access to the 
   MLM results.
6. **Testing and Submission(ITM)**: (Please make sure that the `stage1_submission.pkl` has been
   pre-generated or downloaded, and placed in the `model_name_or_path`.) Run the following code to run testing:
   ```bash
   python oscar/run_gqa_prompt_itm.py \
       -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
       --data_dir [DATA_ROOT]/gqa/ \
       --model_type bert \
       --model_name_or_path data/model/vinvl/vinvl_bal_itm/ \
       --task_name gqa --do_lower_case --max_seq_length 165 \
       --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
       --learning_rate 5e-05 --num_train_epochs 2 --output_dir gqa_itm \
       --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
       --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
       --eval_data_type bal \
       --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
       --loss_type xe --save_epoch 1 --seed 88 --evaluate_during_training \
       --logging_steps 4000 --drop_out 0.3 --do_test --weight_decay 0.05 --warmup_steps 0 \
       --gradient_accumulation_steps 2
   ```

### VQA v2.0

Please follow the steps bellow to reproduce the results on VQA v2.0:

We first utilize the **masked language model (MLM) task** to fine-tune the model:
1. **Training(MLM)**: Run the following code to train VinVL-DPT(MLM):
   ```bash
   python oscar/run_vqa_prompt_mlm.py -j 4 \
       --img_feature_dim 2054 --max_img_seq_length 50 \
       --data_label_type mask --img_feature_type faster_r-cnn \
       --data_dir [DATA_ROOT]/vqa --model_type bert \
       --model_name_or_path [MODEL_ROOT]/checkpoint-2000000 \
       --task_name vqa_text --do_train --do_lower_case --max_seq_length 158 \
       --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
       --learning_rate 5e-05 --num_train_epochs 25 \
       --output_dir vqa_mlm --label_file [DATA_ROOT]/vqatrainval_ans2label.pkl \
       --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 \
       --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type bce \
       --img_feat_format pt --classifier linear --cls_hidden_scale 3 \
       --txt_data_dir [DATA_ROOT]/vqa
   ```
   We also provide the checkpoint in [Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA) (`data/model/vinvl/vqa_mlm/`).
   Then, we pre-generate the topk results of MLM task via following code:
   ```bash
   python oscar/run_vqa_prompt_mlm.py -j 4 \
       --img_feature_dim 2054 --max_img_seq_length 50 \
       --data_label_type mask --img_feature_type faster_r-cnn \
       --data_dir [DATA_ROOT]/vqa --model_type bert \
       --model_name_or_path data/model/vinvl/vqa_mlm/ \
       --task_name vqa_text --do_train --do_generate --do_lower_case --max_seq_length 158 \
       --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
       --learning_rate 5e-05 --num_train_epochs 25 \
       --output_dir vqa_mlm --label_file [DATA_ROOT]/vqatrainval_ans2label.pkl \
       --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 \
       --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type bce \
       --img_feat_format pt --classifier linear --cls_hidden_scale 3 \
       --txt_data_dir [DATA_ROOT]/vqa
   ```
   Note that `model_name_or_path` and `do_generate` arguments have been changed. In this way, 
   two result files are generated and saved in `model_name_or_path`, i.e., `stage1.pkl` and 
   `stage1_eval.pkl`.
2. **Training(ITM)**: Run the following code to train image-text matching (ITM) task for VQA:
   ```bash
   python oscar/run_vqa_prompt_itm.py -j 4 \
       --img_feature_dim 2054 --max_img_seq_length 50 \
       --data_label_type mask --img_feature_type faster_r-cnn \
       --data_dir [DATA_ROOT]/vqa --model_type bert \
       --model_name_or_path data/model/vinvl/vqa_mlm/ \
       --task_name vqa_text --do_train --do_lower_case --max_seq_length 158 \
       --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 \
       --learning_rate 5e-05 --num_train_epochs 6 \
       --output_dir vqa_itm --label_file [DATA_ROOT]/vqatrainval_ans2label.pkl \
       --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 \
       --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type bce \
       --img_feat_format pt --classifier linear --cls_hidden_scale 3 \
       --txt_data_dir [DATA_ROOT]/vqa
   ```
   We also provide the fine-tuned checkpoint in [Baidu Yun (PSW:8888)](https://pan.baidu.com/s/1BnMODk2q92KQ0FPTz33zkA) (`data/model/vinvl/vqa_itm/`).

## Zero-shot and Few-shot Learning

In zero-shot and few-shot settings, zero or only a few samples (1~128) are used to fine-tune the 
model. Run the following code to split `[K]`-shot training set for fine-tuning, and evaluate on the 
whole validation set.
```bash
python oscar/run_gqa_prompt_zero_few.py \
   -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
   --data_dir [DATA_ROOT]/gqa/ \
   --model_type bert \
   --model_name_or_path [MODEL_ROOT]/checkpoint-2000000/ \
   --task_name gqa --do_lower_case --max_seq_length 165 \
   --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 1 \
   --learning_rate 5e-05 --num_train_epochs 25 --output_dir gqa_subset \
   --label_file [DATA_ROOT]/gqa/trainval_testdev_all_ans2label.pkl \
   --img_feature_type faster_r-cnn --data_label_type all --train_data_type bal \
   --eval_data_type bal \
   --label2ans_file [DATA_ROOT]/gqa/trainval_testdev_all_label2ans.pkl \
   --loss_type xe --save_epoch 10 --seed 88 --evaluate_during_training \
   --logging_steps 4000 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps 0 \
   --gradient_accumulation_steps 1 \
   --num_examples [K] --subset_seed 0
```

## Online Results

+ VinVL Baseline trained on balanced split: 
  + [testdev](https://evalai.s3.amazonaws.com/media/submission_files/submission_176247/bd8113ca-6e55-4559-a508-9ecedbd4a49c.json)
  + [teststd](https://evalai.s3.amazonaws.com/media/submission_files/submission_176248/55c074d6-f381-4458-8413-69333596d8f7.json)
+ VinVL-DPT trained on balanced split:
  + [testdev](https://evalai.s3.amazonaws.com/media/submission_files/submission_176397/6150d2b9-09cf-4ac9-aecc-ae859be3fc79.json)
  + [teststd](https://evalai.s3.amazonaws.com/media/submission_files/submission_176400/e6f063d1-0df9-4c82-9b4c-c1ed130857bc.json)
+ VinVL-DPT trained on all split:
  + [testdev](https://evalai.s3.amazonaws.com/media/submission_files/submission_176134/ec622e48-6b78-47a4-a939-13c466d0622c.json)
  + [teststd](https://evalai.s3.amazonaws.com/media/submission_files/submission_176088/2eee503a-a756-4752-b48e-758a85ebd35b.json)