# Multistage-Collaborative-Knowledge-Distillation  
This repo contains the code accompanying the paper [Multistage Collaborative Knowledge Distillation from a Large Language Model for Semi-Supervised Sequence Generation](https://arxiv.org/pdf/2311.08640) accepted by ACL 2024. 


## Implementations

src/train_eval.py: main code to run finetuning and evaluation  
src/parserlib.py: utility code for constituency parsing  
src/chunking_utils.py: utility code for semantic parsing  

### Pseudolabels  
https://huggingface.co/datasets/nickzzZzz/mckd-pseudolabels 

### MCKD   

```
python -u train_eval.py \
--model_name_or_path t5-base \
--predict_with_generate true \
--checkpointing_steps epoch \
--tokenizer_name t5-base \
--train_file "" \
--validation_file "" \
--max_target_length 512 \
--val_max_target_length 512 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 16 \
--learning_rate 3e-4 \
--num_train_epochs 20 \
--eval_epoch 2 \
--save_epoch 2 \
--do_train 1 \
--source_prefix "" \
--output_dir "" \
--save_state 1 \
--source_lang unparsed \
--target_lang unlabeled \
--report_to wandb \
--with_tracking
```

1. Randomly split pseudolabels into halves, e.g., dataset A and dataset B.
2. For one student model, set the train_file as path to the dataset A and the validation_file as dataset B. Specify the output_dir where the model's predictions on the validation file are stored.
For the other student model, simply set the train_file as dataset B and the validation_file as dataset A. For further iterations, collect the model's output predictions (i.e., relabeled dataset A and B) and repeat the above steps.
You can also modify the run.sh and further write another sh file like the following to automate the above iterations,

```
    train="datasetA.json"
    valid="datasetB.json"
    out="relabel-B-1"
    sh train.sh "$train" "$valid" "$out"

    train="datasetB.json"
    valid="datasetA.json"
    out="relabel-A-1"
    sh train.sh "$train" "$valid" "$out"

    train="relabel-A-1/epoch_{epoch}/result_valid_format_train.json"
    valid="datasetB.json"
    out="relabel-B-2"
    sh train.sh "$train" "$valid" "$out"

    train="relabel-B-1/epoch_{epoch}/result_valid_format_train.json"
    valid="datasetA.json"
    out="relabel-A-2"
    sh train.sh "$train" "$valid" "$out"
```

3. Merge the final students' relabeled A and B to train the ultimate student. 

## Citing
```bibtex  
@inproceedings{zhao2024multistage,
  title={Multistage Collaborative Knowledge Distillation from a Large Language Model for Semi-Supervised Sequence Generation},
  author={Zhao, Jiachen and Zhao, Wenlong and Drozdov, Andrew and Rozonoyer, Benjamin and Sultan, Arafat and Lee, Jay-yoon and Iyyer, Mohit and McCallum, Andrew},
  booktitle={Annual Meeting of the Association for Computational Linguistics},
  year={2024}
}
``` 
