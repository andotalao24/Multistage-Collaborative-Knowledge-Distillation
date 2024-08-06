
python -u train_eval.py \
--model_name_or_path t5-base \
--predict_with_generate true \
--checkpointing_steps epoch \
--tokenizer_name t5-base \
--validation_file "" \
--max_target_length 512 \
--val_max_target_length 512 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 16 \
--learning_rate 3e-4 \
--source_prefix "" \
--output_dir "" \
--source_lang unparsed \
--target_lang unlabeled \
--report_to wandb \
--do_train 0 \
--resume_from_checkpoint ""
