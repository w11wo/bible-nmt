python src/run_evaluation.py \
    --model_name nllb-200-distilled-1.3B-ebible-corpus-mt-ind-ptu \
    --dataset_name bible-nlp/biblenlp-corpus \
    --src_lang ind \
    --tgt_lang ptu \
    --src_lang_nllb ind_Latn \
    --tgt_lang_nllb ptu_Latn \
    --max_length 256 \
    --num_beams 8 \
    --per_device_eval_batch_size 16