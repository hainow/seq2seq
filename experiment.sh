#!/usr/bin/env bash
# TENSORFLOW virtualenv
source ${HOME}/env_tensorflow/bin/activate

export VOCAB_SOURCE=${HOME}/nmt_data/toy_reverse/train/vocab.sources.txt
export VOCAB_TARGET=${HOME}/nmt_data/toy_reverse/train/vocab.targets.txt
export TRAIN_SOURCES=${HOME}/nmt_data/toy_reverse/train/sources.txt
export TRAIN_TARGETS=${HOME}/nmt_data/toy_reverse/train/targets.txt
export DEV_SOURCES=${HOME}/nmt_data/toy_reverse/dev/sources.txt
export DEV_TARGETS=${HOME}/nmt_data/toy_reverse/dev/targets.txt

export DEV_TARGETS_REF=${HOME}/nmt_data/toy_reverse/dev/targets.txt

export TRAIN_STEPS=1000

export MODEL_DIR=${TMPDIR:-/tmp}/nmt_tutorial
mkdir -p $MODEL_DIR



# ----------------- TRAIN -----------------------
echo "TRAINING NOW"
python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

echo "TRAINING COMPLETE!"


## --------------- INFERENCE ----------------------
#echo "INFERENCING NOW..."
#export PRED_DIR=${TMPDIR:-/tmp}/nmt_tutorial
#python -m bin.infer \
#  --tasks "
#    - class: DecodeText
#    - class: DumpBeams
#      params:
#        file: ${PRED_DIR}/beams.npz" \
#  --model_dir $MODEL_DIR \
#  --model_params "
#    inference.beam_search.beam_width: 5" \
#  --input_pipeline "
#    class: ParallelTextInputPipeline
#    params:
#      source_files:
#        - $DEV_SOURCES" \
#  > ${PRED_DIR}/predictions.txt
#
#echo "INFERENCE COMPLETE!"
#


## --------------- BLEU SCORES -----------------------
#./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt

