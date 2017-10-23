# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import yaml
import numpy as np
import tensorflow as tf

from seq2seq.data import vocab, input_pipeline
from seq2seq.training import utils as training_utils
from seq2seq.test import utils as test_utils
from seq2seq.models import BasicSeq2Seq, AttentionSeq2Seq


class EncoderDecoderTest(object):
  def __init__(self):
    self.batch_size = 2
    self.input_depth = 4
    self.sequence_length = 10

    # Create vocabulary
    self.vocab_list = [str(_) for _ in range(10)]
    self.vocab_list += ["笑う", "泣く", "了解", "はい", "＾＿＾"]
    self.vocab_size = len(self.vocab_list)
    self.vocab_file = test_utils.create_temporary_vocab_file(self.vocab_list)
    self.vocab_info = vocab.get_vocab_info(self.vocab_file.name)

    tf.contrib.framework.get_or_create_global_step()

    # self.encoder_rnn_cell = tf.contrib.rnn.LSTMCell(32)
    # self.decoder_rnn_cell = tf.contrib.rnn.LSTMCell(32)
    # self.attention_dim = 128

    self.example = self._create_example()
    print(self.example.source.shape, self.example.source_len)
    print(self.example.target.shape, self.example.target_len)
    print(self.example.labels, self.example.labels.shape)

    self.mode = tf.contrib.learn.ModeKeys.TRAIN
    # self.model = self.create_model(mode=self.mode)
    # print("Model {} created".format(self.model))

  def _create_example(self):
    """Creates example data for a test"""
    source = np.random.randn(self.batch_size, self.sequence_length,
                             self.input_depth)
    source_len = np.random.randint(0, self.sequence_length, [self.batch_size])
    target_len = np.random.randint(0, self.sequence_length * 2,
                                   [self.batch_size])
    target = np.random.randn(self.batch_size,
                             np.max(target_len), self.input_depth)
    labels = np.random.randint(0, self.vocab_size,
                               [self.batch_size, np.max(target_len) - 1])

    example_ = namedtuple(
        "Example", ["source", "source_len", "target", "target_len", "labels"])
    return example_(source, source_len, target, target_len, labels)

  def create_model(self, mode=None, params=None):
    params_ = AttentionSeq2Seq.default_params().copy()
    params_.update(TEST_PARAMS)
    params_.update({
        "source.reverse": True,
        "vocab_source": self.vocab_file.name,
        "vocab_target": self.vocab_file.name,
    })
    params_.update(params or {})
    return AttentionSeq2Seq(params=params_, mode=mode)

  def _test_pipeline(self, mode, params=None):
    """Helper function to test the full model pipeline.
    """
    # Create source and target example
    source_len = self.sequence_length + 5
    target_len = self.sequence_length + 10
    source = " ".join(np.random.choice(self.vocab_list, source_len))
    target = " ".join(np.random.choice(self.vocab_list, target_len))
    sources_file, targets_file = test_utils.create_temp_parallel_data(
        sources=[source], targets=[target])

    # Build model graph
    model = self.create_model(mode, params)
    print("Model {} created".format(model))

    input_pipeline_ = input_pipeline.ParallelTextInputPipeline(
        params={
            "source_files": [sources_file.name],
            "target_files": [targets_file.name]
        },
        mode=mode)
    input_fn = training_utils.create_input_fn(
        pipeline=input_pipeline_, batch_size=self.batch_size)
    features, labels = input_fn()

    # will return predictions, loss, train_op
    fetches = model(features, labels, None)  # call __call__ of BaseModel
    fetches = [_ for _ in fetches if _ is not None]

    # import pprint; import pdb; pdb.set_trace()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        # while True:
        fetches_ = sess.run(fetches)
        print("Feches = {}".format(fetches_))

    sources_file.close()
    targets_file.close()

    return model, fetches_

  def test_train(self):
    model, fetches_ = self._test_pipeline(tf.contrib.learn.ModeKeys.TRAIN)
    predictions_, loss_, _ = fetches_

    target_len = self.sequence_length + 10 + 2
    max_decode_length = model.params["target.max_seq_len"]
    expected_decode_len = np.minimum(target_len, max_decode_length)

    np.testing.assert_array_equal(predictions_["logits"].shape, [
        self.batch_size, expected_decode_len - 1,
        model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(predictions_["losses"].shape,
                                  [self.batch_size, expected_decode_len - 1])
    np.testing.assert_array_equal(predictions_["predicted_ids"].shape,
                                  [self.batch_size, expected_decode_len - 1])
    assert not(np.isnan(loss_))

  def test_infer(self):
    model, fetches_ = self._test_pipeline(tf.contrib.learn.ModeKeys.INFER)
    predictions_, = fetches_
    pred_len = predictions_["predicted_ids"].shape[1]

    print("Predictions:")
    print(predictions_)

    np.testing.assert_array_equal(predictions_["logits"].shape, [
        self.batch_size, pred_len, model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(predictions_["predicted_ids"].shape,
                                  [self.batch_size, pred_len])

  def test_infer_beam_search(self):
    self.batch_size = 1
    beam_width = 10
    model, fetches_ = self._test_pipeline(
        mode=tf.contrib.learn.ModeKeys.INFER,
        params={"inference.beam_search.beam_width": 10})
    predictions_, = fetches_
    pred_len = predictions_["predicted_ids"].shape[1]

    vocab_size = model.target_vocab_info.total_size
    np.testing.assert_array_equal(predictions_["predicted_ids"].shape,
                                  [1, pred_len, beam_width])
    np.testing.assert_array_equal(
        predictions_["beam_search_output.beam_parent_ids"].shape,
        [1, pred_len, beam_width])
    np.testing.assert_array_equal(
        predictions_["beam_search_output.scores"].shape,
        [1, pred_len, beam_width])
    np.testing.assert_array_equal(
        predictions_["beam_search_output.original_outputs.predicted_ids"].shape,
        [1, pred_len, beam_width])
    np.testing.assert_array_equal(
        predictions_["beam_search_output.original_outputs.logits"].shape,
        [1, pred_len, beam_width, vocab_size])

if __name__ == "__main__":
  TEST_PARAMS = yaml.load("""
  embedding.dim: 5
  encoder.params:
    rnn_cell:
      dropout_input_keep_prob: 0.8
      num_layers: 2
      residual_connections: True,
      cell_class: LSTMCell
      cell_params:
        num_units: 4
  decoder.params:
    rnn_cell:
      num_layers: 2
      cell_class: LSTMCell
      cell_params:
        num_units: 4
  """)
  endec = EncoderDecoderTest()
  # endec.test_train()
  # endec.test_infer()
  endec.test_infer_beam_search()