# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for input-related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import yaml

from seq2seq.data import input_pipeline
from seq2seq.test import utils as test_utils
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder


class TestInputPipelineDef(tf.test.TestCase):
  """Tests InputPipeline string definitions"""

  def test_without_extra_args(self):
    pipeline_def = yaml.load("""
      class: ParallelTextInputPipeline
      params:
        source_files: ["file1"]
        target_files: ["file2"]
        num_epochs: 1
        shuffle: True
    """)
    pipeline = input_pipeline.make_input_pipeline_from_def(
      pipeline_def, tf.contrib.learn.ModeKeys.TRAIN)
    self.assertIsInstance(pipeline, input_pipeline.ParallelTextInputPipeline)
    # pylint: disable=W0212
    self.assertEqual(pipeline.params["source_files"], ["file1"])
    self.assertEqual(pipeline.params["target_files"], ["file2"])
    self.assertEqual(pipeline.params["num_epochs"], 1)
    self.assertEqual(pipeline.params["shuffle"], True)

  def test_with_extra_args(self):
    pipeline_def = yaml.load("""
      class: ParallelTextInputPipeline
      params:
        source_files: ["file1"]
        target_files: ["file2"]
        num_epochs: 1
        shuffle: True
    """)
    pipeline = input_pipeline.make_input_pipeline_from_def(
      def_dict=pipeline_def,
      mode=tf.contrib.learn.ModeKeys.TRAIN,
      num_epochs=5,
      shuffle=False)
    self.assertIsInstance(pipeline, input_pipeline.ParallelTextInputPipeline)
    # pylint: disable=W0212
    self.assertEqual(pipeline.params["source_files"], ["file1"])
    self.assertEqual(pipeline.params["target_files"], ["file2"])
    self.assertEqual(pipeline.params["num_epochs"], 5)
    self.assertEqual(pipeline.params["shuffle"], False)


class TFRecordsInputPipelineTest(tf.test.TestCase):
  """
  Tests Data Provider operations.
  """

  def setUp(self):
    super(TFRecordsInputPipelineTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)

  def test_pipeline(self):
    tfrecords_file = test_utils.create_temp_tfrecords(
      sources=["Hello World . 笑"], targets=["Bye 泣"])
    tfrecords_file2 = test_utils.create_temp_tfrecords(
      sources=["Bye 泣"], targets=["Hello World . 笑"])

    pipeline = input_pipeline.TFRecordInputPipeline(
      params={
        "files": [tfrecords_file.name, tfrecords_file2.name],
        "source_field": "source",
        "target_field": "target",
        "num_epochs": 5,
        "shuffle": False
      },
      mode=tf.contrib.learn.ModeKeys.TRAIN)

    data_provider = pipeline.make_data_provider()

    features = pipeline.read_from_data_provider(data_provider)
    import pprint
    pprint.pprint(features)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        for _ in range(5):
          res = sess.run(features)
          pprint.pprint(res)

    self.assertEqual(res["source_len"], 5)
    self.assertEqual(res["target_len"], 4)
    np.testing.assert_array_equal(
      np.char.decode(res["source_tokens"].astype("S"), "utf-8"),
      ["Hello", "World", ".", "笑", "SEQUENCE_END"])
    np.testing.assert_array_equal(
      np.char.decode(res["target_tokens"].astype("S"), "utf-8"),
      ["SEQUENCE_START", "Bye", "泣", "SEQUENCE_END"])


class ParallelTextInputPipelineTest(tf.test.TestCase):
  """
  Tests Data Provider operations.
  """

  def setUp(self):
    super(ParallelTextInputPipelineTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)

  def test_pipeline(self):
    file_source, file_target = test_utils.create_temp_parallel_data(
      sources=["Hello World . 笑", "Hello"], targets=["Bye 泣", "Halo"])

    pipeline = input_pipeline.ParallelTextInputPipeline(
      params={
        "source_files": [file_source.name],
        "target_files": [file_target.name],
        "num_epochs": 5,
        "shuffle": False
      },
      mode=tf.contrib.learn.ModeKeys.TRAIN)

    data_provider = pipeline.make_data_provider()

    features = pipeline.read_from_data_provider(data_provider)
    import pprint
    pprint.pprint(features)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        for _ in range(5):
          res = sess.run(features)
          pprint.pprint(res)

    self.assertEqual(res["source_len"], 5)
    self.assertEqual(res["target_len"], 4)
    np.testing.assert_array_equal(
      np.char.decode(res["source_tokens"].astype("S"), "utf-8"),
      ["Hello", "World", ".", "笑", "SEQUENCE_END"])
    np.testing.assert_array_equal(
      np.char.decode(res["target_tokens"].astype("S"), "utf-8"),
      ["SEQUENCE_START", "Bye", "泣", "SEQUENCE_END"])


class SimpleTextFileInputPipelineTest(tf.test.TestCase):
  """
  Tests Data Provider operations.
  """

  def setUp(self):
    super(SimpleTextFileInputPipelineTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)

  def test_pipeline(self):

    pipeline = input_pipeline.SimpleTextFileInputPipeline(
      params={
        "files":
          ["/Users/hai/PycharmProjects/seq2seq/seq2seq/test/data.txt"],
        "num_epochs": 5,
        "shuffle": False
      },
      mode=tf.contrib.learn.ModeKeys.TRAIN)

    data_provider = pipeline.make_data_provider()

    features = pipeline.read_from_data_provider(data_provider)
    import pprint
    pprint.pprint(features)
    print()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        for _ in range(5):
          res = sess.run(features)
          pprint.pprint(res)


class ImageCaptioningInputPipelineTest(tf.test.TestCase):
  """
  Tests Data Provider operations.
  """

  def setUp(self):
    super(ImageCaptioningInputPipelineTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)

  def _EncodedFloatFeature(self, ndarray):
    return tf.train.Feature(float_list=tf.train.FloatList(
      value=ndarray.flatten().tolist()))

  def _EncodedInt64Feature(self, ndarray):
    return tf.train.Feature(int64_list=tf.train.Int64List(
      value=ndarray.flatten().tolist()))

  def _EncodedBytesFeature(self, tf_encoded):
    with self.test_session():
      encoded = tf_encoded.eval()

    def BytesList(value):
      return tf.train.BytesList(value=[value])

    return tf.train.Feature(bytes_list=BytesList(encoded))

  def _BytesFeature(self, ndarray):
    values = ndarray.flatten().tolist()
    for i in range(len(values)):
      values[i] = values[i].encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

  def _StringFeature(self, value):
    value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _Encoder(self, image, image_format):
    assert image_format in ['jpeg', 'png']
    if image_format == 'jpeg':
      tf_image = tf.constant(image, dtype=tf.uint8)
      return tf.image.encode_jpeg(tf_image)
    if image_format == 'png':
      tf_image = tf.constant(image, dtype=tf.uint8)
      return tf.image.encode_png(tf_image)

  def GenerateImage(self, image_format, image_shape):
    """Generates an image and an example containing the encoded image.

    Args:
      image_format: the encoding format of the image.
      image_shape: the shape of the image to generate.

    Returns:
      image: the generated image.
      example: a TF-example with a feature key 'image/encoded' set to the
        serialized image and a feature key 'image/format' set to the image
        encoding format ['jpeg', 'png'].
    """
    image = np.linspace(0, 17, num=18).reshape(image_shape).astype(np.uint8)
    tf_encoded = self._Encoder(image, image_format)
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': self._EncodedBytesFeature(tf_encoded),
      'image/format': self._StringFeature(image_format)
    }))

    return image, example.SerializeToString()

  def DecodeExample(self, serialized_example, item_handler, image_shape,
                    image_format):
    """Decodes the given serialized example with the specified item handler.

    Args:
      serialized_example: a serialized TF example string.
      item_handler: the item handler used to decode the image.
      image_shape: the shape of the image being decoded.
      image_format: the image format being decoded.

    Returns:
      the decoded image found in the serialized Example.
    """
    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features={
          'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
          'image/format': tf.FixedLenFeature(
            (), tf.string, default_value=image_format),
        },
        items_to_handlers={'image': item_handler}
      )
      [tf_image] = decoder.decode(serialized_example, ['image'])
      decoded_image = tf_image.eval()

    # We need to recast them here to avoid some issues with uint8.
    return decoded_image.astype(np.float32)

  def testDecodeExampleWithJpegEncoding(self):
    image_shape = (2, 3, 3)
    image, serialized_example = self.GenerateImage(
      image_format='jpeg',
      image_shape=image_shape)

    decoded_image = self.DecodeExample(
      serialized_example,
      tfexample_decoder.Image(),
      image_shape=image_shape,
      image_format='jpeg')

    # Need to use a tolerance of 1 because of noise in the jpeg encode/decode
    self.assertAllClose(image, decoded_image, atol=1.001)

  def testDecodeExampleWithPngEncoding(self):
    image_shape = (2, 3, 3)
    image, serialized_example = self.GenerateImage(
      image_format='png',
      image_shape=image_shape)

    decoded_image = self.DecodeExample(
      serialized_example,
      tfexample_decoder.Image(),
      image_shape=image_shape,
      image_format='png')

    self.assertAllClose(image, decoded_image, atol=0)

  def test_pipeline(self):
    tfrecords_file = test_utils.create_temp_image_captioning_tfrecords(
      img_data=[np.random.random([10, 20, 1])],
      img_formats=['jpg'],
      caption_ids=[1],
      caption_tokens=['This is a test image']
    )

    pipeline = input_pipeline.ImageCaptioningInputPipeline(
      params={
        "files": [tfrecords_file.name],
        # "image_field": "image/data",
        # "image_format": "jpg",
        # "caption_ids_field": "image/caption_ids",
        # "caption_tokens_field": "image/caption",
        "num_epochs": 5,
        "shuffle": False
      },
      mode=tf.contrib.learn.ModeKeys.TRAIN)

    data_provider = pipeline.make_data_provider()
    features = pipeline.read_from_data_provider(data_provider)
    import pprint
    pprint.pprint(features)

    # with self.test_session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   sess.run(tf.local_variables_initializer())
    #   with tf.contrib.slim.queues.QueueRunners(sess):
    #     # for _ in range(5):
    #     res = sess.run(features)
    #     pprint.pprint(res)

  def testDecodeExampleShapeKeyTensor(self):

    np_image = np.random.rand(2, 3, 1).astype('f')
    np_labels = np.array([[[1], [2], [3]],
                          [[4], [5], [6]]])

    example = tf.train.Example(features=tf.train.Features(feature={
      'image': self._EncodedFloatFeature(np_image),
      'image/shape': self._EncodedInt64Feature(np.array(np_image.shape)),
      'labels': self._EncodedInt64Feature(np_labels),
      'labels/shape': self._EncodedInt64Feature(np.array(np_labels.shape)),

    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
        'image': tf.VarLenFeature(dtype=tf.float32),
        'image/shape': tf.VarLenFeature(dtype=tf.int64),
        'labels': tf.VarLenFeature(dtype=tf.int64),
        'labels/shape': tf.VarLenFeature(dtype=tf.int64),
      }
      items_to_handlers = {
        'image': tfexample_decoder.Tensor('image',
                                          shape_keys='image/shape'),
        'labels': tfexample_decoder.Tensor('labels',
                                           shape_keys='labels/shape'),
      }
      decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

      [tf_image, tf_labels] = decoder.decode(serialized_example,
                                             ['image', 'labels'])
      self.assertAllEqual(tf_image.eval(), np_image)
      self.assertAllEqual(tf_labels.eval(), np_labels)


if __name__ == "__main__":
  tf.test.main()
