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
"""Various testing utilities
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import tensorflow as tf


def create_temp_parallel_data(sources, targets):
  """
  Creates a temporary TFRecords file.

  Args:
    source: List of source sentences
    target: List of target sentences

  Returns:
    A tuple (sources_file, targets_file).
  """
  file_source = tempfile.NamedTemporaryFile()
  file_target = tempfile.NamedTemporaryFile()
  file_source.write("\n".join(sources).encode("utf-8"))
  file_source.flush()
  file_target.write("\n".join(targets).encode("utf-8"))
  file_target.flush()
  # print("source tmp file created: {}, target tmp file created: {}".
  #       format(file_source.name, file_target.name))
  return file_source, file_target


def create_temp_data(sources):
  """
  Creates a temporary TFRecords file.

  Args:
    source: List of source sentences
    target: List of target sentences

  Returns:
    A tuple (sources_file, targets_file).
  """
  file_source = tempfile.NamedTemporaryFile()
  file_source.write("\n".join(sources).encode("utf-8"))
  file_source.flush()
  # print("source tmp file created: {}, target tmp file created: {}".
  #       format(file_source.name, file_target.name))
  return file_source


def create_temp_tfrecords(sources, targets):
  """
  Creates a temporary TFRecords file.

  Args:
    source: List of source sentences
    target: List of target sentences

  Returns:
    A tuple (sources_file, targets_file).
  """

  output_file = tempfile.NamedTemporaryFile()
  writer = tf.python_io.TFRecordWriter(output_file.name)
  for source, target in zip(sources, targets):
    ex = tf.train.Example()
    #pylint: disable=E1101
    ex.features.feature["source"].bytes_list.value.extend(
        [source.encode("utf-8")])
    ex.features.feature["target"].bytes_list.value.extend(
        [target.encode("utf-8")])
    writer.write(ex.SerializeToString())
    print(ex)
  writer.close()

  return output_file


def create_temp_image_captioning_tfrecords(img_data, img_formats, caption_ids,
                                           caption_tokens):
  output_file = tempfile.NamedTemporaryFile()
  writer = tf.python_io.TFRecordWriter(output_file.name)

  for data, format, id, tokens in zip(img_data, img_formats, caption_ids,
                                      caption_tokens):
    ex = tf.train.Example()
    # pylint: disable=E1101
    ex.features.feature["image/data"].bytes_list.value.extend(
        [data.tobytes()])
    ex.features.feature["images/format"].bytes_list.value.extend(
        [format.encode("utf-8")])
    ex.features.feature["image/caption_ids"].bytes_list.value.extend(
        [str(id).encode("utf-8")])
    ex.features.feature["image/caption"].bytes_list.value.extend(
        [tokens.encode("utf-8")])
    writer.write(ex.SerializeToString())
    print(ex)
  writer.close()

  return output_file


def create_temporary_vocab_file(words, counts=None):
  """
  Creates a temporary vocabulary file.

  Args:
    words: List of words in the vocabulary

  Returns:
    A temporary file object with one word per line
  """
  vocab_file = tempfile.NamedTemporaryFile()
  if counts is None:
    for token in words:
      vocab_file.write((token + "\n").encode("utf-8"))
  else:
    for token, count in zip(words, counts):
      vocab_file.write("{}\t{}\n".format(token, count).encode("utf-8"))
  vocab_file.flush()
  return vocab_file
