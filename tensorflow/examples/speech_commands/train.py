# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio/simple_audio.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. Learn more at 
https://blog.research.google/2017/08/launching-speech-commands-dataset.html.

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
import argparse
import os.path
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

FLAGS = None
def load_data():
  df_yes=pd.DataFrame()
  df_no=pd.DataFrame()
  df_unknown=pd.DataFrame()
  df_silence=pd.DataFrame()
  file1="/kaggle/input/audiofile/yes_n.xlsx"
  file2="/kaggle/input/audiofile/yes1.xlsx"
  file3="/kaggle/input/audiofile/yes2.xlsx"
  file4="/kaggle/input/audiofile/yes3.xlsx"
  data=pd.read_excel(file1)
  df_yes=df_yes._append(data)
  data=pd.read_excel(file2)
  df_yes=df_yes._append(data)
  data=pd.read_excel(file3)
  df_yes=df_yes._append(data)
  data=pd.read_excel(file4)
  df_yes=df_yes._append(data)
  
  
  file1="/kaggle/input/audiofile/no.xlsx"
  file2="/kaggle/input/audiofile/no1.xlsx"
  file3="/kaggle/input/audiofile/no2.xlsx"
  file4="/kaggle/input/audiofile/no3.xlsx"
  data=pd.read_excel(file1)
  df_no=df_no._append(data)
  data=pd.read_excel(file2)
  df_no=df_no._append(data)
  data=pd.read_excel(file3)
  df_no=df_no._append(data)
  data=pd.read_excel(file4)
  df_no=df_no._append(data)

  file1="/kaggle/input/audiofile/un.xlsx"
  file2="/kaggle/input/audiofile/un1.xlsx"
  file3="/kaggle/input/audiofile/un2.xlsx"
  file4="/kaggle/input/audiofile/un3.xlsx"
  file5="/kaggle/input/audiofile/un4.xlsx"
  data=pd.read_excel(file1)
  df_unknown=df_unknown._append(data)
  data=pd.read_excel(file2)
  df_unknown=df_unknown._append(data)
  data=pd.read_excel(file3)
  df_unknown=df_unknown._append(data)
  data=pd.read_excel(file4)
  df_unknown=df_unknown._append(data)
  data=pd.read_excel(file5)
  df_unknown=df_unknown._append(data)

  file1="/kaggle/input/audiofile/silence.xlsx"
  data=pd.read_excel(file1)
  df_silence=df_silence._append(data)
  df_silence=df_silence._append(data)
  df_silence=df_silence._append(data)
  df_silence=df_silence._append(data)
  print("pulled_dataframes")
  return df_yes,df_no,df_unknown,df_silence


  

# def return_data(training_step,start):
#   if training_step<=140:
#     excel_file = "C:/Users/badal/Downloads/trainndata/yes_n.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     yes_data = np.array(reader.iloc[:, 2:1962].values)
#     yes_label = np.full(25, 2)
#     excel_file = "C:/Users/badal/Downloads/trainndata/no.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     no_data = np.array(reader.iloc[:, 2:1962].values)
#     no_label =  np.full(25, 3)
#     excel_file = "C:/Users/badal/Downloads/trainndata/un.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     un_data = np.array(reader.iloc[:, 2:1962].values)
#     un_label =  np.full(25, 1)
#     excel_file = "C:/Users/badal/Downloads/trainndata/silence.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=0, nrows=25)
#     silence_data = np.array(reader.iloc[:, 2:1962].values)
#     silence_label =  np.full(25, 0)
#     out1=np.concatenate((yes_data,no_data))
#     out2=np.concatenate((un_data,silence_data))
#     fdata=np.concatenate((out1,out2))
#     out1=np.concatenate((yes_label,no_label))
#     out2=np.concatenate((un_label,silence_label))
#     flabel=np.concatenate((out1,out2))
#     return fdata, flabel
#   elif training_step<=740:
#     excel_file = "C:/Users/badal/Downloads/trainndata/yes1.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     yes_data = np.array(reader.iloc[:, 2:1962].values)
#     yes_label =  np.full(25, 2)
#     excel_file = "C:/Users/badal/Downloads/trainndata/no1.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     no_data = np.array(reader.iloc[:, 2:1962].values)
#     no_label = np.full(25, 3)
#     excel_file = "C:/Users/badal/Downloads/trainndata/un2.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     un_data = np.array(reader.iloc[:, 2:1962].values)
#     un_label =  np.full(25, 1)
#     excel_file = "C:/Users/badal/Downloads/trainndata/silence.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=0, nrows=25)
#     silence_data = np.array(reader.iloc[:, 2:1962].values)
#     silence_label = np.full(25, 0)
#     out1=np.concatenate((yes_data,no_data))
#     out2=np.concatenate((un_data,silence_data))
#     fdata=np.concatenate((out1,out2))
#     out1=np.concatenate((yes_label,no_label))
#     out2=np.concatenate((un_label,silence_label))
#     flabel=np.concatenate((out1,out2))
#     return fdata, flabel
#   elif training_step<=1340:
#     excel_file = "C:/Users/badal/Downloads/trainndata/yes2.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     yes_data = np.array(reader.iloc[:, 2:1962].values)
#     yes_label = np.full(25, 2)
#     excel_file = "C:/Users/badal/Downloads/trainndata/no2.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     no_data = np.array(reader.iloc[:, 2:1962].values)
#     no_label = np.full(25, 3)
#     excel_file = "C:/Users/badal/Downloads/trainndata/un3.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     un_data = np.array(reader.iloc[:, 2:1962].values)
#     un_label = np.full(25, 1)
#     excel_file = "C:/Users/badal/Downloads/trainndata/silence.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=0, nrows=25)
#     silence_data = np.array(reader.iloc[:, 2:1962].values)
#     silence_label = np.full(25, 0)
#     out1=np.concatenate((yes_data,no_data))
#     out2=np.concatenate((un_data,silence_data))
#     fdata=np.concatenate((out1,out2))
#     out1=np.concatenate((yes_label,no_label))
#     out2=np.concatenate((un_label,silence_label))
#     flabel=np.concatenate((out1,out2))
#     return fdata, flabel
#   else:
#     excel_file = "C:/Users/badal/Downloads/trainndata/yes3.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     yes_data = np.array(reader.iloc[:, 2:1962].values)
#     yes_label =  np.full(25, 2)
#     excel_file = "C:/Users/badal/Downloads/trainndata/no3.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     no_data = np.array(reader.iloc[:, 2:1962].values)
#     no_label = np.full(25, 3)
#     excel_file = "C:/Users/badal/Downloads/trainndata/un4.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=start, nrows=25)
#     un_data = np.array(reader.iloc[:, 2:1962].values)
#     un_label = np.full(25, 1)
#     excel_file = "C:/Users/badal/Downloads/trainndata/silence.xlsx"
#     reader = pd.read_excel(excel_file, skiprows=0, nrows=25)
#     silence_data = np.array(reader.iloc[:, 2:1962].values)
#     silence_label =  np.full(25, 0)
#     out1=np.concatenate((yes_data,no_data))
#     out2=np.concatenate((un_data,silence_data))
#     fdata=np.concatenate((out1,out2))
#     out1=np.concatenate((yes_label,no_label))
#     out2=np.concatenate((un_label,silence_label))
#     flabel=np.concatenate((out1,out2))
#     return fdata, flabel

    




def main(_):
  # Set the verbosity based on flags (default is INFO, so we see all messages)
  tf.compat.v1.logging.set_verbosity(FLAGS.verbosity)

  # Start a new TensorFlow session.
  sess = tf.compat.v1.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir,
      FLAGS.silence_percentage, FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  input_placeholder = tf.compat.v1.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')
  if FLAGS.quantize:
    fingerprint_min, fingerprint_max = input_data.get_features_range(
        model_settings)
    fingerprint_input = tf.quantization.fake_quant_with_min_max_args(
        input_placeholder, fingerprint_min, fingerprint_max)
  else:
    fingerprint_input = input_placeholder

  logits, dropout_rate = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.compat.v1.placeholder(
      tf.int64, [None], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.compat.v1.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.compat.v1.name_scope('cross_entropy'):
    cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  if FLAGS.quantize:
    try:
      tf.contrib.quantize.create_training_graph(quant_delay=0)
    except AttributeError as e:
      msg = e.args[0]
      msg += ('\n\n The --quantize option still requires contrib, which is not '
              'part of TensorFlow 2.0. Please install a previous version:'
              '\n    `pip install tensorflow<=1.15`')
      e.args = (msg,)
      raise e

  with tf.compat.v1.name_scope('train'), tf.control_dependencies(
      control_dependencies):
    learning_rate_input = tf.compat.v1.placeholder(
        tf.float32, [], name='learning_rate_input')
    if FLAGS.optimizer == 'gradient_descent':
      train_step = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate_input).minimize(cross_entropy_mean)
    elif FLAGS.optimizer == 'momentum':
      train_step = tf.compat.v1.train.MomentumOptimizer(
          learning_rate_input, .9,
          use_nesterov=True).minimize(cross_entropy_mean)
    else:
      raise Exception('Invalid Optimizer')
  predicted_indices = tf.argmax(input=logits, axis=1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.math.confusion_matrix(labels=ground_truth_input,
                                              predictions=predicted_indices,
                                              num_classes=label_count)
  evaluation_step = tf.reduce_mean(input_tensor=tf.cast(correct_prediction,
                                                        tf.float32))
  with tf.compat.v1.get_default_graph().name_scope('eval'):
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)
    tf.compat.v1.summary.scalar('accuracy', evaluation_step)

  global_step = tf.compat.v1.train.get_or_create_global_step()
  increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)

  saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.compat.v1.summary.merge_all(scope='eval')
  train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                                 sess.graph)
  validation_writer = tf.compat.v1.summary.FileWriter(
      FLAGS.summaries_dir + '/validation')

  tf.compat.v1.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.compat.v1.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.io.write_graph(sess.graph_def, FLAGS.train_dir,
                    FLAGS.model_architecture + '.pbtxt')

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  # Training loop.
  start=0
  val=0
  print("pulling data")
  df_yes,df_no,df_unknown,df_silence=load_data()
  training_steps_max = np.sum(training_steps_list)
  for training_step in range(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
    # Pull the audio samples we'll use for training.
    
    

    values_df1 = np.array(df_yes.iloc[val:val+25,2:].values)
    values_df2 = np.array(df_no.iloc[val:val+25,2:].values)
    values_df3 = np.array(df_unknown.iloc[val:val+25,2:].values)
    values_df4 = np.array(df_silence.iloc[val:val+25,2:].values)
    val=val+25
    #train_fingerprints,train_ground_truth=return_data(training_step,start)
    train_fingerprints = np.concatenate((values_df1, values_df2, values_df3, values_df4))
    yes_label = np.full(25, 2)
    no_label=np.full(25,3)
    unknown_label=np.full(25,1)
    silence_label=np.full(25,0)
    train_ground_truth=((yes_label,no_label,unknown_label,silence_label))
    print("pulled the data "+str(start_step))
    


    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries,
            evaluation_step,
            cross_entropy_mean,
            train_step,
            increment_global_step,
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_rate: 0.5
        })
    train_writer.add_summary(train_summary, training_step)
    tf.compat.v1.logging.debug(
        'Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
        (training_step, learning_rate_value, train_accuracy * 100,
         cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      tf.compat.v1.logging.info(
          'Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
          (training_step, learning_rate_value, train_accuracy * 100,
           cross_entropy_value))
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in range(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_rate: 0.0
            })
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      tf.compat.v1.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.compat.v1.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                (training_step, total_accuracy * 100, set_size))

    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.compat.v1.logging.info('Saving to "%s-%d"', checkpoint_path,
                                training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)

  set_size = audio_processor.set_size('testing')
  tf.compat.v1.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in range(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_rate: 0.0
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.compat.v1.logging.warn('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.compat.v1.logging.warn('Final test accuracy = %.1f%% (N=%d)' %
                            (total_accuracy * 100, set_size))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is.',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between spectrogram timeslices.',
  )
  parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--quantize',
      type=bool,
      default=False,
      help='Whether to train the model for eight-bit deployment')
  parser.add_argument(
      '--preprocess',
      type=str,
      default='mfcc',
      help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')

  # Function used to parse --verbosity argument
  def verbosity_arg(value):
    """Parses verbosity argument.

    Args:
      value: A member of tf.logging.
    Raises:
      ArgumentTypeError: Not an expected value.
    """
    value = value.upper()
    if value == 'DEBUG':
      return tf.compat.v1.logging.DEBUG
    elif value == 'INFO':
      return tf.compat.v1.logging.INFO
    elif value == 'WARN':
      return tf.compat.v1.logging.WARN
    elif value == 'ERROR':
      return tf.compat.v1.logging.ERROR
    elif value == 'FATAL':
      return tf.compat.v1.logging.FATAL
    else:
      raise argparse.ArgumentTypeError('Not an expected value')
  parser.add_argument(
      '--verbosity',
      type=verbosity_arg,
      default=tf.compat.v1.logging.INFO,
      help='Log verbosity. Can be "DEBUG", "INFO", "WARN", "ERROR", or "FATAL"')
  parser.add_argument(
      '--optimizer',
      type=str,
      default='gradient_descent',
      help='Optimizer (gradient_descent or momentum)')

  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
