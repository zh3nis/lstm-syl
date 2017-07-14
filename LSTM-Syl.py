import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
import time
from pyphen import Pyphen
import sys
import pickle
import argparse
import copy


class SylConcatSmallConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.5
  learning_rate = 1.0
  init_scale = 0.05
  num_epochs = 50
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 300
  num_layers = 2
  keep_prob = 0.5

  # Syllable embedding hyperparameters
  syl_vocab_size = 0 # to be determined later
  syl_emb_dim = 50
  highway_size = 300
  max_word_len = 0   # to be determined later
    
  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later


class SylConcatMediumConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.5
  learning_rate = 1.0
  init_scale = 0.05
  num_epochs = 50
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 439
  num_layers = 2
  keep_prob = 0.5

  # Syllable embedding hyperparameters
  syl_vocab_size = 0 # to be determined later
  syl_emb_dim = 228
  highway_size = 781
  max_word_len = 0   # to be determined later
    
  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later


class SylSumSmallConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.5
  learning_rate = 1.0
  init_scale = 0.05
  num_epochs = 50
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 300
  num_layers = 2
  keep_prob = 0.5

  # Syllable embedding hyperparameters
  syl_vocab_size = 0 # to be determined later
  syl_emb_dim = 175
  highway_size = 175
  max_word_len = 0   # to be determined later
    
  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later


class SylSumMediumConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.5
  learning_rate = 1.0
  init_scale = 0.05
  num_epochs = 50
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 435
  num_layers = 2
  keep_prob = 0.5

  # Syllable embedding hyperparameters
  syl_vocab_size = 0 # to be determined later
  syl_emb_dim = 438
  highway_size = 1256
  max_word_len = 0   # to be determined later
    
  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later


class SylCNNSmallConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.5
  learning_rate = 1.0
  init_scale = 0.05
  num_epochs = 50
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 300
  num_layers = 2
  keep_prob = 0.5

  # Syllable embedding hyperparameters
  syl_vocab_size = 0 # to be determined later
  syl_emb_dim = 175
  filter_widths = list(range(1, 4))
  feat_per_width = 60
  cnn_output_dim = (1 + 2 + 3) * 60
  highway_size = cnn_output_dim
  max_word_len = 0   # to be determined later
    
  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later


class SylCNNMediumConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.5
  learning_rate = 1.0
  init_scale = 0.05
  num_epochs = 50
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 380
  num_layers = 2
  keep_prob = 0.5

  # Syllable embedding hyperparameters
  syl_vocab_size = 0 # to be determined later
  syl_emb_dim = 242
  filter_widths = list(range(1, 4))
  feat_per_width = 60
  cnn_output_dim = (1 + 2 + 3) * 60
  highway_size = cnn_output_dim
  max_word_len = 0   # to be determined later
    
  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later


def parse_args():
  '''Parse command line arguments'''
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', 
      default='concat', 
      help='a syllable-aware model. ' \
           'concat = Syl-Concat, sum = Syl-Sum, cnn = Syl-CNN')
  parser.add_argument(
      '--size',
      default='small',
      help='model size. small or medium')
  parser.add_argument(
      '--lang', 
      default='en_US', 
      help='a language which is supported by Pyphen')
  parser.add_argument(
      '--is_train', 
      default='1', 
      help='mode. 1 = training, 0 = evaluation')
  parser.add_argument(
      '--data_dir', 
      default='data/ptb', 
      help='data directory. ' \
           'Should have train.txt/valid.txt/test.txt with input data')
  parser.add_argument(
      '--save_dir', 
      default='saves', 
      help='saves directory')
  parser.add_argument(
      '--save_name', 
      default='Syl-Concat',
      help='prefix for filenames when saving data and model')
  parser.add_argument(
      '--eos', 
      default='<eos>',
      help='EOS marker')
  parser.add_argument(
      '--ssm', 
      default='0',
      help='sampled softmax. 1 = yes, 0 = no')
  parser.add_argument(
      '--verbose', 
      default='1',
      help='print intermediate results. 1 = yes, 0 = no')
  return parser.parse_args()


def read_data(args, config):
  '''read data sets, construct all needed structures and update the config'''
  if args.ssm == '1': config.ssm = 1
  
  hyphenator = Pyphen(lang=args.lang)

  def my_syllables(word):
    return hyphenator.inserted(word).split('-')

  if args.is_train == '1':
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    with open(
        os.path.join(args.save_dir, args.save_name + '-data.pkl'), 
        'wb') as data_file:
      word_data = open(
          os.path.join(args.data_dir, 'train.txt'), 'r').read() \
          .replace('\n', args.eos).split()
      words = list(set(word_data))
      
      syllables = set()
      word_lens_in_syl = []

      for word in words:
        syls = my_syllables(word)
        word_lens_in_syl.append(len(syls))
        for syl in syls:
          syllables.add(syl)

      syls_list = list(syllables)
      pickle.dump((word_data, words, word_lens_in_syl, syls_list), data_file)
  else:
    with open(
        os.path.join(args.save_dir, args.save_name + '-data.pkl'), 
        'rb') as data_file:
      word_data, words, word_lens_in_syl, syls_list = pickle.load(data_file)

  word_data_size, word_vocab_size = len(word_data), len(words)
  print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
  config.word_vocab_size = word_vocab_size
  config.num_sampled = int(word_vocab_size * 0.2)

  word_to_ix = { word:i for i,word in enumerate(words) }
  ix_to_word = { i:word for i,word in enumerate(words) }

  def get_word_raw_data(input_file):
    data = open(input_file, 'r').read().replace('\n', args.eos).split()
    return [word_to_ix[w] for w in data]

  train_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'train.txt'))
  valid_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'valid.txt'))
  test_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'test.txt'))

  syl_vocab_size = len(syls_list)
  max_word_len = int(np.percentile(word_lens_in_syl, 100))
  config.max_word_len = max_word_len
  print('data has %d unique syllables' % syl_vocab_size)
  print('max word length in syllables is set to', max_word_len)

  # a fake syllable for zero-padding
  zero_pad_syl = ' '
  syls_list.insert(0, zero_pad_syl)
  syl_vocab_size += 1
  config.syl_vocab_size = syl_vocab_size

  syl_to_ix = { syl:i for i,syl in enumerate(syls_list) }
  ix_to_syl = { i:syl for i,syl in enumerate(syls_list) }

  word_ix_to_syl_ixs = {}
  for word in words:
    word_ix = word_to_ix[word]
    word_in_syls = my_syllables(word)
    word_in_syls += [zero_pad_syl] * (max_word_len - len(word_in_syls))
    word_ix_to_syl_ixs[word_ix] = [syl_to_ix[syl] for syl in word_in_syls]

  return train_raw_data, valid_raw_data, test_raw_data, word_ix_to_syl_ixs


class batch_producer(object):
  '''Slice the raw data into batches'''
  def __init__(self, raw_data, batch_size, num_steps):
    self.raw_data = raw_data
    self.batch_size = batch_size
    self.num_steps = num_steps
    
    self.batch_len = len(self.raw_data) // self.batch_size
    self.data = np.reshape(self.raw_data[0 : self.batch_size * self.batch_len],
                           (self.batch_size, self.batch_len))
    
    self.epoch_size = (self.batch_len - 1) // self.num_steps
    self.i = 0
  
  def __next__(self):
    if self.i < self.epoch_size:
      # batch_x and batch_y are of shape [batch_size, num_steps]
      batch_x = self.data[::, self.i * self.num_steps : (self.i + 1) \
          * self.num_steps : ]
      batch_y = self.data[::, self.i * self.num_steps + 1 : (self.i + 1) \
          * self.num_steps + 1 : ]
      self.i += 1
      return (batch_x, batch_y)
    else:
      raise StopIteration()

  def __iter__(self):
    return self


class Model:
  '''Syllable-aware language model'''
  def __init__(self, config, model, need_reuse=False):
    # get hyperparameters
    batch_size = config.batch_size
    num_steps = config.num_steps
    self.max_word_len = max_word_len = config.max_word_len
    self.syl_emb_dim = syl_emb_dim = config.syl_emb_dim
    self.highway_size = highway_size = config.highway_size
    self.init_scale = init_scale = config.init_scale
    num_sampled = config.num_sampled
    syl_vocab_size = config.syl_vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    word_vocab_size = config.word_vocab_size
    keep_prob = config.keep_prob
    if model == 'cnn':
      self.filter_widths = config.filter_widths1
      self.feat_per_width = config.feat_per_width
      self.cnn_output_dim = config.cnn_output_dim

    # syllable embedding matrix
    self.syl_embedding = tf.get_variable(
        "syl_embedding", [syl_vocab_size, syl_emb_dim], dtype=tf.float32)
    
    # placeholders for training data and labels
    self.x = tf.placeholder(tf.int32, [batch_size, num_steps, max_word_len])
    self.y = tf.placeholder(tf.int32, [batch_size, num_steps])
    y_float = tf.cast(self.y, tf.float32)
    
    # we first embed syllables ...
    x_embedded = tf.nn.embedding_lookup(self.syl_embedding, self.x)
    
    # ... and then compose syllable vectors to get a word vector
    if model == 'concat':
      words_embedded_reshaped_proj = self.syl_concat(x_embedded, need_reuse)
    elif model == 'sum':
      words_embedded_reshaped_proj = self.syl_sum(x_embedded, need_reuse)
    elif model == 'cnn':
      words_embedded_reshaped_proj = self.syl_cnn(x_embedded, need_reuse)
    
    # we feed the word vector into HW layer(s) ...
    with tf.variable_scope('highway1'):
      highw_output = self.highway_layer(words_embedded_reshaped_proj)
    
    with tf.variable_scope('highway2'):
      highw_output = self.highway_layer(highw_output)
      
    highw_output_reshaped = tf.reshape(
        highw_output, [batch_size, num_steps, -1])
    
    # ... and then process it with a stack of two LSTMs
    lstm_input = tf.unstack(highw_output_reshaped, axis=1)
    # basic LSTM cell
    def lstm_cell():
      return tf.contrib.rnn.core_rnn_cell.LSTMCell(
          hidden_size, forget_bias=1.0, reuse=need_reuse)
    cells = []
    for i in range(num_layers):
      with tf.variable_scope('layer' + str(i)):
        if not need_reuse:
          if i == 0:
            cells.append(tf.contrib.rnn.DropoutWrapper(
                lstm_cell(), 
                output_keep_prob=keep_prob,
                input_size=highway_size,
                dtype=tf.float32))
          else:
            cells.append(tf.contrib.rnn.DropoutWrapper(
                lstm_cell(),
                output_keep_prob=keep_prob,
                input_size=hidden_size,
                dtype=tf.float32))
        else:
          cells.append(lstm_cell())
    self.cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cells)
    
    self.init_state = self.cell.zero_state(batch_size, dtype=tf.float32)
    #with tf.variable_scope('lstm_rnn', reuse=need_reuse):
    outputs, self.state = tf.contrib.rnn.static_rnn(
        self.cell, 
        lstm_input, 
        dtype=tf.float32, 
        initial_state=self.init_state)
    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])
      
    # finally we predict the next word according to a softmax normalization
    #with tf.variable_scope('softmax_params', reuse=need_reuse):
    weights = tf.get_variable(
        'weights', [word_vocab_size, hidden_size], dtype=tf.float32)      
    biases = tf.get_variable('biases', [word_vocab_size], dtype=tf.float32)
        
    # and compute the cross-entropy between labels and predictions
    if config.ssm == 1 and not need_reuse:
      loss = tf.nn.sampled_softmax_loss(
          weights, biases, 
          tf.reshape(y_float, [-1, 1]), output, num_sampled, word_vocab_size, 
          partition_strategy="div")
    else:
      logits = tf.matmul(output, tf.transpose(weights)) + biases
      loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
          [logits],
          [tf.reshape(self.y, [-1])],
          [tf.ones([batch_size * num_steps], dtype=tf.float32)])
    self.cost = tf.reduce_sum(loss) / batch_size
  
  def syl_concat(self, x_embedded, need_reuse):
    '''Syl-Concat'''
    x_in_syls = tf.unstack(x_embedded, axis=2)
    # concatenate syllable vectors to obtain word vectors
    words_embedded = tf.concat(axis=1, values=x_in_syls)
    words_embedded_reshaped = tf.reshape(words_embedded, 
        [-1, self.max_word_len * self.syl_emb_dim])
    # we project word vectors to match the dimensionality of 
    # the highway layer
    #with tf.variable_scope('projection', reuse=need_reuse):
    proj_w = tf.get_variable(
        'proj_w', 
        [self.max_word_len * self.syl_emb_dim, self.highway_size],
        dtype=tf.float32)
    words_embedded_reshaped_proj = tf.matmul(words_embedded_reshaped, proj_w)
    return words_embedded_reshaped_proj
  
  def syl_sum(self, x_embedded, need_reuse):
    words_embedded = tf.reduce_sum(x_embedded, axis=2)
    words_embedded_reshaped = tf.reshape(words_embedded, 
        [-1, self.syl_emb_dim])
    # we project word vectors to match the dimensionality of 
    # the highway layer (if needed)
    if self.syl_emb_dim != self.highway_size:
      #with tf.variable_scope('projection', reuse=need_reuse):
      proj_w = tf.get_variable(
          'proj_w', 
          [self.syl_emb_dim, self.highway_size],
          dtype=tf.float32)
      words_embedded_reshaped_proj = tf.matmul(words_embedded_reshaped, proj_w)
    else:
      words_embedded_reshaped_proj = words_embedded_reshaped
    return words_embedded_reshaped_proj

  def syl_cnn(self, x_embedded, need_reuse):
    words_embedded = tf.nn.embedding_lookup(self.syl_embedding, self.x)
    words_embedded_reshaped = tf.reshape(
        words_embedded, [-1, self.max_word_len, self.syl_emb_dim])
    
    def conv_layer(cur_syl_inputs, filt_shape, bias_shape):
      new_filt_shape = [1, 1] + filt_shape
      filt = tf.get_variable('filt', new_filt_shape)
      bias = tf.get_variable('bias', bias_shape)
      cur_syl_inputs = tf.expand_dims(tf.expand_dims(cur_syl_inputs, 1), 1)
      conv = tf.nn.conv3d(
          cur_syl_inputs, filt, [1, 1, 1, 1, 1], padding='VALID')
      feature_map = tf.nn.tanh(conv + bias)
      feature_map_reshaped = tf.squeeze(feature_map, axis=1)
      pool = tf.nn.max_pool(
          feature_map_reshaped, 
          [1, 1, self.max_word_len - filt_shape[0] + 1, 1], 
          [1, 1, 1, 1], 
          'VALID')
      return(tf.squeeze(pool, axis=[1,2]))
    
    def words_filter(cur_syl_inputs):
      pools = []
      for w in self.filter_widths:
        with tf.variable_scope('filter' + str(w)):
          pools.append(conv_layer(
              cur_syl_inputs, 
              [w, self.syl_emb_dim, w * self.feat_per_width], 
              [w * self.feat_per_width]))
      return tf.concat(pools, 1)
    
    #with tf.variable_scope('cnn_output', reuse=need_reuse) as scope:
    cnn_output = tf.reshape(
        words_filter(words_embedded_reshaped), [-1, self.cnn_output_dim])
    return cnn_output

  def highway_layer(self, highway_inputs):
    '''Highway layer'''
    transf_weights = tf.get_variable(
        'transf_weights', 
        [self.highway_size, self.highway_size],
        dtype=tf.float32)
    transf_biases = tf.get_variable(
        'transf_biases', 
        [self.highway_size],
        initializer=tf.random_uniform_initializer(-2-0.01, -2+0.01),
        dtype=tf.float32)
    highw_weights = tf.get_variable(
        'highw_weights', 
        [self.highway_size, self.highway_size],
        dtype=tf.float32)
    highw_biases = tf.get_variable(
        'highw_biases', 
        [self.highway_size],
        dtype=tf.float32)
    transf_gate = tf.nn.sigmoid(
        tf.matmul(highway_inputs, transf_weights) + transf_biases)
    highw_output = tf.multiply(
        transf_gate, 
        tf.nn.relu(
            tf.matmul(highway_inputs, highw_weights) + highw_biases)) \
        + tf.multiply(
        tf.ones([self.highway_size], dtype=tf.float32) - transf_gate, 
        highway_inputs)
    return highw_output


class Train(Model):
  '''for training we need to compute gradients'''
  def __init__(self, config, model):
    super(Train, self).__init__(config, model)
    self.clear_syl_embedding_padding = tf.scatter_update(
        self.syl_embedding, 
        [0], 
        tf.constant(0.0, shape=[1, config.syl_emb_dim], dtype=tf.float32))
    
    self.lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(self.cost, tvars), 
        config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    
    self.new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self.lr_update = tf.assign(self.lr, self.new_lr)

  # this will update the learning rate
  def assign_lr(self, session, lr_value):
    session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def model_size():
  '''finds the total number of trainable variables a.k.a. model size'''
  params = tf.trainable_variables()
  size = 0
  for x in params:
    sz = 1
    for dim in x.get_shape():
      sz *= dim.value
    size += sz
  return size


def run_epoch(sess, model, raw_data, config, is_train=False, lr=None):
  start_time = time.time()
  if is_train: model.assign_lr(sess, lr)

  iters = 0
  costs = 0
  state = sess.run(model.init_state)

  batches = batch_producer(raw_data, config.batch_size, config.num_steps)

  for batch in batches:
    my_x = np.empty(
        [config.batch_size, config.num_steps, config.max_word_len], 
        dtype=np.int32)

    # split words into syllables
    for t in range(config.num_steps):
      for i in range(config.batch_size):
        my_x[i, t] = word_ix_to_syl_ixs[batch[0][i, t]]

    # run the model on current batch
    if is_train:
      _, c, state = sess.run(
          [model.train_op, model.cost, model.state],
          feed_dict={model.x: my_x, model.y: batch[1], 
          model.init_state: state})
      sess.run(model.clear_syl_embedding_padding)
    else:
      c, state = sess.run([model.cost, model.state], 
          feed_dict={model.x: my_x, model.y: batch[1], 
          model.init_state: state})      

    costs += c
    step = iters // config.num_steps
    if is_train and args.verbose == '1' \
        and step % (batches.epoch_size // 10) == 10:
      print('%.3f' % (step * 1.0 / batches.epoch_size), end=' ')
      print('train ppl = %.3f' % np.exp(costs / iters), end=', ')
      print('speed =', 
          round(iters * config.batch_size / (time.time() - start_time)), 
          'wps')
    iters += config.num_steps
  
  return np.exp(costs / iters)


if __name__ == '__main__':
  args = parse_args()

  if args.model == 'concat':
    if args.size == 'small': config = SylConcatSmallConfig()
    elif args.size == 'medium': config = SylConcatMediumConfig()
  elif args.model == 'sum':
    if args.size == 'small': config = SylSumSmallConfig()
    elif args.size == 'medium': config = SylSumMediumConfig()
  elif args.model == 'cnn':
    if args.size == 'small': config = SylCNNSmallConfig()
    elif args.size == 'medium': config = SylCNNMediumConfig()
  else:
    print('Unknown model:', args.model)
    exit(0)

  initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale)    
    
  train_raw_data, valid_raw_data, test_raw_data, word_ix_to_syl_ixs \
      = read_data(args, config)

  with tf.variable_scope('Model', reuse=False, initializer=initializer):
    train = Train(config, model=args.model)
  print('Model size is: ', model_size())

  with tf.variable_scope('Model', reuse=True, initializer=initializer):
    valid = Model(config, model=args.model, need_reuse=True)

  test_config = copy.deepcopy(config)
  test_config.batch_size = 1
  test_config.ssm = 0
  with tf.variable_scope('Model', reuse=True, initializer=initializer):
    test = Model(test_config, model=args.model, need_reuse=True)

  saver = tf.train.Saver()

  if args.is_train == '1':
    '''Training and evaluation'''
    num_epochs = config.num_epochs
    init = tf.global_variables_initializer()
    learning_rate = config.learning_rate

    with tf.Session() as sess:
      sess.run(init)
      sess.run(train.clear_syl_embedding_padding)
      prev_valid_ppl = float('inf')

      for epoch in range(num_epochs):
        train_ppl = run_epoch(
            sess, train, train_raw_data, config, is_train=True, 
            lr=learning_rate)
        print('epoch', epoch + 1, end = ': ')
        print('train ppl = %.3f' % train_ppl, end=', ')
        print('lr = %.3f' % learning_rate, end=', ')

        # Get validation set perplexity
        valid_ppl = run_epoch(
            sess, valid, valid_raw_data, config, is_train=False)
        print('valid ppl = %.3f' % valid_ppl)
        
        # Update the learning rate if necessary
        if prev_valid_ppl - valid_ppl < 0:
          learning_rate *= config.lr_decay
        prev_valid_ppl = valid_ppl

      # Get test set perplexity after training is done
      test_ppl = run_epoch(
          sess, test, test_raw_data, test_config, is_train=False)
      print('Test set perplexity = %.3f' % test_ppl)

      save_path = saver.save(sess, os.path.join(
          args.save_dir, args.save_name + '-model.ckpt'))
      print('Model saved in file: %s' % save_path)

  else:
    '''Just evaluation of a trained model on test set'''
    with tf.Session() as sess:
      # Restore variables from disk.
      saver.restore(
          sess, os.path.join(args.save_dir, args.save_name + '-model.ckpt'))
      print('Model restored.')

      # Get test set perplexity
      test_ppl = run_epoch(
          sess, test, test_raw_data, test_config, is_train=False)
      print('Test set perplexity = %.3f' % test_ppl)
