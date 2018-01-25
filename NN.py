class DNN(object):
	def __init__(self, name, units=[
			{'n_nodes': 10, 'activate': 'sigmoid'},
			{'n_nodes': 10, 'activate': 'sigmoid'},
			{'n_nodes': 10, 'activate': 'sigmoid'}],
			batch_size=10,
			learning_rate=1e-3,
			n_epochs=10,
			feature_range=[], minimum_error=1e-3):
		self.name = name
		self.units = units
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		# this is hidden layer
		self.layers = []

		self.graph = tf.Graph()
		with self.graph.as_default() as g:
			self.feature_range = tf.constant(feature_range, name='feature_range')
			self.minimum_error = tf.constant(minimum_error, name='minimum_error')
			self.sess = tf.Session()
			# self.writer = tf.summary.FileWriter(os.path.join('/tmp/tf_logs', self.name))

	def fit(self, x_train, y_train):
		with self.graph.as_default() as g:
			# preprocessing data
			if len(self.sess.run(self.feature_range)) == 2:
				x_train = scaler(data=x_train, feature_range=self.sess.run(self.feature_range))
			y_train = y_train.reshape(-1,1)

			self.dims = x_train.shape[1]

			# build model
			with tf.name_scope('Input'):
				self.input = tf.placeholder(shape=[None, self.dims], dtype=tf.float32, name='X')
				self.target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='Y')

			with tf.name_scope('Model'):
				with tf.name_scope('HL'):
					self.layers.append(
						{'weights': tf.Variable(tf.zeros(shape=[self.dims, self.units[0]['n_nodes']], dtype=tf.float32), name='W'),
						 'biases': tf.Variable(tf.zeros(shape=[self.units[0]['n_nodes']], dtype=tf.float32), name='B')}
					)
					prev_layer = tf.matmul(self.input, self.layers[0]['weights']) + self.layers[0]['biases']
					prev_layer = activate[self.units[0]['activate']](prev_layer)

				for _ in range(1, len(self.units)):
					with tf.name_scope('HL'):
						self.layers.append(
							{'weights': tf.Variable(tf.zeros(shape=[self.units[_-1]['n_nodes'], self.units[_]['n_nodes']], dtype=tf.float32), name='W'),
							 'biases': tf.Variable(tf.zeros(shape=[self.units[_]['n_nodes']], dtype=tf.float32), name='B')}
						)
						curr_layer = tf.matmul(prev_layer, self.layers[_]['weights']) + self.layers[_]['biases']
						curr_layer = activate[self.units[_]['activate']](curr_layer)
					prev_layer = curr_layer

				with tf.name_scope('Output'):
					self.output_layer = {
						'weights': tf.Variable(tf.zeros(shape=[self.units[-1]['n_nodes'], 1], dtype=tf.float32), name='W'),
						'biases': tf.Variable(tf.zeros(shape=[1], dtype=tf.float32), name='B')
					}
					self.output = tf.add(tf.matmul(prev_layer, self.output_layer['weights']), self.output_layer['biases'], name='output')

			with tf.name_scope('Train'):
				self.loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.output)
				tf.summary.scalar('Loss', self.loss)
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				self.train_ops = self.optimizer.minimize(self.loss, name='train_ops')

			with tf.name_scope('Init'):
				self.init = tf.global_variables_initializer()

			self.saver = tf.train.Saver()

			# train model
			batches = get_batch(self.batch_size, x_train.shape[0])
			merge_summary = tf.summary.merge_all()

			self.sess.run(self.init)
			for epoch in range(self.n_epochs):
				cost = 0
				for batch in batches:
					x_batch, y_batch = x_train[batch[0]:batch[1]], y_train[batch[0]:batch[1]]
					if epoch % 5 == 0:
						s = self.sess.run(merge_summary, feed_dict={self.input: x_batch, self.target: y_batch})
						self.writer.add_summary(s, epoch)
					_, c = self.sess.run([self.train_ops, self.loss], feed_dict={self.input: x_batch, self.target: y_batch})
					cost += c
				# print("Epoch {} loss = {}".format(epoch, cost))
				if cost <= self.sess.run(self.minimum_error):
					break

	def predict(self, x_pred):
		with self.graph.as_default() as g:
			# preprocessing data
			if len(self.sess.run(self.feature_range)) == 2:
				x_pred = scaler(data=x_pred, feature_range=self.sess.run(self.feature_range))

			y_pred = np.empty(shape=(0,1), dtype=np.float32)
			batches = get_batch(self.batch_size, x_pred.shape[0])
			for batch in batches:
				pred = self.sess.run(self.output, feed_dict={self.input: x_pred[batch[0]:batch[1]]})
				y_pred = np.concatenate((y_pred, pred), axis=0)
			y_pred = y_pred.reshape(-1)

			return y_pred

	def evaluate(self, x_test, y_test, scoring='rmse'):
		p = self.predict(x_pred=x_test)
		if scoring == 'rmse':
			e = metrics.mean_squared_error(y_test, p)
			return e**0.5
		elif scoring == 'mae':
			e = metrics.mean_absolute_error(y_test, p)
			return e

	def visualization(self):
		with self.graph.as_default() as g:
			self.writer.add_graph(self.sess.graph)
			self.writer.close()

	def __del__(self):
		with self.graph.as_default() as g:
			self.sess.close()
