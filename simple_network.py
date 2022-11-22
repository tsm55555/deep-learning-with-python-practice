'''
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss="sparse_categorical_crossentropy",
              metrics=['acc'])


model.fit(train_images, train_labels, epochs=10, batch_size=128)
'''
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
import time
class BatchGenerator:
    def __init__(self, images, labels, batch_size):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size) # how many batches we have
        self.index = 0 

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size # next batch
        return images, labels

# implemeting layers.Dense from stratch
class NaiveDense:
    def __init__(self, input_size, output_size, acitvation):
        self.acitvation = acitvation
        
        # weight
        w_shape = (input_size, output_size)
        w_init_val = tf.random.uniform(w_shape, minval=0, maxval=1e-1) # random intialization
        self.w = tf.Variable(w_init_val)

        # bias
        b_shape = (output_size, )
        b_init_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_init_value)

    def __call__(self, input):
        return self.acitvation(tf.matmul(input, self.w) + self.b)
    
    @property
    def weights(self):
        return [self.w, self.b]

# implemeting keras.Sequential from stratch     
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights

learning_rate = 0.003
def update_weights(gradient, weights):
    for g, w in zip(gradient, weights):
        w.assign_sub(g*learning_rate) #w -= g*learning_rate

def one_training_step(model, images, labels):
    with tf.GradientTape() as tape:
        pred = model(images)
        losses = (tf.keras.losses.sparse_categorical_crossentropy(labels, pred))
        avg_loss = tf.reduce_mean(losses)

    grad = tape.gradient(avg_loss, model.weights)
    update_weights(grad, model.weights)
    return avg_loss

def fit(model, train_images, train_labels, epochs, batch_size):
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        batch_generator = BatchGenerator(train_images, train_labels, batch_size)

        for batch in tqdm(range(batch_generator.num_batches)):
            images, labels = batch_generator.next()
            # print(images, labels)
            loss = one_training_step(model, images, labels)
            # if batch % 100 == 0:
            #     print(f'loss at batch {batch}: {loss:.2f}')

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

model = NaiveSequential([
    NaiveDense(input_size=28*28, output_size=512, acitvation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, acitvation=tf.nn.softmax)
])

start = time.time()
fit(model, train_images, train_labels, epochs=10, batch_size=128)
end = time.time()

pred = model(test_images)
pred = pred.numpy()
pred_labels = np.argmax(pred, axis=1)
matches = pred_labels == test_labels
print("training time: ", end - start)
print(f'acc: {matches.mean():.2f}')