import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

BATCH_SIZE = 1024
EPOCHS = 40
LEARNING_RATE = 1E-3


def preprocess(dataset):
    image = tf.cast(dataset['image'], dtype=tf.float32) / 255
    label = tf.cast(dataset['label'], dtype=tf.float32)

    return image, label


# ===================== DEFINING CLASSIFICATION MODEL ======================= #

def simple_mlp_model(num_classes):
    input_ = tf.keras.layers.Input(shape=(28, 28, 1,))  # input layer, dimension of input matrix
    x = tf.keras.layers.Flatten()(input_)  # turns multidimensional structure into a vector
    # fully connected layer
    # all x input on neuron input + activation function
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.7)(x)  # 0.5 probability neuron is off while learning
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    # softmax - normalizing network outputs into probabilities
    # the probability that an object belongs to a class
    output_ = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.models.Model(input_, output_, name='Classifier')


# ===================== PREPARING TRAINING AND TEST DATASETS ======================= #

# load prepared data
(train_data, validation_data, test_data), metadata = tfds.load(
    'mnist',  # handwritten character set
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True
)
"""
    train - minimize loss function
    validation - compare models
    test - final accuracy
"""

print(metadata.features)

# tfds.visualization.show_examples(train_data, metadata)

# mix data + batch size
train_data = train_data.map(preprocess).shuffle(buffer_size=1024).batch(BATCH_SIZE)
validation_data = validation_data.map(preprocess).batch(BATCH_SIZE)
test_data = test_data.map(preprocess).batch(BATCH_SIZE)

num_classes = metadata.features['label'].num_classes
model = simple_mlp_model(num_classes)
model.summary()

# tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)

# ===================== TRAINING MODEL ======================= #

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    LEARNING_RATE,
    decay_steps=1e6,
    decay_rate=0.97
)

# train parameters
model.compile(
    # стохастичний градієнтний спуск
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),  # train algorithm
    # кросс-ентропія - міра інформаційної близькості двох розподілів (імовірності)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # loss function
    metrics=['accuracy']  # interested parameters
)

history = model.fit(train_data, epochs=EPOCHS, validation_data=validation_data)

# ===================== EVALUATION MODEL RESULTS ======================= #

metrics = model.evaluate(test_data, batch_size=BATCH_SIZE, verbose=1)
metric_names = ['Test Loss', 'Test Accuracy']

for name, value in zip(metric_names, metrics):
    print(f'{name}: {value}')

# ===================== CONFUSION MATRIX ======================= #

# prediction = np.argmax(model.predict(test_data), axis=1)
#
# ground_truth = np.concatenate([y for x, y in test_data], axis=0)
# matrix = tf.math.confusion_matrix(prediction, ground_truth)
#
# df_cm = pd.DataFrame(matrix.numpy(), range(num_classes), range(num_classes))
# plt.figure(figsize=(100, 100))
# svm = sn.heatmap(df_cm, annot=True)
# plt.show()

# ===================== PLOTTING TRAIN, TEST LOSSES AND ACCURACY ======================= #

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Train and Test Accuracy')
plt.legend()
plt.figure(figsize=(100, 100))
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.figure(figsize=(200, 200))
plt.show()
