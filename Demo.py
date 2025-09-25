Cell 1 — Install dependencies
# Install the ART library and ensure TF/Keras are present
!pip install -q adversarial-robustness-toolbox tensorflow matplotlib

Cell 2 — Imports & helper functions
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

# simple helper to show image + prediction
def show_image(img, title=''):
    plt.figure(figsize=(3,3))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


Cell 3 — Load MNIST (small, fast) and preprocess
# Load MNIST from Keras datasets (handwritten digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0,1] and add channel dim, ensure float32
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

# Convert labels to categorical for Keras
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print("Shapes:", x_train.shape, y_train_cat.shape, x_test.shape)


Cell 4 — Build a tiny CNN (fast to train)
def build_model():
    inputs = keras.Input(shape=(28,28,1))
    x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

Cell 5 — Train (or load small weights if you prefer)
# Train the model for a few epochs (keeps demo quick)
history = model.fit(x_train[:20000], y_train_cat[:20000], epochs=3, batch_size=128,
                    validation_data=(x_test[:2000], y_test_cat[:2000]), verbose=2)

# Evaluate
loss, acc = model.evaluate(x_test[:5000], y_test_cat[:5000], verbose=0)
print(f"Accuracy (sample): {acc:.3f}")

# Speaker note: if you want maximum reliability on stage, pretrain locally and upload saved weights,
# or run this earlier in the day and save the model in Colab.


Cell 6 — Wrap the model with ART KerasClassifier
# ART classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(0.0, 1.0))
print("ART classifier ready.")

Cell 7 — Pick an example image and show the model's prediction
idx = 7  # try other idx values: 0..9999
orig = x_test[idx:idx+1].astype('float32')
pred = np.argmax(classifier.predict(orig), axis=1)[0]
print("Original predicted label:", pred, "true label:", y_test[idx])
show_image(orig, title=f"Original (true={y_test[idx]}, pred={pred})")

Cell 8 — Craft FGSM adversarial example (fast gradient sign method)
# Create FGSM attack instance; eps controls perturbation size
attack = FastGradientMethod(estimator=classifier, eps=0.15)  # tune eps: 0.05..0.3
adv = attack.generate(x=orig)

# Show adversarial image and new prediction
adv_pred = np.argmax(classifier.predict(adv), axis=1)[0]
print("Adversarial predicted label:", adv_pred)
show_image(adv, title=f"Adversarial (pred={adv_pred})")

Cell 9 — Compare several examples and attack strengths
# Show a grid of original vs adversarial for multiple eps values
indices = [1, 5, 12]  # sample indices; change as you like
eps_values = [0.02, 0.08, 0.15, 0.25]

for idx in indices:
    orig = x_test[idx:idx+1].astype('float32')
    print(f"\n=== Example index {idx}, true label {y_test[idx]} ===")
    show_image(orig, title=f"Original (true={y_test[idx]})")
    for eps in eps_values:
        attack = FastGradientMethod(estimator=classifier, eps=eps)
        adv = attack.generate(x=orig)
        adv_pred = np.argmax(classifier.predict(adv), axis=1)[0]
        print(f" eps={eps:0.2f}  -> pred={adv_pred}")
        show_image(adv, title=f"eps={eps:.2f} pred={adv_pred}")

Cell 10 — Measure accuracy drop on a test subset
# Evaluate baseline accuracy
baseline_preds = np.argmax(classifier.predict(x_test[:1000].astype('float32')), axis=1)
baseline_acc = (baseline_preds == y_test[:1000]).mean()
print("Baseline accuracy (first 1000 test):", baseline_acc)

# Apply FGSM with eps=0.15 to whole subset and re-evaluate
attack = FastGradientMethod(estimator=classifier, eps=0.15)
x_test_adv = attack.generate(x=x_test[:1000].astype('float32'))
adv_preds = np.argmax(classifier.predict(x_test_adv), axis=1)
adv_acc = (adv_preds == y_test[:1000]).mean()
print("Adversarial accuracy (eps=0.15):", adv_acc)
print(f"Accuracy drop: {baseline_acc - adv_acc:.3f}")


