import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout, BatchNormalization)
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# SEED SETUP
# ----------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------------------------------
# TRANSFER LEARNING MODEL
# ----------------------------------------------------
def create_transfer_model(input_shape=(224, 224, 3), num_classes=3, unfreeze_layers=30):
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# ----------------------------------------------------
# OPTIMIZER WITH COSINE DECAY
# ----------------------------------------------------
def get_optimizer(optimizer_name):
    initial_lr = 1e-4

    if optimizer_name == "RMSprop":
        optimizer = RMSprop(learning_rate=initial_lr, epsilon=1e-7)
    elif optimizer_name == "SGD":
        optimizer = SGD(learning_rate=initial_lr, momentum=0.9, nesterov=True)
    elif optimizer_name == "Adam":
        optimizer = Adam(learning_rate=initial_lr, epsilon=1e-7)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Optimizer: {optimizer_name}, Learning Rate: {optimizer.learning_rate.numpy()}")  
    return optimizer
    
# ----------------------------------------------------
# TRAIN/EVAL FUNCTION 
# ----------------------------------------------------
def run_experiment(optimizer_name, train_generator, val_generator, test_generator, input_shape, num_classes, epochs, unfreeze_layers=30):
    tf.keras.backend.clear_session()
    
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    model = create_transfer_model(input_shape=input_shape, num_classes=num_classes, unfreeze_layers=unfreeze_layers)

    optimizer = get_optimizer(optimizer_name)
    print(f"Menjalankan eksperimen dengan optimizer: {optimizer_name}, Learning Rate: {optimizer.learning_rate.numpy()}")

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(f"best_model_{optimizer_name}.keras", monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]  

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    plot_training_history(history, optimizer_name)

    test_accuracy = model.evaluate(test_generator, verbose=0)
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    # Classification report
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    print(f"\nClassification Report ({optimizer_name}):")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, optimizer_name)

    # Aggregated metrics
    accuracy = test_accuracy * 100
    precision = np.mean([report_dict[c]['precision'] for c in class_names]) * 100
    recall    = np.mean([report_dict[c]['recall']    for c in class_names]) * 100
    f1_score  = np.mean([report_dict[c]['f1-score']  for c in class_names]) * 100

    metrics = {
        'Accuracy':  accuracy,
        'Precision': precision,
        'Recall':    recall,
        'F1-Score':  f1_score
    }

    return metrics, model

# ----------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(points, factor=0.8):
    """Smooth the curve by applying exponential moving average."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_training_history(history, optimizer_name, smoothing_factor=0.8):
    plt.figure(figsize=(12, 4))

    n_epochs    = len(history.history['accuracy'])
    epochs      = range(1, n_epochs + 1)
    epoch_ticks = np.arange(0, n_epochs + 1, 20)

    # Smooth curves
    train_acc  = smooth_curve(history.history['accuracy'],  factor=smoothing_factor)
    val_acc    = smooth_curve(history.history['val_accuracy'],factor=smoothing_factor)
    train_loss = smooth_curve(history.history['loss'],      factor=smoothing_factor)
    val_loss   = smooth_curve(history.history['val_loss'],  factor=smoothing_factor)

    # --- ACCURACY PLOT ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy (Smoothed)')
    plt.plot(epochs, val_acc,   label='Validation Accuracy (Smoothed)')
    plt.title(f'Accuracy ({optimizer_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # determine top/bottom and ticks
    max_acc = max(train_acc + val_acc)
    top_acc = 0.9 if max_acc < 0.9 else 1.0

    min_acc = min(train_acc + val_acc)
    bot_acc = np.floor(min_acc * 10) / 10  

    plt.xticks(epoch_ticks)
    plt.xlim(-5, n_epochs + 5)

    plt.ylim(bot_acc, top_acc)
    plt.yticks(np.arange(bot_acc, top_acc + 1e-6, 0.1))

    plt.legend()

    # --- LOSS PLOT ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss (Smoothed)')
    plt.plot(epochs, val_loss,   label='Validation Loss (Smoothed)')
    plt.title(f'Loss ({optimizer_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # determine top/bottom and ticks
    max_loss     = max(train_loss + val_loss)
    top_loss     = np.ceil(max_loss * 5) / 5    # e.g. 1.3539 → 1.4

    min_loss     = min(train_loss + val_loss)
    bot_loss     = 0.0 if min_loss < 0.2 else 0.2

    plt.xticks(epoch_ticks)
    plt.xlim(-5, n_epochs + 5)

    plt.ylim(bot_loss, top_loss)
    plt.yticks(np.arange(bot_loss, top_loss + 1e-6, 0.2))

    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, optimizer_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({optimizer_name})')
    plt.show()

# ----------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------
if __name__ == "__main__":
    # Directories
    train_dir = "archive_replicated_result/train"
    val_dir   = "archive_replicated_result/val"
    test_dir  = "archive_replicated_result/test"

    # Hyperparameters (fixed constraints)
    input_shape = (224, 224, 3)
    batch_size  = 32 
    epochs      = 100
    num_classes = 3
    unfreeze_layers = 30

    # Data Generators
    datagen_train = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.6, 1.0],
        horizontal_flip=True,
        vertical_flip=False, 
        fill_mode='nearest'
    )

    datagen_val_test = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = datagen_train.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )

    val_generator = datagen_val_test.flow_from_directory(
        val_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = datagen_val_test.flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    optimizers = ["RMSprop", "Adam", "SGD"]
    results = {}


    # Run Experiments
    for opt_name in optimizers:
        print(f"\nRunning partial-finetuning with {opt_name} optimizer...")
        metrics, model = run_experiment(
            optimizer_name   = opt_name,
            train_generator  = train_generator,
            val_generator    = val_generator,
            test_generator   = test_generator,
            input_shape      = input_shape,
            num_classes      = num_classes,
            epochs           = epochs,
            unfreeze_layers  = unfreeze_layers
        )
        results[opt_name] = metrics

        # Save model
        model.save(f"MobileNetV3-Small_finetune_{opt_name}.h5")
        print(f"Saved MobileNetV3-Small_finetune_{opt_name}.h5\n")

    # Final Results
    print("Final Results:")
    headers = ["Optimizer", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    table_data = []
    for opt_name, m in results.items():
        row = [
            opt_name,
            f"{m['Accuracy']:.2f}",
            f"{m['Precision']:.2f}",
            f"{m['Recall']:.2f}",
            f"{m['F1-Score']:.2f}"
        ]
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Best Optimizer
    best_optimizer = max(results, key=lambda x: results[x]['F1-Score'])
    print(f"\nBest optimizer is {best_optimizer} with F1-Score: {results[best_optimizer]['F1-Score']:.2f}%")