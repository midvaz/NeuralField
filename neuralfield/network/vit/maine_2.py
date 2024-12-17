import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.utils import plot_model

# Убедимся, что TensorFlow использует GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU доступен и будет использован для обучения.")
    except RuntimeError as e:
        print(f"Ошибка настройки памяти GPU: {e}")
else:
    print("GPU не обнаружен. Используется CPU.")

# Параметры датасета
DATASET_DIR = "dataset"  # Задайте путь к датасету
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
IMG_SIZE = 512  # Размер входного изображения
BATCH_SIZE = 32
NUM_CLASSES = 9  # Количество классов сегментации

# Патчинг изображения
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.projection = layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid"
        )
        self.flatten = layers.Reshape((-1, self.embed_dim))

    def call(self, inputs):
        x = self.projection(inputs)
        x = self.flatten(x)
        return x

# Трансформерный блок
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        mlp_output = self.mlp(out1, training=training)
        return self.norm2(out1 + mlp_output)

# Модель ViT для семантической сегментации
class VisionTransformer(Model):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, depth, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        self.position_embedding = self.add_weight(
            shape=(1, self.num_patches, embed_dim),
            initializer="random_normal",
            trainable=True
        )

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate) for _ in range(depth)
        ]

        self.decoder = tf.keras.Sequential([
            layers.Dense(embed_dim),
            layers.Reshape((img_size // patch_size, img_size // patch_size, embed_dim)),
            layers.Conv2DTranspose(filters=embed_dim // 2, kernel_size=2, strides=2, activation="relu"),
            layers.Conv2DTranspose(filters=embed_dim // 4, kernel_size=2, strides=2, activation="relu"),
            layers.Conv2D(num_classes, kernel_size=1, activation="softmax")
        ])

    def call(self, inputs):
        # Преобразование в патчи и добавление позиционной эмбеддинга
        x = self.patch_embedding(inputs)
        x += self.position_embedding

        # Пропуск через трансформерные блоки
        for block in self.transformer_blocks:
            x = block(x)

        # Декодер для восстановления пространственной размерности
        x = self.decoder(x)
        return x

# Параметры модели
PATCH_SIZE = 32 # Размер патча
EMBED_DIM = 512  # Увеличенная размерность эмбеддингов
DEPTH = 12  # Количество трансформерных блоков
NUM_HEADS = 16  # Голов для Multi-Head Attention
MLP_DIM = 1024  # Размерность скрытого слоя в MLP
DROPOUT_RATE = 0.1

# Создание модели
vit_segmentation_model = VisionTransformer(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    dropout_rate=DROPOUT_RATE
)

# Компиляция модели
vit_segmentation_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Вывод структуры модели
vit_segmentation_model.build((None, IMG_SIZE, IMG_SIZE, 3))
vit_segmentation_model.summary()

# Сохранение схемы модели
plot_model(vit_segmentation_model, to_file="vit_model_architecture.png", show_shapes=True)

# Функции для оценки
from sklearn.metrics import confusion_matrix, f1_score, jaccard_score

def evaluate_model(y_true, y_pred, num_classes):
    y_true_flat = y_true.flatten()
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(num_classes))
    iou = jaccard_score(y_true_flat, y_pred_flat, average="macro")
    f1 = f1_score(y_true_flat, y_pred_flat, average="macro")

    mean_accuracy = np.diag(cm).sum() / cm.sum()

    return mean_accuracy, iou, f1

# Генерация гистограммы результатов
def plot_metrics(metrics, labels):
    plt.bar(labels, metrics, color=['blue', 'green', 'red'])
    plt.xlabel("Метрики")
    plt.ylabel("Значения")
    plt.title("Оценка модели")
    plt.show()

# График обучения
class TrainingPlotCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.history = {"loss": [], "accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        self.history["loss"].append(logs.get("loss"))
        self.history["accuracy"].append(logs.get("accuracy"))
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["loss"], label="Loss")
        plt.title("Training Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["accuracy"], label="Accuracy")
        plt.title("Training Accuracy")
        plt.legend()

        plt.show()

# Функция для загрузки данных
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def process_path(image_path, mask_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    mask = load_img(mask_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img = img_to_array(img) / 255.0
    mask = img_to_array(mask).astype("int")
    return img, mask

def data_generator(image_dir, mask_dir, max_samples=2000):
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))
    count = 0  # Счётчик обработанных файлов

    for img_filename, mask_filename in zip(image_filenames, mask_filenames):
        if count >= max_samples:  # Проверка ограничения
            break
        img_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        yield process_path(img_path, mask_path)
        count += 1


def load_dataset(image_dir, mask_dir):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_dir, mask_dir),
        output_types=(tf.float32, tf.int32),
        output_shapes=((IMG_SIZE, IMG_SIZE, 3), (IMG_SIZE, IMG_SIZE, 1))
    )
    return dataset

train_dataset = load_dataset(os.path.join(TRAIN_DIR, "images"), os.path.join(TRAIN_DIR, "masks"))
val_dataset = load_dataset(os.path.join(VAL_DIR, "images"), os.path.join(VAL_DIR, "masks"))

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Обучение модели
history = vit_segmentation_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[TrainingPlotCallback()]
)

# Пример использования метрик на валидационном наборе
val_images, val_masks = next(iter(val_dataset))
val_predictions = vit_segmentation_model.predict(val_images)
metrics = evaluate_model(val_masks.numpy(), val_predictions, NUM_CLASSES)
plot_metrics(metrics, ["Mean Accuracy", "IoU", "F1-Score"])

print("Обучение завершено. Схема модели сохранена в 'vit_model_architecture.png'")
