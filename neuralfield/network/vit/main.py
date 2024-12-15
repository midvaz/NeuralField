import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np
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
IMG_SIZE = 512  # Размер входного изображения (512x512), специфичный для Agriculture-Vision
PATCH_SIZE = 32 # Размер патча, оптимальный для больших изображений
NUM_CLASSES = 9  # Количество классов сегментации в Agriculture-Vision
EMBED_DIM = 512  # Увеличенная размерность эмбеддингов для сложных данных
DEPTH = 12  # Увеличенное количество трансформерных блоков
NUM_HEADS = 16  # Больше голов для Multi-Head Attention
MLP_DIM = 1024  # Увеличенная размерность скрытого слоя в MLP
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

# Пример использования (замените на свои данные)
# y_true и y_pred должны быть массивами numpy одинаковой формы
# y_true = np.array([...])
# y_pred = np.array([...])
# metrics = evaluate_model(y_true, y_pred, NUM_CLASSES)
# plot_metrics(metrics, ["Mean Accuracy", "IoU", "F1-Score"])
