import tensorflow as tf

# Force legacy loading
from keras.saving import legacy_sm_saving_lib

model = legacy_sm_saving_lib.load_model("model/model.h5", compile=False)

# Save in new format
model.save("model/model_fixed.keras")

print("✅ Model converted successfully!")


# ── train_colab.ipynb ──────────────────────────────────────────────────────

# 1. Install & import
# !pip install tensorflow kaggle -q

# # 2. Download PlantVillage dataset from Kaggle
# # Upload your kaggle.json API key first, then:
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d emmarex/plantdisease -p /content/data --unzip

# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os, json

# # 3. Data generators
# DATA_DIR = "/content/data/PlantVillage"
# IMG_SIZE = (224, 224)
# BATCH    = 32

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     horizontal_flip=True,
#     zoom_range=0.2,
#     rotation_range=20
# )

# train_gen = datagen.flow_from_directory(
#     DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
#     subset="training", class_mode="categorical"
# )
# val_gen = datagen.flow_from_directory(
#     DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
#     subset="validation", class_mode="categorical"
# )

# # Save class names for Flask
# class_names = list(train_gen.class_indices.keys())
# with open("/content/class_names.json", "w") as f:
#     json.dump(class_names, f)

# NUM_CLASSES = len(class_names)
# print(f"Classes: {NUM_CLASSES}")

# # 4. Build model (MobileNetV2 transfer learning)
# base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
# base.trainable = False  # freeze base first

# model = models.Sequential([
#     base,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(256, activation="relu"),
#     layers.Dropout(0.4),
#     layers.Dense(NUM_CLASSES, activation="softmax")
# ])

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# # 5. Train — Phase 1 (frozen base)
# model.fit(train_gen, validation_data=val_gen, epochs=10)

# # 6. Fine-tune — unfreeze last 30 layers
# base.trainable = True
# for layer in base.layers[:-30]:
#     layer.trainable = False

# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
#               loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(train_gen, validation_data=val_gen, epochs=5)

# # 7. Save model
# model.save("/content/model.h5")
# print("✅ model.h5 saved!")

# # 8. Download files to your machine
# from google.colab import files
# files.download("/content/model.h5")
# files.download("/content/class_names.json")