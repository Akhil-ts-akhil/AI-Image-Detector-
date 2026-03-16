import os
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


DATASET_DIR = "dataset"
MODEL_PATH = "model.h5"
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


if not os.path.exists(MODEL_PATH):
    print("Training model...")

    img_size = 224
    batch_size = 32

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2)

    train_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save(MODEL_PATH)
    print("Model trained and saved!")

else:
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""
    img_path = ""

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = load_img(filepath, target_size=(224,224))
        img = img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)


        prediction = model.predict(img,verbose=0)[0][0]

        if prediction > 0.5:
            result = "Original Image"
            confidence = round(prediction*100,2)
        else:
            result = "AI Generated Image"
            confidence = round((1-prediction)*100,2)

        img_path = filepath

    return render_template("index.html", result=result, confidence=confidence, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)