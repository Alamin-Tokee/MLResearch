import os
import numpy as np
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import normalize
from tqdm import tqdm

# ============================
# 1. U-Net Model Definition
# ============================

def build_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)

    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, 3, activation='relu', padding='same')(u4)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(64, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================
# 2. GLCM Feature Extraction
# ============================

def extract_glcm_features(segmented_image):
    image = cv2.normalize(segmented_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    return {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0],
        'ASM': graycoprops(glcm, 'ASM')[0, 0]
    }

# ============================
# 3. Loop over Image Folder
# ============================

def process_folder(image_folder, model, save_csv='glcm_features.csv'):
    feature_list = []

    for img_name in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load & preprocess
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        norm_img = normalize(np.expand_dims(img, axis=-1), axis=1)
        input_img = np.expand_dims(norm_img, axis=0)

        # Predict segmentation
        pred = model.predict(input_img)[0, :, :, 0]
        mask = (pred > 0.5).astype(np.uint8)

        # GLCM on masked region
        masked_region = mask * img
        features = extract_glcm_features(masked_region)
        features['image_name'] = img_name
        feature_list.append(features)

    # Save features to CSV
    df = pd.DataFrame(feature_list)
    df.to_csv(save_csv, index=False)
    print(f"\n✅ GLCM features saved to: {save_csv}")

# ============================
# Run Everything
# ============================

# Set paths
image_folder = "/kaggle/input/brain-tumor/Brain Tumor/Brain Tumor"  # 🔁 Change this to your image directory

# Load or train U-Net
unet_model = build_unet()
# Optional: Load weights if trained
# unet_model.load_weights("your_weights.h5")

# Process all images in folder
process_folder(image_folder, unet_model)
