import os
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.optimizers import Adam
from skimage import io, color, transform

def process_tile(tile, num_clusters):
    # Flatten the tile and normalize pixel values
    flattened_tile = tile.reshape((-1, tile.shape[-1]))
    flattened_tile = flattened_tile.astype(float) / 255.0

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels_tile = kmeans.fit_predict(flattened_tile)

    # Reshape the labels to the original tile shape
    segmented_tile = labels_tile.reshape(tile.shape[:-1])

    return segmented_tile

def create_model(input_shape, num_clusters):
    model = Sequential()
    model.add(Conv2D(num_clusters, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Reshape((input_shape[0] * input_shape[1], num_clusters)))
    return model

# Define tile size
tile_size = 512
num_clusters = 18  # Adjust based on your number of labels

# Folder containing TIFF files
folder_path = "path/to/your/folder"

# List all TIFF files in the folder
tif_files = [file for file in os.listdir(folder_path) if file.endswith('.tif')]

# Accumulate data from all TIFF files
all_input_data = []
all_target_data = []

for tif_file in tif_files:
    # Load the large image
    image_path = os.path.join(folder_path, tif_file)
    large_image = io.imread(image_path)

    # Process the image in tiles
    segmented_image = np.zeros_like(large_image[:, :, 0])  # Initialize segmented image

    for i in range(0, large_image.shape[0], tile_size):
        for j in range(0, large_image.shape[1], tile_size):
            # Extract a tile from the large image
            tile = large_image[i:i+tile_size, j:j+tile_size, :]
            
            # Apply segmentation on the tile
            segmented_tile = process_tile(tile, num_clusters)

            # Update the segmented image with the current tile
            segmented_image[i:i+tile_size, j:j+tile_size] = segmented_tile

    # Accumulate data
    input_data = large_image.reshape((-1, tile_size, tile_size, 3))
    target_data = segmented_image.reshape((-1, tile_size * tile_size))
    
    all_input_data.append(input_data)
    all_target_data.append(target_data)

# Combine data from all files
combined_input_data = np.concatenate(all_input_data, axis=0)
combined_target_data = np.concatenate(all_target_data, axis=0)

# Create a model and save its weights
model = create_model((tile_size, tile_size, 3), num_clusters)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the combined dataset
model.fit(combined_input_data, combined_target_data, epochs=5, batch_size=32)  # Adjust epochs and batch_size as needed

# Save model weights
model.save_weights("combined_model.h5")
