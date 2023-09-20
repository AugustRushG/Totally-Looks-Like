import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import random
import math
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

def show_img(path):
    image_path = path
    img = mpimg.imread(image_path)
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

def read_csv_to_df(path):
    file_path =  path

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    return df

def load_and_preprocess_image(image_path, target_size=(245, 200), random_transform=False):
    # Open the image using Pillow (PIL)
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Rescale pixel values to [0, 1]
        rotation_range=0,      # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,  # Randomly shift image width by up to 20%
        height_shift_range=0.2, # Randomly shift image height by up to 20%
        shear_range=0.2,        # Shear transformations
        zoom_range=0.2,         # Randomly zoom in on images by up to 20%
        horizontal_flip=True,   # Randomly flip images horizontally
        fill_mode='nearest'     # Fill mode for newly created pixels
    )

    img = load_img(image_path, target_size = target_size)
    img = img_to_array(img,dtype='int32')

    if random_transform:
        img = train_datagen.random_transform(img)

    
    return img

def create_train_valid_dataset(random_transform,  train_pairing_df, num_dissimilar_pairs = 2000):
    # Create lists to store paired left and right images
    image_pairs_with_label = []

    # Iterate through the rows of the CSV file and load/preprocess the images
    for index, row in train_pairing_df.iterrows():
        # load and pair the similar image first
        left_image = load_and_preprocess_image(f"dataset/train/left/{row['left']}.jpg", random_transform = random_transform)
        right_image = load_and_preprocess_image(f"dataset/train/right/{row['right']}.jpg",random_transform = random_transform)
        image_pair_with_label = [[left_image,right_image],1.0]
        image_pairs_with_label.append(image_pair_with_label)
    
    num_dissimilar_pairs = num_dissimilar_pairs  # You may adjust this number
    for _ in range(num_dissimilar_pairs):
        left_idx = random.randint(0, len(train_pairing_df) - 1)
        right_idx = random.randint(0, len(train_pairing_df) - 1)

        # Ensure left and right images are not the same
        while left_idx == right_idx:
            right_idx = random.randint(0, len(train_pairing_df) - 1)

        left_image = load_and_preprocess_image(f"dataset/train/left/{train_pairing_df.iloc[left_idx]['left']}.jpg", random_transform=random_transform)
        right_image = load_and_preprocess_image(f"dataset/train/right/{train_pairing_df.iloc[right_idx]['right']}.jpg", random_transform=random_transform)
        image_pair_with_label = [[left_image, right_image], 0.0]  # Label 0 for dissimilar pair
        image_pairs_with_label.append(image_pair_with_label)

    # Shuffle the list to mix similar and dissimilar pairs
    random.shuffle(image_pairs_with_label)

    return image_pairs_with_label

def display_image_pair(image_pair):
    left_image, right_images = image_pair
    num_right_images = len(right_images)
    
    max_images_per_row = 10  # Maximum number of images to display in a single row

    num_rows = (num_right_images - 1) // max_images_per_row + 1  # Calculate the number of rows

    plt.figure(figsize=(15, 5 * num_rows))

    for i in range(num_rows):
        start_idx = i * max_images_per_row
        end_idx = min((i + 1) * max_images_per_row, num_right_images)
        
        row_right_images = right_images[start_idx:end_idx]

        plt.subplot(num_rows, max_images_per_row + 1, i * (max_images_per_row + 1) + 1)
        plt.imshow(left_image)
        plt.title("Left Image")
        plt.axis("off")

        for j, right_image in enumerate(row_right_images):
            plt.subplot(num_rows, max_images_per_row + 1, i * (max_images_per_row + 1) + j + 2)
            plt.imshow(right_image)
            plt.title(f"Right Image {start_idx + j + 1}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_test_case(model, test_candidates_df, row_number=10):
    for index, row in test_candidates_df.iterrows():
        if index >= row_number:
            break

        print(f"Row {index}")

        left_image = None
        right_images = []

        for column, value in row.items():
            print(f"  Column {column}: {value}")
            if column == 'left':
                left_image = load_and_preprocess_image(f"dataset/test/left/{value}.jpg")
            else:
                test_img_right = load_and_preprocess_image(f"dataset/test/right/{value}.jpg")
                right_images.append(test_img_right)

        # Convert to NumPy arrays
        left_image = np.array([left_image])
        right_images = np.array([right_images])

        # Run prediction
        try:
            similarity_scores = model.predict([left_image, right_images], verbose=0)[0]
            print("Similarity Scores:", similarity_scores)
        except Exception as e:
            print("An error occurred during prediction:", e)

        # Plot images
        num_rows = math.ceil((len(right_images[0]) + 1) / 10)
        fig, axes = plt.subplots(num_rows, 10, figsize=(20, 5 * num_rows))

        # Show left image
        if num_rows > 1:
            axes[0, 0].imshow(left_image[0])
            axes[0, 0].set_title("Left Image")
            axes[0, 0].axis('off')
        else:
            axes[0].imshow(left_image[0])
            axes[0].set_title("Left Image")
            axes[0].axis('off')

        # Show right images
        for i in range(len(right_images[0])):
            row_idx = (i + 1) // 10
            col_idx = (i + 1) % 10
            if num_rows > 1:
                axes[row_idx, col_idx].imshow(right_images[0][i])
                axes[row_idx, col_idx].set_title(f"Right {i+1}\nScore: {similarity_scores[i]:.2f}")
                axes[row_idx, col_idx].axis('off')
            else:
                axes[col_idx].imshow(right_images[0][i])
                axes[col_idx].set_title(f"Right {i+1}\nScore: {similarity_scores[i]:.2f}")
                axes[col_idx].axis('off')

        plt.show()

def extract_features(x,model):
    
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract features from the image
    features = model.predict(x,verbose = 0)

    return features

def visualize_feature_maps(features, number_of_feature_to_show=10, figsize=(10, 10), cmap='viridis'):
    num_features = features.shape[-1]
    num_rows = (num_features + 4) // 5  # Calculate the number of rows needed

    plt.figure(figsize=figsize)
    for i in range(number_of_feature_to_show):
        if i < num_features:
            plt.subplot(num_rows, 5, i + 1)
            feature_map = features[0, :, :, i]

            # Resize the feature map for better visualization
            resized_feature_map = cv2.resize(feature_map, (200, 200))
            plt.imshow(resized_feature_map, cmap=cmap, interpolation='nearest')
            plt.axis('off')
            plt.title(f'Feature {i + 1}')

    plt.suptitle(f'Feature Maps Visualization', fontsize=16)
    plt.tight_layout()
    plt.show()

def calculate_cosine_distance(left_image, right_image,model):
    left_image = extract_features(left_image,model)
    right_image = extract_features(right_image,model)
    return cosine_similarity([left_image.flatten()], [right_image.flatten()])[0][0]

def show_local_features(detector, image):
    img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    key_points, description = detector.detectAndCompute(img, None)
    img_keypoints = cv2.drawKeypoints(img, 
                                            key_points, 
                                            img, 
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw circles.
    rgb = cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.title('detector Interest Points')
    plt.imshow(rgb); plt.show()

def image_detect_and_compute(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img = cv2.cvtColor(img_name.astype(np.uint8), cv2.COLOR_RGB2BGR)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des
    
def draw_image_matches(detector, image_pair, distance_threshold = 0.75):
    """Draw ORB feature matches of the given two images."""

    img1, kp1, des1 = image_detect_and_compute(detector, image_pair[0])
    img2, kp2, des2 = image_detect_and_compute(detector, image_pair[1])
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < distance_threshold*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb),plt.show()