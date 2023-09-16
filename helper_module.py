import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

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


def create_train_valid_dataset(random_transform, num_dissimilar_pairs = train_pairing_df.size):
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



def display_image_pairs(image_pairs, num_pairs_to_display=5):
    plt.figure(figsize=(12, 6))

    for i in range(num_pairs_to_display):
        left_image = image_pairs[i][0][0]
        right_image = image_pairs[i][0][1]
        label = image_pairs[i][1]

        # Choose a title based on similarity label
        if label == 1:
            title = "Similar Pair"
        else:
            title = "Dissimilar Pair"

        # Display the left image
        plt.subplot(2, num_pairs_to_display, i + 1)
        plt.imshow(left_image)
        plt.title(title)
        plt.axis("off")

        # Display the right image
        plt.subplot(2, num_pairs_to_display, num_pairs_to_display + i + 1)
        plt.imshow(right_image)
        plt.title(f"Label: {label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    
