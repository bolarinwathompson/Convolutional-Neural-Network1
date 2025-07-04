# Convolutional Neural Network - Image Search Engine for ABC Grocery

## Project Overview:
The **ABC Grocery Image Search Engine** project utilizes a **Convolutional Neural Network (CNN)**, specifically the pre-trained **VGG16 model**, to create an image search system for ABC Grocery’s product catalog. The goal of this project is to find similar products based on a given query image, allowing customers to efficiently discover related products through visual search. By leveraging **transfer learning**, the VGG16 model extracts high-level feature vectors that represent the content of product images, enabling the system to compare and retrieve similar products.

## Objective:
The primary objective of this project is to build an **image retrieval system** that helps customers of ABC Grocery find visually similar products based on an image input. This system enhances the shopping experience by allowing users to search for products using images, thus improving user engagement and satisfaction.

## Key Features:
- **Feature Extraction with Pre-trained CNN**: The project uses the **VGG16 model**, a popular CNN pre-trained on a large image dataset (ImageNet), as a feature extractor. By excluding the top layers of VGG16 and utilizing its convolutional layers, we extract **512-dimensional feature vectors** from each product image in ABC Grocery's catalog.
- **Image Preprocessing**:
  - **Resizing**: Product images are resized to a consistent dimension of **224x224 pixels**, which is required by the VGG16 model.
  - **Normalization**: Pixel values are scaled to the range [0, 1] by dividing them by 255 to ensure consistent input for the CNN.
- **Feature Storage**: Feature vectors extracted from the images are stored for future use, allowing for efficient and quick similarity search operations when a query image is provided.
- **Similarity Search**: Using the feature vectors, the project implements a **nearest neighbor search** mechanism to find the most similar products to a given query image. The **cosine similarity** metric is used to measure the similarity between the feature vectors of images.

## Methods & Techniques:

### **1. Pre-trained Model (VGG16)**:
The **VGG16 model** is used as a feature extractor. The model is loaded without its top layers (i.e., excluding the fully connected layers), and the output of the final convolutional layer is used as the feature vector for each image.

### **2. Image Preprocessing**:
Product images are preprocessed before feeding them into the model:
- **Resizing**: Each image is resized to **224x224 pixels**.
- **Normalization**: The pixel values are scaled to the range of [0, 1] by dividing by 255 to normalize them for the CNN input.
- **Featurization**: After preprocessing, the images are passed through the VGG16 model to extract their feature vectors, which represent high-level information such as texture, shape, and color.

### **3. Feature Vector Storage**:
The feature vectors for each image are stored in a **pickle file** for future similarity searches. These vectors are essential for comparing query images with the product catalog to find visually similar items.

### **4. Nearest Neighbor Search**:
The **k-Nearest Neighbors (k-NN)** algorithm is used to find similar images. Once the feature vector of the query image is computed, it is compared with all other product images using the **cosine similarity metric**. The nearest neighbors are identified, and their corresponding product images are returned as search results.

### **5. Image Search Visualization**:
The system visualizes the search results by displaying the most similar images alongside their **cosine similarity distances**. This provides the user with a clear and intuitive way of exploring similar products.

## Technologies Used:
- **Python**: Programming language for implementing the image search engine.
- **TensorFlow/Keras**: For utilizing the **VGG16 model** and performing feature extraction.
- **NumPy**: For handling arrays and mathematical operations during the feature extraction and comparison process.
- **scikit-learn**: Implements the **Nearest Neighbors** algorithm for efficient image search based on feature vectors.
- **matplotlib**: Used for visualizing search results and displaying the most similar images with their similarity scores.
- **pickle**: For saving and loading the feature vectors and pre-trained model, making the system reusable and efficient.

## Key Results & Outcomes:
- The **Image Search Engine** successfully identifies and ranks similar products based on the input query image. By leveraging the **VGG16 CNN model**, the system captures intricate visual features and enables high-accuracy image retrieval.
- The system's performance is validated by testing the similarity search functionality, ensuring that visually similar products are returned first.
- **Cosine similarity** ensures the efficiency of the search, producing relevant results based on the similarity between feature vectors.

## Lessons Learned:
- **Transfer Learning** with pre-trained models like **VGG16** is highly effective for tasks such as image feature extraction, significantly reducing the need for training a model from scratch.
- **Feature representation** through CNNs captures intricate details that are vital for performing accurate image similarity searches.
- **Nearest Neighbors** is a simple yet powerful technique for searching through high-dimensional spaces, providing a scalable solution for content-based image retrieval.

## Future Enhancements:
- **Model Fine-tuning**: Fine-tuning the VGG16 model by adding custom layers on top of the pre-trained network could improve the accuracy of feature extraction specific to ABC Grocery’s product images.
- **Efficient Search Algorithms**: Implementing more efficient algorithms like **Approximate Nearest Neighbors** (ANN) could further improve search speed, especially for large product catalogs.
- **Search Expansion**: Future versions of the system could allow the user to filter search results by categories, price ranges, and other attributes to refine product recommendations.

