"""
Author: Haripreeth Avarur
Date: 12/09/2024
Version: 2
Contact: hari.avarur@gmail.com
Description: Code for grouping products based on ResNet feature extraction and SSIM clustering using HDBSCAN.
"""


###-----IMPORTS-----###
import torch
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity



###-----FUNCTIONS-----###

#Loading a pretrained ResNet50
def load_resnet_model():
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

#Extracting ResNet features from product images
def extract_resnet_features(model, image_paths):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = model(image)
        features.append(feature.flatten().numpy())
    return np.vstack(features)

#Computing SSIM between two images
def compute_ssim(imageA, imageB, size=(224, 224)):
    imageA_resized = cv2.resize(imageA, size)
    imageB_resized = cv2.resize(imageB, size)
    grayA = cv2.cvtColor(imageA_resized, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB_resized, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

#SSIM matrix computation
def compute_ssim_matrix(image_paths):
    images = [cv2.imread(img_path) for img_path in image_paths]
    n = len(images)
    ssim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            ssim_matrix[i, j] = compute_ssim(images[i], images[j])
            ssim_matrix[j, i] = ssim_matrix[i, j]

    return ssim_matrix

#ResNet + SSIM 
def combine_features_and_ssim(resnet_features, ssim_matrix, weight_ssim=0.5):
    resnet_similarity = cosine_similarity(resnet_features)
    combined_similarity = (1 - weight_ssim) * resnet_similarity + weight_ssim * ssim_matrix
    return combined_similarity

#HDBSCAN Clustering
def cluster_with_hdbscan(combined_similarity):
    distance_matrix = 1 - combined_similarity
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2)
    labels = clusterer.fit_predict(distance_matrix)
    return labels

#Image Processing
def process_image_and_group(image_paths, weight_ssim=0.5):
    model = load_resnet_model()
    resnet_features = extract_resnet_features(model, image_paths)
    ssim_matrix = compute_ssim_matrix(image_paths)
    combined_similarity = combine_features_and_ssim(resnet_features, ssim_matrix, weight_ssim)
    group_ids = cluster_with_hdbscan(combined_similarity)
    return group_ids




# import torch
# from torchvision.models import resnet50
# import torchvision.transforms as transforms
# from PIL import Image
# from skimage.metrics import structural_similarity as ssim
# import cv2
# import numpy as np
# import hdbscan
# from sklearn.metrics.pairwise import cosine_similarity

# # Load pretrained ResNet50
# def load_resnet_model():
#     model = resnet50(pretrained=True)
#     model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
#     model.eval()
#     return model

# # Transform image for ResNet
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Extract ResNet features from product images
# def extract_resnet_features(model, image_paths):
#     features = []
#     for image_path in image_paths:
#         image = Image.open(image_path).convert('RGB')
#         image = transform(image).unsqueeze(0)  # Add batch dimension
#         with torch.no_grad():
#             feature = model(image)
#         features.append(feature.flatten().numpy())
#     return np.vstack(features)

# # Modify this function to resize images to the same size
# def compute_ssim(imageA, imageB, size=(224, 224)):
#     # Resize both images to the same size (224x224)
#     imageA_resized = cv2.resize(imageA, size)
#     imageB_resized = cv2.resize(imageB, size)
    
#     # Convert to grayscale
#     grayA = cv2.cvtColor(imageA_resized, cv2.COLOR_BGR2GRAY)
#     grayB = cv2.cvtColor(imageB_resized, cv2.COLOR_BGR2GRAY)
    
#     # Compute SSIM between the resized grayscale images
#     score, _ = ssim(grayA, grayB, full=True)
#     return score


# # Compute SSIM matrix for all product image pairs
# def compute_ssim_matrix(image_paths):
#     images = [cv2.imread(img_path) for img_path in image_paths]
#     n = len(images)
#     ssim_matrix = np.zeros((n, n))

#     for i in range(n):
#         for j in range(i, n):
#             ssim_matrix[i, j] = compute_ssim(images[i], images[j])  # Resizing happens in compute_ssim
#             ssim_matrix[j, i] = ssim_matrix[i, j]  # Symmetric matrix

#     return ssim_matrix


# # Combine ResNet features and SSIM into a similarity matrix
# def combine_features_and_ssim(resnet_features, ssim_matrix, weight_ssim=0.5):
#     resnet_similarity = cosine_similarity(resnet_features)
#     combined_similarity = (1 - weight_ssim) * resnet_similarity + weight_ssim * ssim_matrix
#     return combined_similarity

# # Perform clustering using HDBSCAN
# def cluster_with_hdbscan(combined_similarity):
#     distance_matrix = 1 - combined_similarity  # Convert similarity to distance
#     clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2)
#     labels = clusterer.fit_predict(distance_matrix)
#     return labels

# # Process images for clustering
# def process_image_and_group(image_paths, weight_ssim=0.5):
#     model = load_resnet_model()
#     resnet_features = extract_resnet_features(model, image_paths)
#     ssim_matrix = compute_ssim_matrix(image_paths)
#     combined_similarity = combine_features_and_ssim(resnet_features, ssim_matrix, weight_ssim)
#     group_ids = cluster_with_hdbscan(combined_similarity)
#     return group_ids













