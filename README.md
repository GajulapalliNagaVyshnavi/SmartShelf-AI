# SmartShelf: An Automated Detection and Grouping of Retail Products
An AI-powered pipeline for automated retail product detection and grouping using YOLOv10 and deep learning. This project leverages a custom-trained YOLO model on the SKU-110K dataset for object detection, combined with ResNet50, SSIM, and HDBSCAN for accurate product grouping. Built with Flask, the solution offers a web-based interface for image uploads, real-time detection, and clustering of products on supermarket shelves, making it ideal for inventory management and retail analytics.

---

## Overview

This AI pipeline is designed for detecting and grouping retail products using a custom YOLOv10n model, trained on the SKU-110K dataset. The pipeline features a web-based image upload interface built with Flask, real-time object detection using YOLOv10n, and product grouping using deep learning (ResNet50), SSIM, and HDBSCAN.

The following guide will take you step-by-step through setting up the environment, training the model, running the Flask app, and testing the solution using both Python and cURL.

---

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.11**  
- **Conda**: For managing virtual environments  
- **pip**: For installing the required dependencies  

You can download and install Conda from:  
[Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

---

## Setup and Implementation

### 1. **Create a Virtual Environment**  
Use Conda to create an environment:

```bash
conda create --name <environment_name> python=<python_version>
```

Example used in this implementation:

```bash
conda create --name env_SmartShelf python=3.11
```

Activate the environment:

```bash
conda activate env_SmartShelf
```

### 2. **Install the Dependencies**

After creating and activating the environment, install the required dependencies via `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the following key libraries:

- **Flask**: For the web interface  
- **ultralytics**: For YOLO model implementation  
- **torch**: For running the YOLO and ResNet models  
- **opencv-python-headless**: For image processing  
- **hdbscan**: For clustering product groups  
- **scikit-image**: For SSIM comparison 

(The above libraries are the fundamental libraries, the requirement.txt files have the entire freeze of requirments used during the original implementation) 

### 3. **Train the YOLOv10n Model on SKU110K Dataset**

Open the `Yolo_on_SKU110K.ipynb` notebook in Jupyter and run it step by step. Ensure the SKU110K dataset and the configuration file (`SKU-110K.yaml`) are properly set up.

```bash
jupyter notebook ModelTraining/Yolo_on_SKU110K.ipynb
```

This notebook trains the YOLOv10n model for 10 epochs and saves the best model weights at:

```
ModelTraining/runs/detect/train/weights/best.pt
```

These weights are crucial for product detection in the Flask app.

### 4. **Running the Flask Application**

Navigate to the `Src` directory where the `app.py` script is located:

```bash
cd source
```

Now, run the Flask web app:

```bash
python app.py
```

The Flask app will start running on `http://127.0.0.1:5001/`. You can access this URL in your browser to upload images for product detection and grouping. Detected products will be grouped and shown on the webpage, along with a downloadable JSON file containing details of the detections.

#### 4a. **SUBCODE: Running Detection Script Manually (`detectProducts.py`)**

You can run the detection manually using the following script:

```bash
python detectProducts.py
```

This script loads a sample image from the `Data/` folder, performs detection, and saves the result in the `Results/` folder.

#### 4b. **SUBCODE: Product Grouping**

The `grouping.py` script is responsible for grouping detected products using a combination of deep learning (ResNet50), SSIM comparison, and clustering (HDBSCAN). This is integrated into the Flask application, so when you upload an image, it will automatically perform product grouping.

---

## Running the Flask App via CURL

To interact with the Flask web app via CURL, you can use the following syntax for uploading images:

### Syntax:

```bash
curl -X POST -F 'file=@<image_path>' http://127.0.0.1:5001/upload
```

### Example:

```bash
curl -X POST -F 'file=@Data/download1.jpg' http://127.0.0.1:5001/upload
```

The response will contain a link to the processed image and the JSON output showing the detected products and their assigned groups.

---

## Directory Structure

The project is organized as follows:

```
SmartShelf/
├── Data/                 # Sample images for testing
├── ModelTraining/        # YOLO training notebook and weights
├── Results/              # Detection results and JSON outputs
├── Source/                 
│   ├── app.py            # Flask app for uploading and processing images
│   ├── detectProducts.py # YOLO-based detection script
│   └── grouping.py       # Product grouping script
├── requirements.txt      # List of required dependencies
```

---

## Example Output

After uploading an image through the web app, the following actions occur:

1. **Product Detection**: YOLOv10n detects products and crops them from the image.
2. **Product Grouping**: Cropped images are grouped based on similarity using SSIM, ResNet features, and HDBSCAN.
3. **Visualization**: The grouped products are visualized with bounding boxes, and a JSON file containing detection details is generated.

The JSON results file will be saved in the `Results/JSON/` folder, and the visualization images will be saved in the `Results/Images/` folder.

---

## References

Here are some useful resources that were used as references during the implementation:

- **YOLOv10 Documentation**: [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- **SKU-110K Dataset**: [SKU-110K Dataset GitHub](https://github.com/eg4000/SKU110K_CVPR19)
- **Flask Documentation**: [Flask Official Documentation](https://flask.palletsprojects.com/en/2.0.x/)
- **ResNet50 Feature Extraction**: [PyTorch ResNet Tutorial](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet50)
- **HDBSCAN Clustering**: [HDBSCAN Documentation](https://hdbscan.readthedocs.io/en/latest/)
- **SSIM (Structural Similarity)**: [scikit-image SSIM](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html)
[SSIM Implementation](https://youtu.be/mggQIEZY4rE?si=OMqSlxmjEiiMoNaE)(https://www.youtube.com/watch?v=16s3Pi1InPU)
- **Grouping Experimentation**: [GitHub Reference](https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/evaluation/image-object-detection/image-object-detection.ipynb)
[Paper Reference](https://www.researchgate.net/publication/373919010_Research_of_image_object_detection_using_deep_learning)
---

## Author and Acknowledgements

**Author:** Gajulapalli Naga Vyshnavi  
**Contact:** nvyshnavi36@gmail.com

This project is built upon existing open-source libraries and frameworks like Ultralytics YOLO, Flask, PyTorch, and scikit-image. I would like to thank the creators and maintainers of these amazing tools for making this project possible.

For any inquiries, feel free to reach out via the contact information provided.
