"""
Author: Haripreeth Avarur
Date: 12/09/2024
Version: 1
Contact: hari.avarur@gmail.com
Description: YOLO-based inference script for detecting objects in images.
"""



###-----IMPORTS-----###
from ultralytics import YOLO
import os
import cv2



###-----SETUP-----###
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'ModelTraining/runs/detect/train/weights/best.pt')
model = YOLO(model_path)



###-----INFERENCE-----###
image_path = os.path.join(base_dir, 'Data/Vitaly.Okhonya_2020_11_26_10_33_39_1606376019160.jpg')
results = model(image_path)



###-----SAVE RESULTS-----###
result_save_path = os.path.join(base_dir, 'Results/result.jpg')
results[0].save(result_save_path)

res_plotted = results[0].plot()
result_cv2_path = os.path.join(base_dir, 'Results/result_cv2.jpg')
cv2.imwrite(result_cv2_path, res_plotted)





# ''' MAIN VERSION: 21:47, 12.09.24'''
# from ultralytics import YOLO
# import os
# import cv2

# # Dynamically get the base directory of the project (AI Pipeline)
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Path to the trained YOLO model (relative to base directory)
# model_path = os.path.join(base_dir, 'ModelTraining/runs/detect/train/weights/best.pt')

# # Load your custom-trained YOLO model
# model = YOLO(model_path)

# # Path to your test image (relative to base directory)
# image_path = os.path.join(base_dir, 'Data/Vitaly.Okhonya_2020_11_26_10_33_39_1606376019160.jpg')

# # Perform inference on your test image
# results = model(image_path)

# # Print results to verify detections
# print(results)

# # Save the results with bounding boxes (relative to base directory)
# result_save_path = os.path.join(base_dir, 'Results/result.jpg')
# results[0].save(result_save_path)

# # Alternatively, save using OpenCV
# res_plotted = results[0].plot()
# result_cv2_path = os.path.join(base_dir, 'Results/result_cv2.jpg')
# cv2.imwrite(result_cv2_path, res_plotted)
