Execution Steps:
 1. Install requirements using 'pip install -r requirements.txt'.
 2. Run the 'train.ipynb' python notebook to download and train the 'YOLOv8n' model on the custom dataset.
 3. The custom dataset is already divided into training and validation sets.
 4. The results from the training and validation will be stored in './runs' directory.
 5. The best model parameters will be stored under './runs/detect/train/weights' path as 'best.pt'.
 6. Load up the 'test_video_capture.py' script and add the path from step 3 to the script and run. This will test the model performance on the unseen video dataset.