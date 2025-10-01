Breast Cancer Prediction Project (Windows)

Follow these steps to set up, prepare the dataset, train the model, and run the Streamlit app.
----------------------------------------------------------------------------------------------------
1. Create a Virtual Environment

Open a Command Prompt and run:

python -m venv myenv
----------------------------------------------------------------------------------------------------
2. Activate the Virtual Environment

myenv\Scripts\activate

You should see (myenv) at the start of your command line indicating the environment is active.
----------------------------------------------------------------------------------------------------
3. Install Required Packages

Install the necessary Python libraries:

pip install streamlit numpy opencv-python tensorflow
----------------------------------------------------------------------------------------------------
4. Prepare the Dataset

Your model expects:

Grayscale ultrasound images

Size: 224x224 (images will be resized automatically in code)

Two classes:

"Normal" → healthy tissue

"Cancer" → any kind of abnormality

Optional: Convert a Multi-Class Dataset to Two Classes

If your dataset has multiple cancer subfolders (e.g., Benign, Malignant), you can reorganize them with the following Python script:

import os
import shutil

source_folder = "raw_dataset"
target_folder = "Dataset"

os.makedirs(os.path.join(target_folder, "Normal"), exist_ok=True)
os.makedirs(os.path.join(target_folder, "Cancer"), exist_ok=True)

for class_name in os.listdir(source_folder):
    class_path = os.path.join(source_folder, class_name)
    if class_name.lower() in ["normal", "healthy"]:
        dest = os.path.join(target_folder, "Normal")
    else:
        dest = os.path.join(target_folder, "Cancer")
    for img_file in os.listdir(class_path):
        shutil.copy(os.path.join(class_path, img_file), os.path.join(dest, img_file))


After running this script, your dataset should have the structure:

Dataset/
├── Normal/
└── Cancer/
----------------------------------------------------------------------------------------------------
5. Train and Save the Model

Use model.py to train your CNN on the prepared dataset. Running model.py will save the trained model as busi_cnn_model.h5, which is required for the Streamlit app.

python model.py
----------------------------------------------------------------------------------------------------
6. Run the Streamlit App

Once the model is saved, start the Streamlit application:

streamlit run app.py

----------------------------------------------------------------------------------------------------

7. Deactivate the Virtual Environment (Optional)

When done, deactivate the environment:

deactivate
