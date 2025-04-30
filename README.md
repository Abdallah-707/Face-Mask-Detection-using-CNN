# Face-Mask-Detection-using-CNN
Got it! Below is the **README.md** file written in proper **GitHub Markdown format**, following the syntax rules from the link you shared:  
🔗 [GitHub Basic Writing and Formatting Syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

You can copy-paste this directly into your GitHub repository's `README.md` file.

---

# Face Mask Detection using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify whether a person in an image is wearing a face mask or not. It uses a dataset of images with and without masks, trains a model using TensorFlow/Keras, and includes preprocessing, training, evaluation, and prediction capabilities.

---

## 📁 Dataset Used

The dataset used in this project is available on Kaggle:

- [Face Mask Dataset by Omkar Gurav](https://www.kaggle.com/omkargurav/face-mask-dataset)

It contains two directories:
- `with_mask`: Images of people wearing masks
- `without_mask`: Images of people **not** wearing masks

To use the dataset locally or in Colab:
1. Download the dataset as a `.zip` file.
2. Extract the contents to a folder named `data` inside your project directory.

Folder structure should look like this:
```
data/
├── with_mask/
└── without_mask/
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- PIL (Image processing)
- Matplotlib (Visualization)
- Scikit-learn (Train/Test Split)

---

## 🧠 Model Architecture

The CNN model includes the following layers:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'), Dropout(0.5),
    Dense(64, activation='relu'), Dropout(0.5),
    Dense(2, activation='sigmoid')  # Sigmoid for binary classification
])
```

Compiled with:
- Optimizer: `Adam`
- Loss function: `sparse_categorical_crossentropy`
- Metric: `accuracy`

---

## 🏃‍♂️ How to Run

### Local Setup (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Face-Mask-Detection.git
   cd Face-Mask-Detection
   ```

2. Install required packages:
   ```bash
   pip install numpy matplotlib tensorflow opencv-python pillow scikit-learn
   ```

3. Place the dataset in the `data/` folder as explained above.

4. Run the Python script or open the Jupyter Notebook.

---

## 📊 Training Summary

After training for **5 epochs**, the model achieves:

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~92%
- **Test Accuracy**: ~91%

Plots are generated to show:
- Training vs Validation Loss
- Training vs Validation Accuracy

These help visualize how well the model learns and generalizes over time.

---

## 📷 Prediction Example

You will be prompted to enter the path of an image file:

```bash
Path of the image to be predicted: test_images/person_with_mask.jpg
Output: The person in the image is wearing a mask
```

The trained model takes an image of a face, resizes it to 128x128 pixels, normalizes it, and predicts whether the person is wearing a mask (`1`) or not (`0`).

---

## 📋 Requirements File

Here’s a list of dependencies to create a `requirements.txt` file:

```
numpy
matplotlib
tensorflow
opencv-python
pillow
scikit-learn
```

Generate it using:
```bash
pip freeze > requirements.txt
```

---

## 🚀 Future Enhancements

- Add support for real-time webcam input
- Improve accuracy with data augmentation
- Convert model to `.tflite` for mobile deployment
- Create a GUI using Tkinter or Streamlit

---

## 📄 License

This project is under the **MIT License**.  
See [LICENSE.md](LICENSE) for details.
