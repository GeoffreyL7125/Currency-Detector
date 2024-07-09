# Currency Detector
The **Currency Detector** is an AI model designed to assist individuals with visual impairments, especially those who are blind, in identifying various currencies. Recognizing the challenges faced by people with eyesight problems in everyday transactions, this project leverages the power of artificial intelligence to provide a practical solution.\
The model works by capturing the image of the currency through a camera. It then processes the image and outputs information about the currency via audio, making it accessible and user-friendly for those with visual impairments.\
Our dataset includes **1,287 images** of banknotes from **11 different Southeast Asian countries and Taiwan**, ensuring a diverse representation of currencies. By utilizing ***Convolutional Neural Networks (CNNs)*** and ***Transfer Learning*** techniques, we have developed a robust AI model capable of accurately detecting and identifying these currencies.

This project was created as a final project for the Introduction to AI class at National Chengchi University (NCCU) in Taiwan. Our goal is to offer a reliable and accessible tool that can make a meaningful impact in the lives of those with visual impairments.



## Getting Started
1. Download [**Visual Studio Code**](https://code.visualstudio.com/ "**Visual Studio Code**"), then install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python "Python") and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter "Jupyter") extensions within it.
> [!IMPORTANT]
> Due to the dependency on a computer's webcam, this model might not perform optimally on online platforms like Google Colab or Jupyter Notebook. For the best experience, it is recommended to use Visual Studio Code or another local development environment.\
> 
> **If you are using a local development environment other than Visual Studio Code, you can skip this part.**

2. Download the [latest version of Python](https://www.python.org/downloads/ "latest version of Python").

3. Install the requirements.
```
pip install numpy opencv-python tensorflow gtts pygame keras scikit-learn
```

4. Download our [Banknotes Dataset](https://github.com/GeoffreyL7125/Currency-Detector/tree/main/Banknotes%20Dataset "Banknotes Dataset") and [AI model script](https://github.com/GeoffreyL7125/Currency-Detector/blob/main/src/Currency%20Detector.ipynb "AI model script").

5. Run the script using your local development environment.
> [!TIP]
> To capture the image, press **q** or **Q** on the keyboard.



## Dataset Structure
The **Banknotes Dataset** is organized into **11 main folders**, each corresponding to a **Southeast Asian country** and **Taiwan**. Within these country-specific folders, there are subfolders dedicated to different denominations of banknotes. For instance, the Taiwan folder includes subfolders such as 100 New Taiwan Dollar, 200 New Taiwan Dollar, and 500 New Taiwan Dollar, among others.\
Each of these denomination subfolders contains a collection of images sourced from various platforms including *Pinterest*, *Getty Images*, *Shutterstock*, *Google*, and other specialized websites.
> [!IMPORTANT]
> The names of these folders are directly utilized within the code for data fetching and model training processes. It is recommended to keep these folder names unchanged to maintain seamless integration and functionality within the training pipeline.



## How the Model Works?
The **Currency Detector** uses a sophisticated AI model that combines a ***Convolutional Neural Network (CNN)*** built with [TensorFlow Keras](https://www.tensorflow.org/guide/keras "TensorFlow Keras") and ***Transfer Learning*** from the [ResNet50V2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2 "ResNet50V2") architecture to identify various banknotes from different Southeast Asian countries and Taiwan. The process involves capturing an image through a webcam and using this integrated model to classify the currency. Here's a step-by-step breakdown of how the model works:

1. **Dataset Preparation and Loading:** \
The dataset is organized into folders representing 11 different countries, each containing subfolders for various banknote denominations. These images are loaded, resized to 224x224 pixels, and normalized. The labels for each banknote are converted into numerical values for training.

2. **Model Architecture:** \
The model is built by combining CNN and Transfer Learning:
	- **CNN with TensorFlow Keras:** The architecture starts with the ResNet50V2, a powerful CNN pre-trained on the ImageNet dataset. This pre-trained model provides a robust feature extractor for image classification tasks.
	- **Transfer Learning from ResNet50V2:** The ResNet50V2 model is used without its top layer, enabling the transfer of learned features to our specific task. This helps in leveraging the general features learned by ResNet50V2 on a large dataset and applying them to recognize various banknotes.
	- The top layer of ResNet50V2 is replaced with a custom classifier, which includes:
		- **Global Average Pooling Layer:** This layer reduces the spatial dimensions of the feature maps, making the model more efficient.
		- **Dense Layers with ReLU Activation:** Several Dense layers are added to learn intricate features relevant to currency classification.
		- **Dropout Layer:** A Dropout layer is included to prevent overfitting by randomly disabling a fraction of neurons during training.
		- **Final Dense Layer with Softmax Activation:** The final layer outputs probabilities for each class (currency type), enabling classification.

3. **Model Training:** \
The model is compiled using the *Adam optimizer* and trained with a sparse categorical cross-entropy loss function. The dataset is split into training and validation sets, and the model is trained over 20 epochs with a batch size of 32. This training process helps the model to accurately recognize and differentiate between various banknotes.

4. **Currency Detection:** \
The detection process involves capturing an image of the currency through a webcam. The image is resized and pre-processed to match the input format required by the model. The model predicts the class of the currency, identifying both the country and the banknote's denomination.

5. **Speech Output:** \
After detecting the currency, the system generates a speech output to provide information about the identified banknote. This is achieved using [Google Text-to-Speech (gTTS)](https://pypi.org/project/gTTS/ "Google Text-to-Speech (gTTS)") to convert the text to audio. The audio is played back using the `pygame` library, making the information accessible to individuals with visual impairments.

6. **Exchange Rate Calculation:** \
The model also provides exchange rate information for the detected currency. It retrieves current exchange rates for USD and EUR and calculates the equivalent value of the detected currency in these terms. This exchange rate information is included in the audio output, providing a comprehensive understanding of the currency's value.



## Results
The **Currency Detector** model achieves an accuracy of **90-93%** in identifying banknotes from various Southeast Asian countries and Taiwan. Despite its high accuracy, the model exhibits several weaknesses that affect its real-world applicability, particularly for visually impaired users.

### Identified Weaknesses
1. **Accuracy with Partial or Occluded Images:** \
The model struggles when the currency is not fully visible or when other objects are in the frame. Misclassification occurs if the currency is folded or obstructed. This issue stems from a lack of diverse training data, limiting the model's ability to handle real-world conditions.

2. **Absence of Camera Guidance:** \
The absence of a guidance system makes it challenging for visually impaired users to align the currency correctly in front of the camera, which is crucial for accurate detection.

3. **Requirement for Button Press:** \
Users need to press a specific button to capture an image, which can be challenging for visually impaired individuals who may struggle to locate and accurately press the button.

### Recommendations for Improvement
To enhance the model's effectiveness, several key improvements are suggested. Expanding the dataset to include various conditions (angles, lighting, occlusions) will help the model generalize better. Developing a camera guidance system with audio or haptic feedback will assist users in properly aligning the currency, thus improving detection accuracy. Removing the need for a specific button press and introducing voice commands or automatic detection can make the system more user-friendly for visually impaired users. Additionally, using advanced preprocessing techniques like object detection and segmentation will help isolate the currency from the background, enhancing model accuracy. Implementing real-time feedback to alert users when the currency is not fully visible will further improve reliability and user experience.

### Future Research Directions
Future research should focus on integrating multimodal data, such as combining visual and tactile inputs, to better handle occlusions and partial visibility. Developing adaptive learning models that continuously learn from new data and user interactions will improve the model's accuracy and adaptability. Conducting user studies to understand the specific challenges faced by visually impaired individuals will guide the development of more effective and user-friendly solutions. Addressing these areas can significantly advance the utility and effectiveness of the Currency Detector for those who rely on it.



## Contributors
Project Manager | Designer: Jason Min Te 黄敏德 - 112305019 \
Writer | Designer: Owen Polim 傅宥景 - 112703010 \
Project Manager | Coder: Geoffrey 劉國明 - 112703009 \
Writer | Designer: Ziven 張紹恩 - 112301015



## Acknowledgments
We extend our heartfelt gratitude to **Professor Torrent Pien (卞中佩)** and **Teaching Assistant Ian** for their invaluable guidance and support throughout the semester in the Introduction to AI class. Their expert teachings, continuous support, and insightful recommendations were crucial in the development of the Currency Detector AI model. We appreciate their dedication to fostering our learning and growth in the field of artificial intelligence.



## References
[https://www.kaggle.com/code/iabhishekmaurya/transfer-learning-resnet50v2](https://www.kaggle.com/code/iabhishekmaurya/transfer-learning-resnet50v2)\
[https://www.kaggle.com/code/garginirmal/cnn-keras-image-classification](https://www.kaggle.com/code/garginirmal/cnn-keras-image-classification)\
[https://medium.com/@golnaz.hosseini/step-by-step-tutorial-image-classification-with-keras-7dc423f79a6b](https://medium.com/@golnaz.hosseini/step-by-step-tutorial-image-classification-with-keras-7dc423f79a6b)
