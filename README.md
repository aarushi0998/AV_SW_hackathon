# AV_SW_hackathon
The task was to train models with the training data given and adding my own images as a separate class. The link to the folder containing my own images is: 

https://drive.google.com/drive/folders/1w7MSQyObYaoI7DR42mfNpFQQg34Pw44_?usp=sharing

The dataset was unzipped and my images were added as a separate class to the last folder of the given dataset.

I then used haarcascade provided facextraction .xml file and applied it to the entire training dataset (around 4000 images)

Only the images with one face were selected and a new folder was created to which all the images were added, in their own sub-folders with titles- Class0, Class1,...Class1013 (Classes provided in the original dataset were 1012)
The link to this folder is:

https://drive.google.com/drive/folders/1w7MSQyObYaoI7DR42mfNpFQQg34Pw44_?usp=sharing

The new folder titled pre_processed_images was then used to train the models. Data augmentation was used on these images.

I first tried using just ANN models, with simple combinations of dense layers for an inputted number of maximum layers. However, the training accuracies were not at all usable, they were way too low.
Then, I switched to doing the same for CNN models i.e. trying combinations with and inputted number of maximum layers

The CNN model giving me the most accuracy was finally used and trained again, from the beginning. This file was saved and is in the github repo

## Downloads

The files available for download from the github repo are:
1. The .ipynb code file used for execution of all the above
2. The finally selected model .h5 file (sent by WeTransfer)
3. Screenshots of the plots of accuracy and loss of the ANN and CNN models. ANN was tried for a maximum of 2 layers (i.e. 1, 2 layers) and CNN was tried for a maximum of 3 layers
4. Screenshot of the final training accuracy
5. model.yaml file for the structure of the model
6. model_weights.h5 file for the optimised weights (sent by WeTransfer)

## Usage

1. In order to just test the already trained model, you must download the .h5 file, upload it to the server (Google drive for google colab or local server for Jupyter notebooks, etc. Then use the following code to load the model 

import tensorflow as tf
import tensorflow as keras
new_model = tf.keras.models.load_model('final_model.h5')

###In order to see the model architecture, the following code can be used

new_model.summary()

###To view the model weights

new_model.get_weights()

###Then to evaluate the above model,

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print(loss, acc)

OR

###To load the .yaml and model_weights.h5 files separately,

yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights("model_weights.h5")

2. In order to check model accuracies for models other than the finally selected model (i.e. for model with a differing number of maximum layers), the .ipynb file must be downloaded and uploaded to use with Jupyter notebooks, google colab, etc.

Note: The path would differ based on if the platform is google colab or local. Since I have used google colab, the path was connected to my drive. That part can be omitted if a local server is used. Furthermore, if the face extracted images are only used to train the model, the pre_processed_images folder itself can be used directly, (again the path would differ).
