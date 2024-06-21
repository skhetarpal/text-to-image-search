# text-to-image-search

INTRODUCTION
The goal of this project is to create a multi-modal machine learning tool that can receive a text description and then select from a library of images the image that most closely aligns with that description. The model architecture is a dual encoder that is trained to projects images and their descriptions onto the same space and at the same location. To serve as the base models of the dual encoder, we will use Inception V3 to encode the image data and BERT to encode the text data. On top of these base models will sit two small feedforward networks that will project the encoded images and text onto the common space where they will have the same dimensions. Let us call these two feedforward networks the "projection heads".

Because training for this dual encoder is so time consuming, we will speed up the process by breaking the training process into two phases.

TRAINING PHASES
Phase 1:
During phase 1, we will circumvent the base models completely, dramatically reducing the iteration time as we tune the model. We will use the base models just once to encode the entire dataset, and then we will train the projection heads using just the pre-encoded data.

Phase 2:
During phase 2, we will use the original data to fine tune the entire dual encoder, including the base models.

Note
This project draws inspiration and some design features from the keras project "Natural language image search with a Dual Encoder", located at https://keras.io/examples/vision/nl_image_search/.

## Project Structure
- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for exploration, model development, and visualization.
- `src/`: Core scripts for preprocessing, training, and evaluating.
- `models/`: Directory for saving trained models.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of dependencies.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt

## Usage
Clone the repository to your Projects folder:
>>cd Projects
>>git clone https://github.com/skhetarpal/text-to-image-search.git

To load and preprocess the data, run:
>>python src/data_preprocessing.py

To train the model, run:
>>python src/model_training.py

## Results
After Phase 1 of training, we acheived a 50% success rate.
After Phase 2, fine tuning drove the success rate up to 78%!
