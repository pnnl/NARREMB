# Using Social Proximity to Generate Tweet Representations with Shared-viewpoints
***
Contents of code development for this project

### 01_data_preparation_notebooks/
* data_preparation.ipynb -- This notebook generates the data splits for training and testing based on a sampling strategy for close and distant pairs

### 02_model_training_scripts/
* contrast.py -- This script is used to define parameters for contrastive model training
* edcontrast.py -- This script is used to defined parameters for topic-aware contrastive model training

### 03_model_evaluation_notebooks/
* generate_bertopics_notebook.ipynb -- This notebook is used for the analysis on BERTopics
* Filemodel_evaluation_notebook.ipynb -- This notebook is used for model evaluation, including computing TopicNN and ViewpointNN metrics

### core/
* datasets.py -- Script to create PyTorch Dataset objects
* losses.py -- Contains contrastive loss functions
* models.py -- Contains the development of PyTorch model classes
* trainers.py -- Script for training models
* utils.py -- Contains various helper functions for building and training models, and logging model performance

### libs/
* data_process.py -- contains helper functions for data preprocessing
* file_utils.py -- helper functions for read/write files
* model_evaluation_helpers.py -- helper functions for model evaluation
* network_sampling.py -- helper functions for generating data examples
* visualization_helpers.py -- helper functions for visualizations

### dummy_data/
* embeds/ -- contains synthetic/dummy data for tweet embeddings
* network/ -- contains synthetic/dummy data for networks
* text/ -- contains synthetic/dummy data for texts

## Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
