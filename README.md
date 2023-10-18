# TD_Systemes_Intelligents_Avances

## Loading tensorboard to view graphs

To view the graphs in the MLP using the following command

```bash
tensorboard --logdir /path/to/logs --load_fast true
```

And replace "/path/to/logs" with the output of the writer so here for the MLP for example the command is:

```bash
tensorboard --logdir Results/MLP/ --load_fast true
```

Or you can just install an extension on VSCode

## Code explanation

### MLP

1- Imports and Torch Version:

Imports necessary libraries including PyTorch modules and the required datasets.
Downloads and prints the PyTorch version.

2- Downloading MNIST Dataset:

Downloads the MNIST dataset using PyTorch's datasets.MNIST.
Defines a training set and a test set with appropriate transformations.

3- Data Loaders and Training/Validation Split:

Creates data loaders for training, validation, and test sets using torch.utils.data.DataLoader.
Splits the training set into training and validation indices for later use.

4- MLP Model:

Defines an MLP (Multi-Layer Perceptron) model using PyTorch's nn.Module class.
The model has three linear layers with ReLU activations and dropout layers for regularization.

5- Training the Model:

Initializes an instance of the MLP model, specifies the RMSprop optimizer, and sets the learning rate.
Initializes a Tensorboard writer for logging.
Trains the model for 20 epochs, monitoring training and validation loss, accuracy, and logging the results.

6- Confusion Matrix:

Evaluates the resulting model on the test set and constructs a confusion matrix.
Displays the confusion matrix using ConfusionMatrixDisplay from scikit-learn.
Calculates and prints test loss and accuracy for each class.