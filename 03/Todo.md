# ECE324: Assignment 3

## 3: Data Pre-Processing and Visualization (29 Points)

### 3.1: Initial Data Tasks:
- [ ] Take a gander at `adult.csv`.
- [ ] Use `pd.read_csv()` to read the `adult.csv` data into `data`.

### 3.2: Sanity Checks:
- [ ] Print the `.shape` field of the dataframe.
- [ ] Print the `.columns` field of the dataframe (column names).
- [ ] Print the `.head()` of the dataframe.
- [ ] Use the `data['income'].value_counts()` to determine the number of high and low income earners.

Questions (Answer in Final PDF):
- [ ] How many high income earners are there? Low income?
- [ ] Is the dataset balanced? What are some problems with training on an unbalanced dataset?

### 3.3: Cleaning:
- [ ] Missing values are indicated by the "?" string. Figure out **how many missing values** there are.
    - [ ] Do this by iterating over the columns and use the `.isin("?").sum()` function.
- [ ] Remove any row that has ≥1 "?" value. Use `[data[data[column != value]]]`
- [ ] Print out the shape of the dataset.

Questions (Answer in Final PDF): 
- [ ] How many samples were removed during cleaning? How many are left?
- [ ] Is this a reasonable number of rows to throw out?

### 3.4: Balancing the Dataset
- [ ] Use `DataFrame.sample` function to balance the dataset. 
    - [ ] Use the `random_state` argument to ensure same results each time.

### 3.5: Visualization and Understanding
- [ ] Use the `.describe()` function in the DataFrame class to determine statistics on data.
- [ ] Use `verbose_print` method. Ensure the following information is there.
    - [ ] Count (number of samples with non-null values)
    - [ ] Mean
    - [ ] Standard deviation
    - [ ] Minimum
    - [ ] Lowest 25% of the field.
- [ ] Print the number of times each value of the **categorical** features occur in the dataset.
    - [ ] Use `pie_chart` from `util.py` to visualize the first 3 categorical features using pie charts.
    - [ ] Include these plots in your final report.
- [ ] Use the `binary_bar_chart` method to plot the binary bar graph for the first 3 categorical features.
    - [ ] Include charts in the report.

Questions (Answer in Final PDF):
- [ ] What is the minimum age of individuals? Minimum number of hours worked per week in the entire dataset?
- [ ] Are certain groups over or under-represented? Do you expect the results traing on the dataset to generalize well to different races?
- [ ] What other biases in the dataset can you find? (2 points).
---
- [ ] List the top three features useful for distinguishing between high and low salary earners.
- [ ] How likely would someone who has a high school-level of education be to earn above 50K? How about with Bachelors?

### 3.6: Pre-Processing
Continuous values should be normalized -> mean = 0, standard deviation = 1. Categorical features should be encoded as 1-hot vectors.

- [ ] Extract continuous features into a separate variable.
    - [ ] Subtract the average (`.mean()`) and divide by the standard deviation (`.std()`).
    - [ ] Return numpy representation using `.values`.
- [ ] Use `LabelEncoder` class from sklearn to turn categorical features into integers.
    - [ ] Use `OneHotEncoder` class from sklearn to convert integers into 1-hot vectors.
    - [ ] Call `fit_transform` in `LabelEncoder` to convert (back to?) integer representation. 
    - [ ] Be sure to include the "income" column!
- [ ] Extract the `income` column and store it as a separate variable (numpy array).
    - [ ] Do not convert to 1-hot.
    - [ ] Remove "income" field from feature DataFrame -> separate features from the label.
    - [ ] use OneHotEncoder class to convert each categorical feature from integer to one-hot.
    - [ ] Stitch the categorical and continuous features back together.

Questions (Answer in Final PDF):
- [ ] What are some disadvantages of using an integer representation for categorical data?
- [ ] What are some disadvantages of using un-normalized continuous data?
- [ ] *Bonus*: Create a separate dataset where continuous features are un-normalized and categorical features are repesented as integers. Compare the performance of the NN versus the one created above. Report any differences in the report. 

### 3.7: Training-Validation Split
- [ ] Use `train_test_split` function from sklearn to separate training and validation sets.
    - [ ] `test_size` = percentage of data used for testing portion (0 => all data in training). 
    - [ ] Set `test_size=0.2`. 
    - [ ] Ensure you use a specified random seed for reproducibility.

## 4: Model Training (18 Points)

Stochastic gradient descent: Selects random subset of data of fixed size (`mini-batch` or `batch` size) to perform gradient descent + parameter change. Epoch occurs once all mini-batches are used once. 

### 4.1: DataSet
- [ ] Use PyTorch `DataLoader` to manage batch sampling in training loop.
    - [ ] Define dataset class extending PyTorch `data.Dataset`.
    - [ ] Refer to [this tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
    - [ ] Code has been started in `dataset.py`. Complete the implementation for `AdultDataset` class.

### 4.2: DataLoader
- [ ] Fill in `load_data()` from `main.py`.
    - [ ] Instantiate two `AdultDataset` classes -- one for training, one for validation.
    - [ ] Create two instances of `DataLoader` class with training + validation datasets.
    - [ ] Specify batch size with `batch_size` argument.
    - [ ] Specify `shuffle=True` for train loader.

Questions (Answer in Final PDF):
- [ ] Why is it important to shuffle the data during training? What problem might occur during training if the dataset was collected in a particular order and was not shuffled?

### 4.3: Model 

Model will be a Multi-Layer Perceptron (only linear activation functions) to predict income. Starter code is in `model.py`. 

- [ ] Create a model with two layers: 
    - [ ] Linear hidden layer.
    - [ ] Linear output layer.
    - [ ] First layer should use ReLU activation function.
    - [ ] Choose a size for ReLU first layer that is appropriate.
- [ ] Define complete architecture in `forward` function.
- [ ] Apply `Sigmoid` activation function after the output linear layer.

Questions (Answer in Final PDF):
- [ ] Justify your choice of the size of the first layer. What should the size of the second (output) layer be?
- [ ] Why is the output a probability between 0 and 1? What do 0 and 1 mean if they are the output of the NN? 

### 4.4: Loss Function and Optimizer
- [ ] Define method called `load_model` to instantiate the MLP & optimizer. 
- [ ] Loss function: Use PyTorch's `MSELoss`.
- [ ] Optimizer: `optim.SGD`. 
- [ ] Input for `load_model`: Learning rate. Guess a good one to default to.

### 4.5: Training Loop
- [ ] Call `load_data` and `load_model` from main. 
- [ ] Write a training loop that iterates through total `epochs`.
    - [ ] Each `epoch` should iterate over the training loader and perform a gradient step. 
    - [ ] Use [the same tutorial as before](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) for data loader.
    - [ ] Use [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for the training process. 
    - [ ] Zero the gradients at the beginning of each step. 
- [ ] Inner loop: Print loss and number of correct predictions ever $N=10$ steps (every mini-batch).
- [ ] Decision function: If output >0.5, then we predict label 1. Otherwise, we predict 0. 

### 4.6: Validation
- [ ] Create method `evaluate` in `main`.
    - [ ] Inputs: Model and validation dataset loader.
    - [ ] Output: Accuracy on the training set. 
- [ ] Print out result of `evaluate` every $N$ training steps (every batch) in the training loop. 
- [ ] Train the model.
    - [ ] Working model = 80% accuracy. 
    - [ ] Play with hyperparameters until 80% is reached.
- [ ] Plot training accuracy and validation accuracy as function of number of `mini-batches`. 
    - [ ] Training accuracy = averaged accuracy of 10 most recent batches.
    - [ ] Include plots in report.
    - [ ] Report batch size, MLP hidden size, learning rate.
- [ ] Add smothed plots to graphs for better visualization.
    - [ ] Use `savgol_filter` from scipy.signal.
    - [ ] Small $N$ -> oscillation, consider increasing that too.

## 5: Hyperparameters (34 Points)

Hyperparameter List:
- learning rate
- batch size (number of samples in a mini-batch).
- activation function of first layer.
- number of layers.
- loss function.
- regularization.

We can either use a *random* approach to finding the hyperparameters or a *grid search*. 

### 5.1: Learning Rate
- [ ] Set hidden layer size to 64, batch size to 64.
- [ ] Use grid search to find best learning rate.
    - [ ] Ensure that there is a separate section at the top to set specific hyperparameters.
    - [ ] Vary the learning rate from 1e-3 to 1e+3 (vary by a factor of 10 each time).
    - [ ] Re-train model each time. 
    - [ ] Report highest validation accuracy for each learning rate in a table. 
    - [ ] Include the table in the report.
    - [ ] Plot training accuracy and validation accuracy as a function of the number of steps taken for $\alpha = 0.01, 1, 100$.
    - [ ] Include plot in the final report.

Questions (Answered in Final PDF): 
- [ ] Which learnin rate works the best?
- [ ] What happens if the learning rate is too high? Too low?

### 5.2: Number of Epochs
Good idea to have more rather than fewer. If your validation is still increasing by the end of the run, the training process had too few epochs. Same vice versa. 

- [ ] Adjust epochs to a 'good setting' for the rest of the assignment.

### 5.3: Batch Size
Using the best learning rate from the previous section...
- [ ] Try batch sizes of 1, 64, and 17932.
    - [ ] Make plots of train and validation accuracies versus the number of steps for each.
    - [ ] Include the plots in the report. 
    - [ ] You would need to change $N$ (frequency of recording the training and validation error). 
    - [ ] Try to reduce the number of epochs for the batch size = 1 case. 
- [ ] Measure time of the training loop and plot the training and validation accuracy versus time instead of steps.
    - [ ] Include plots in the report.

Questions (Answered in Final PDF):
- [ ] What batch size gives the highest validation accuracy?
- [ ] Which batch size is fastest in reaching a high validation accuracy in terms of the number of steps? Which batch size is fastest at reaching maximum validation accuracy in terms of time?
- [ ] What happens if batch size is too low? Too high?
- [ ] State advantages and disadvantages of small batch size? Large batch size? Generalize a statement about the value of batch size (relative to 1 and the size of the dataset).

### 5.4: Under-fitting
- [ ] Make MLP have no hidden layers, only one linear layer mapping directly to output (still include Sigmoid).
    - [ ] Plot train and validation accuracy versus steps.
- [ ] What validation accuracy does the small model achieve? How does this compare to best model trained so far?
    - [ ] Is the model underfitting?

### 5.5: Over-fitting
- [ ] Using the best learning rate and batch size found so far...
- [ ] Change MLP to have four layers with hidden layers having size 64. 
    - [ ] Plot training and validation accuracy versus number of steps.
    - [ ] Include plots in report.
- [ ] What validation accuracy does the large model achieve? How does it compare to the best model so far? Is the model over fitting?

### 5.6: Activation Function
- [ ] Take best model trained so far, replace ReLU with tanh THEN sigmoid.
    - [ ] Plot train and validation accuracy of all three models on the same graph.
    - [ ] Any qualitative differences? Quantitative? 
    - [ ] Include plots in final report.
- [ ] Measure time of each of the training runs with each activation function.
    - [ ] Include a table of the times in your report.
    - [ ] Is there a difference between activation functions in terms of how long they take?

### 5.7: Hyperparameter Search

- [ ] Write down the value of the random seed used and hyper parameters values that produced the best results. 