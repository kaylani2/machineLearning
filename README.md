# machineLearning

This repository is used to teach and provide examples for basic and intermediate concepts regarding Machine Learning and Deep Learning.

## Note on file structure:

![Overview (IT MAY CHANGE)](images/ml_repo_file_system.png?raw=true "Overview")

### simple_examples:
  Contains a few implementations of learning models that are classically used for didactic purposes, like neural networks on the MNIST dataset.

### plot_functions:
  A few code snippets for plotting useful stuff, like commonly used activation functions.

### model_mockups:
  Actual learning models applied to famous datasets used in computer networks (usually).

## Files:

### download_datasets.py
  Download the following datasets: CICIDS e NSL-KDD.

### check_version.py
  Self-explanatory.

## Warning:
  Some files used as examples were taken directly from the examples in the corresponding library's documentation and may contain code that is not appropriate for hyperparameter tuning, specifically some [examples](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py#L80. "examples") from the keras documentation.

  As pointed out is this [issue](https://github.com/keras-team/keras/issues/1753 "issue"). Some examples use a test set (named as such) for validation. Although the code is not being used for hyperparameter tuning, the mixed nomenclature between test and validation sets should have been avoided and are currently (as of May 7th, 2020). Be aware that in order to perform hyperparameter tuning there is the need to separate the test set for usage only after the model has been finely tuned.

  ```python
  ## This sould be avoided:
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  .
  .
  .
   model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)
  .
  .
  .

  scores = model.evaluate(x_test, y_test, verbose=1)
  ```
