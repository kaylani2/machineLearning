# machineLearning


## BibTeX:

If you find this code useful in your research, please consider citing:

Se este código foi útil para sua pesquisa, considere citar:

```
@inproceedings{bochie2020_deep,
    author = {Bochie, K. and Gilbert, M. S. and Gantert, L. and Barbosa, M. S. M. and Medeiros, D. S. V. and Campista, M. E. M.},
    booktitle = {Minicursos do XXXVIII Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos (SBRC)},
    month = {12},
    chapter = {4},
    title = {Aprendizado Profundo em Redes Desafiadoras: Conceitos e Aplicações},
    year = {2020},
    pages = {140--189},
    doi = {10.5753/sbc.5033.7.4}
}
```

This repository is used to teach and provide examples for basic and intermediate concepts regarding Machine Learning and Deep Learning.

## Note on file structure:

![Overview (IT MAY CHANGE)](images/ml_repo_file_system.png?raw=true "Overview")

### dataset_analysis:
  Contains a few examples of common operations performed on almost every project, such as: data loading, statistical analysis, simple data pre-processing, visualization, scaling, feature selection, etc.

### simple_examples:
  Contains a few implementations of learning models that are classically used for didactic purposes, like neural networks on the MNIST dataset.

### plot_functions:
  A few code snippets for plotting useful stuff, like commonly used activation functions.

### model_mockups:
  Actual learning models applied to famous datasets used in computer networks (usually).

### specific_models:
  Scripts developed to analyse real, specific datasets.

## Files:

### download_datasets.py
  Download the following datasets (.csv files only): CICIDS, NSL-KDD and UNSW-NB15.

## Warning:
  When downloading UNSW-NB15, "-1 / unknown" could be printed on terminal. This is **not** an error, it is a result of wget not being able to estimate the remaining time for large files. Wait until the program stop running.

## Warning:
  Some files used as examples were taken directly from the examples in the corresponding library's documentation and may contain code that is not appropriate for hyperparameter tuning, specifically some [examples](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py#L80. "examples") from the keras documentation.

  As pointed out in this [issue](https://github.com/keras-team/keras/issues/1753 "issue"). Some examples use a test set (named as such) for validation. Although the code is not being used for hyperparameter tuning, the mixed nomenclature between test and validation sets should have been avoided and are currently (as of May 7th, 2020). Be aware that in order to perform hyperparameter tuning there is the need to separate the test set for usage only after the model has been finely tuned.

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
