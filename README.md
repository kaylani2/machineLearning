# machineLearning

This repository is used to teach and provide examples for basic and intermediate concepts regarding Machine Learning and Deep Learning.

## Note on file structure:

![Overview (IT MAY CHANGE)](images/ml_repo_file_system.png?raw=true "Overview")


## Files:

### download_datasets.py
  Baixa e prepara os diretórios para os seguintes datasets: CICIDS e NSL-KDD.

### visualizar_iris.py
  Faz o plot em 3d e o pair plot do dataset iris.

### classificar_iris.py
  Usa k-nn pra classificar o dataset iris. Exibe os atributos do dataset também.

### check_version.py
  Autoexplicativo. Chega os imports usados nos outros exemplos.

### plot_relu_tanh.py
  Serve pra plotar as funções de ativação mais comuns. Tem a sigmoid também no código.

### plot_derivative.py
  Serve pra plotar uma função qualquer a derivada daquela função em alguns pontos. Bom para explicar o gradiente.

### forge.py
  Usa o dataset forge (sintético) para demonstrar classificação com k-nn.

### generalization.py
  Para explicar o conceito de generalização. Plota um polinômio de grau 2 com ruído. Faz regressão no início da função e compara com o final dela pra mostrar como aumentar o grau do polinômio usado na regressão causa overfitting.

### wave.py
  Exibe o dataset wave (sintético) para regressão.

### linear_regressor.py
  Usa os datasets wave (sintético) e boston para demonstrar regressão com regressor linear e regularização ridge.

### keras_mnist.py
