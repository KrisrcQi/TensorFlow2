# TensorFlow2

Imperial College London TensorFlow2 practices

## Models
1. Feedforward Network
2. Conventional Neural Network

## Data
### [Fashion-MNIST dataset]()
Fashion-MNIST dataset is one of the dataset in tf package, it could be accessed by Keras API:
- fashion_mnist_data = tf.keras.datasets.fashion_mnist
- (train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

### [MNIST dataset of images of handwritten digits](https://yann.lecun.com/exdb/mnist/)
The MNIST dataset consists of a training set of 60,000 handwritten digits with corresponding labels, and a test set of 10,000 images. The images have been normalised and centred. The dataset is frequently used in machine learning research, and has become a standard benchmark for image classification models.

### [Diabetes dataset](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)
Loading the diabetes dataset from sklearn.datasets import load_diabetes
- diabetes_dataset = load_diabetes()
- print(diabetes_dataset['DESCR'])

### [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
The Iris dataset consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. For a reference, see the following papers:
- R. A. Fisher. "The use of multiple measurements in taxonomic problems". Annals of Eugenics. 7 (2): 179–188, 1936.

### [EuroSat dataset](https://github.com/phelber/EuroSAT)
The EuroSAT dataset consists of 27000 labelled Sentinel-2 satellite images of different land uses: residential, industrial, highway, river, forest, pasture, herbaceous vegetation, annual crop, permanent crop and sea/lake. For a reference, see the following papers:
- Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
- Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.

### [SVHN dataset](http://ufldl.stanford.edu/housenumbers/)
The SVHN dataset is an image dataset of over 600,000 digit images in all, and is a harder dataset than MNIST as the numbers appear in the context of natural scene images. SVHN is obtained from house numbers in Google Street View images.
- Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu and A. Y. Ng. "Reading Digits in Natural Images with Unsupervised Feature Learning". NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.

## Keras pre-trained models resources
1. [Keras application page](https://keras.io/api/applications/)
2. [Tensorflow Hub](https://tfhub.dev/)



