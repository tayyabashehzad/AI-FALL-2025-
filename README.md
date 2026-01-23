# Enhanced Federated Learning with Multi-Dataset Integration (EFL-MDI)

## Project Overview

This project presents **Enhanced Federated Learning with Multi-Dataset Integration (EFL-MDI)**, a robust federated learning framework designed to simulate realistic distributed learning environments. The system integrates multiple image datasets and applies adaptive client filtering to defend against malicious or unreliable participants.

The framework combines **MNIST**, **Fashion-MNIST**, and **CIFAR-10** into a unified classification task and trains a global CNN model using secure and adaptive aggregation strategies. It focuses on handling **non-IID data**, **malicious client behavior**, and **communication efficiency** in federated learning.

---

##  Objectives

* Simulate federated learning with multiple distributed clients
* Handle non-IID data distribution across clients
* Integrate multiple image datasets into a single classification task
* Detect and filter malicious or anomalous client updates
* Improve robustness, accuracy, and training stability

---

##  Datasets Used

The following standard benchmark datasets are utilized:

### MNIST

* Handwritten digits (0–9)
* Grayscale images

### Fashion-MNIST

* Clothing and fashion items (10 classes)
* Grayscale images

### CIFAR-10

* Natural color images (10 classes)
* RGB images

---

##  Dataset Integration

To unify the datasets:

* All images are resized to **32×32**
* All images are converted to **RGB** format
* Pixel values are normalized to the range **[0, 1]**
* Class labels are remapped as follows:

  * MNIST → `0–9`
  * Fashion-MNIST → `10–19`
  * CIFAR-10 → `20–29`

The final unified dataset contains **30 total classes**.

---

##  Model Architecture

The global model is a **Convolutional Neural Network (CNN)** composed of:

* Multiple convolutional layers for feature extraction
* Batch normalization for improved training stability
* Max pooling and global average pooling layers
* Fully connected layers with dropout for regularization
* Softmax output layer for **30-class classification**

**Optimizer:** Adam
**Loss Function:** Sparse Categorical Cross-Entropy

---

##  Federated Learning Workflow

1. Initialize the global CNN model at the server
2. Split the unified dataset among clients using a **Dirichlet distribution** to simulate non-IID data
3. For each federated communication round:

   * Randomly select a subset of clients
   * Each client performs local model training
   * Clients send model updates to the server
   * The server detects malicious or unreliable clients
   * Only trusted client updates are aggregated
   * Update and evaluate the global model

---

##  Malicious Client Detection

To enhance robustness, the framework incorporates:

* RAM-safe update summarization (mean, standard deviation, norm)
* **K-Means clustering** of client updates
* Adaptive thresholds based on attack confidence
* Dynamic identification of trusted clients

This mechanism helps defend against:

* Model poisoning attacks
* Outlier or abnormal updates
* Malicious gradient manipulation

---

##  Evaluation Metrics

The following metrics are tracked during federated training:

* Global model accuracy
* Training loss
* Number of trusted clients per round
* Attack confidence score
* Estimated communication cost

---

##  Visualization

The project includes visualizations for:

* Sample images from MNIST, Fashion-MNIST, and CIFAR-10
* Samples from the merged unified dataset
* Global training accuracy across federated rounds
* Training loss trends over time

These visual checks help verify correct preprocessing, dataset integration, and training behavior.

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Matplotlib

---

##  Conclusion

The **EFL-MDI** framework demonstrates how federated learning can be significantly enhanced through:

* Multi-dataset integration
* Non-IID data simulation
* Adaptive malicious client detection

The proposed approach improves robustness, reliability, and overall performance, making it suitable for real-world, privacy-sensitive, and distributed learning applications.

---

##  Author

**Mahnoor Riaz**
**Hifsa Shahid**
AI Lab – Final Term Project
