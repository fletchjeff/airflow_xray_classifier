# ML Ops with Airflow, Tensorflow and MLFlow

 This repo walks through the steps necessary to create and operate an image classifier using Apache Airflow. This includes the DAG and required code to run it, steps to build the kubernetes resources and some helpful hints and tips to apply this into your own environment. This repo provides the code for [this webinar](https://www.astronomer.io/events/webinars/using-airflow-with-tensorflow-mlflow).

![dag](images/xray_dag.png)

## ML Application Overview
The end goal of this project is to deploy a Machine Learning application into production and automate model training when new data is available using Airflow. The actual ML Application is relatively simple in that it uses a pre-trained model and a well documented framework. This will build a Neural Network based image classifier that will predict if the lungs in an xray image have pneumonia or not. The dataset comes from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and the process uses the [transfer learning example](https://www.tensorflow.org/tutorials/images/transfer_learning) from Tensorflow. There are some changes that are model to optimise for 2 classes and to add in a [Lime Image Explainer](https://github.com/marcotcr/lime) option. You can see the changes in the included [Notebook](Xray_Classifier_Notebook.ipynb).

The basic process is that is implemented is:
* Check an S3 bucket location to see if there are new files with an [S3 Async Sensor](https://registry.astronomer.io/providers/astronomer-providers/modules/s3keysizesensorasync)
* If there is new data, download the files and updated training code to a shared storage
* Create a GPU node that can be used to train the model with Tensorflow and tracked with MLFlow
* Update a [Ray serve](https://www.ray.io/ray-serve) endpoint with the new model
* Update a streamlit application to use the new model

![streamlit](images/streamlit.png)

This creates a managed ML model that is updated when additional data becomes available, all managed with Airflow.

## Project Setup
This guide assumes you will be running on an AWS EKS cluster. EKS isn't a hard requirement for this project, but this repo was made with EKS. AKS, GKE etc will also work, but you will need to adapt the kubernetes specific requirements accordingly. The other assumption is that you will be using Astronomer's [astro cli](https://docs.astronomer.io/astro/cli/overview) to run Airflow locally.

### Step 1 - Set Up an EKS Cluster with Required Services
The EKS cluster consists for two node groups, one that uses smaller CPU only nodes (t3.large) for running the required services (MLFlow, Ray and Streamlit) that auto-scales to a minimum of 1 node and another that uses GPU nodes (g4dn.xlarge) that scales to zero when not in use. Most of the pods that are created will need access to a shared PersistentVolume to store image data and trained model artifacts. 

The required services need to run continuously and be accessible from your local machine to the cluster, on all the ports that are needed.

A more detailed setup guide for the EKS cluster used for this repo is available [here](kubernetes_docs/EKS Cluster.md). 

![notebook](images/eks.png)

Before moving to step 2 you need to have:
* The cluster context from your `kubectl` config file
* The exposed IPs/hostnames for the MLFlow, Ray and Streamlit pods
* The PersistentVolumeClaim name for the shared storage PersistentVolume

### Step 2 - Run Local Airflow Deployment
For this step you will need to have the Astronomer [astro cli](https://docs.astronomer.io/astro/cli/overview) installed along with some form of Docker engine (e.g. Docker Desktop). Clone this project into local folder. From there you need to add your kubernetes authentication details.

#### 2.1 Copy awscli and kubectl Config Details
Copy your aws cli credential files (config + credentials) into the `include/.aws/` directory. These files are usually in `~/.aws/`. You may have several sets of credentials, so you can delete the ones you don't need before copying over.
```
$ tree ~/.aws/
/Users/jeff/.aws/
├── config
└── credentials
0 directories, 2 files
```

You also need to copy the `KUBECONFIG` file for the cluster ino the `include/` directory. The is usually located at `~/.kube/config`. Again you may have multiple cluster contexts in that file, so you can delete the ones you don't need before copying over.

These files will be copied into the Docker image used for the local deployment and are *not* pushed to the git repo, so the credentials will remain relatively safe.

#### 2.2 Update the `Dockerfile`
Next open the `Dockerfile` file and update the environment variables with the information you created in step 1.

```Dockerfile
ENV CLUSTER_CONTEXT=[your cluster context] \
    STORAGE_PATH=/[root path for the shared storage e.g. /efs] \
    RAY_SERVER=[hostname / ip for the Ray server] \
    MLFLOW_SERVER=[hostname / ip for the MLFlow server] \
    PVC_NAME=efs-claim
```
![architecture](images/architecture.png)

#### 2.3 Create AWS Connection


#### 2.4 Configure Email

## Tooling Used
* [Astro CLI](https://docs.astronomer.io/astro/cli/overview)
    * Used to create 
* Apache Airflow
* EKS
* Tensorflow
* MLFlow
* Ray
* Streamlit


