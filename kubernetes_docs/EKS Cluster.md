# EKS Autoscaling GPU Cluster

https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html
https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html


### Step 1: Create EKS cluster

You need the aws cli and eksctl installed on the machine you will be doing this from. Start with the following command to create the cluster:


```
eksctl create cluster \
--name <your_cluster_name> \
--region <your_prefered_region> \
--managed --without-nodegroup
```

### Step 2: Create Node Group

Next create the node group. Note in the command below you need to specify the node zones. Its best to use all available node zones and GPU instance availability is complicated. e.g.

`--node-zones eu-central-1a,eu-central-1b,eu-central-1c`

```
eksctl create nodegroup \
  --cluster <your_cluster_name> \
  --region <your_prefered_region> \
  --name <your_nodegroup_name> \
  --node-type g4dn.xlarge \
  --nodes 0 \
  --nodes-min 0 \
  --nodes-max 2 \
  --node-zones <your_prefered_zones>
```

### Step 3: Create Autoscaler Policy

Create a local file called cluster-autoscaler-policy.json with the following content:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": [
                "autoscaling:DescribeAutoScalingGroups",
                "autoscaling:DescribeAutoScalingInstances",
                "autoscaling:DescribeLaunchConfigurations",
                "autoscaling:DescribeTags",
                "autoscaling:SetDesiredCapacity",
                "autoscaling:TerminateInstanceInAutoScalingGroup",
                "ec2:DescribeLaunchTemplateVersions"
            ],
            "Resource": "*",
            "Effect": "Allow"
        }
    ]
}
```

Apply this file using the following command:


```
aws iam create-policy \
--policy-name AmazonEKSClusterAutoscalerPolicy \
--policy-document file://cluster-autoscaler-policy.json
```

### Step 4: Create Service Account

Next step is to create and attach the service account. Run the following commandL

```
eksctl utils associate-iam-oidc-provider \
--cluster <your_cluster_name> \
--region <your_prefered_region> \
--approve
```

Now you need to get the ARN for the AmazonEKSClusterAutoscalerPolicy created in the previous step. You can look it up in the AWS console UI, or you can use the following command which will give you the ARN.

```
aws iam list-policies \
--query 'Policies[*].[PolicyName, Arn]' \
--output text | grep AmazonEKSClusterAutoscalerPolicy
```

This will give you and output that looks like:

`arn:aws:iam::559345414282:policy/AmazonEKSClusterAutoscalerPolicy`

Use that policy arn in the following command.


```
eksctl create iamserviceaccount \
--cluster=<your_cluster_name>   \
--region <your_prefered_region> \
--namespace=kube-system   \
--name=cluster-autoscaler   \
--attach-policy-arn=<your_policy_arn>   \
--override-existing-serviceaccounts   \
--approve
```

### Step 5: Apply the Cluster Autoscaler Autodiscover YAML

Down load the .yaml file locally.

```
curl -o cluster-autoscaler-autodiscover.yaml https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml
```

Open and edit the cluster-autoscaler-autodiscover.yaml, and replace the &lt;YOUR CLUSTER NAME> tag with your cluster name. It should be line 163.

Apply this file to the cluster with `kubectl`


```
kubectl apply -f cluster-autoscaler-autodiscover.yaml
```

Next run the following command:

```
kubectl annotate serviceaccount cluster-autoscaler \ 
-n kube-system eks.amazonaws.com/role-arn=<your_policy_arn>
```

Then the following command:

```
kubectl patch deployment cluster-autoscaler \
-n kube-system \
-p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}}}}}'
```

### Step 6: Test the Cluster

Final step is to test the cluster to see if it autoscales. Check the nodes and pods that are running. There should only be one node.

Create a file called nvidia-smi.yaml with the following contents:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nvidia-smi
spec:
  containers:
    - name: nvidia-smi
      image: nvidia/cuda:11.7.0-base-ubuntu18.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

Apply this file to the cluster

```
kubectl create -f nvidia-smi.yml
```

This will cause a scale up of the GPU cluster. You can watch the logs for the autoscaler using:

```
kubectl -n kube-system logs -f deployment.apps/cluster-autoscaler
```


One you have the GPU node, this will execute the nvidia-smi pod.


```
$ k logs nvidia-smi
Mon Mar 28 14:18:01 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   32C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```


Don’t forget to delete this deployment or the cluster won’t scale down.


```
kubectl delete -f nvidia-smi.yml
```

### Step 7: Create EFS Persistent Volume

For ongoing storage of data, trained models and experiment tracking data the various pods will need access to a persistent storage. For this project I used an EFS drive mounted into the pods as a PersistentVolumeClaim that allowed multiple pods to access the storage simultaneously. The process for this detailed here: [https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html](https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html)

### Step 8: Create the various ML Services
Once the cluster and storage have been created, you need to create the long running ML services used by this project: Ray, MLFlow and Streamlit. The 3 YAML files used are in the `kubernetes_docs/yaml_files` folder of this repo. Before applying the file, check that your persistentVolumeClaim is correct. The MLFlow YAML file is pasted here for reference:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-app
spec:
  selector:
    matchLabels:
      app: mlflow-pods
  replicas: 1      
  template:
    metadata:
      labels:
        app: mlflow-pods
    spec:
      containers:
        - name: mlflow-container
          image: fletchjeffastro/tfmlflow:0.0.1
          ports:
            - containerPort: 5000
              protocol: TCP
          command: [ "/bin/bash", "-c", "--" ]
          args: [ "mlflow server --host 0.0.0.0 --backend-store-uri sqlite:////efs/mlflow_backend.db --default-artifact-root /efs/mlflow_data/"]
          volumeMounts:
          - name: persistent-storage
            mountPath: /efs
      volumes:
      - name: persistent-storage
        persistentVolumeClaim:
          claimName: efs-claim
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-svc
spec:
  allocateLoadBalancerNodePorts: true
  externalTrafficPolicy: Cluster
  internalTrafficPolicy: Cluster
  ipFamilies:
  - IPv4
  ipFamilyPolicy: SingleStack
  ports:
  - port: 5000
    protocol: TCP
    targetPort: 5000
  selector:
    app: mlflow-pods
  sessionAffinity: None
  type: LoadBalancer
```

To create this pod, from the project directory run `kubectl apply -f kubernetes_docs/yaml_files/mlflow_app.yaml`. This will create the application, the pod and expose the service on port 5000 through a LoadBalancer.

Once you have applied all 3 ML service YAML files, you can check the services and retrieve the LoadBalancer hostnames to access the services using: `kubectl get svc`

```
NAME            TYPE           CLUSTER-IP       EXTERNAL-IP                               PORT(S)
mlflow-svc      LoadBalancer   10.100.132.110   [loadbalancer name 1].amazonaws.com   5000:32246/TCP
ray-svc         LoadBalancer   10.100.208.5     [loadbalancer name 2].amazonaws.com   10001:32068/TCP,8265:30545/TCP,6379:31094/TCP,8000:31462/TCP
streamlit-svc   LoadBalancer   10.100.221.7     [loadbalancer name 3].amazonaws.com   8501:30556/TCP
```
