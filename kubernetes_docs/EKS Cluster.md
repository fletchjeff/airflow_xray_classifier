# EKS Autoscaling GPU Cluster



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
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 2 \
  --node-zones <your_prefered_zones>
```

### Step 3: Create Autoscaler Policy

Create a local file called cluster-autoscaler-policy.json with the following content:

```
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

```
apiVersion: v1
kind: Pod
metadata:
  name: nvidia-smi
spec:
  containers:
    - name: nvidia-smi
      image: nvidia/cuda:11.0-base
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

For ongoing  \
[https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html](https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html)
