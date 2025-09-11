# Training & Inference in K8S

This project demonstrates how to train and deploy a neural network for handwritten digit recognition using the MNIST dataset on Kubernetes. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), commonly used for benchmarking machine learning models.
## Prerequisites
- Docker
- kubectl
- a K8S cluster
    - You can simply create one using [kind](https://kind.sigs.k8s.io/).
    - Use [Cilium](https://cilium.io/) as the CNI and Cilium L2 Announcement for LoadBalancer IPs.
        - Example Cilium L2 Announcement Policy:
```yaml
apiVersion: "cilium.io/v2alpha1"
kind: CiliumL2AnnouncementPolicy
metadata:
    name: policy1
spec:
    loadBalancerIPs: true
    interfaces:
        - eth0
    nodeSelector:
        matchExpressions:
            - key: node-role.kubernetes.io/control-plane
                operator: DoesNotExist
```
        - Example Cilium LoadBalancer IP Pool:

```yaml
apiVersion: "cilium.io/v2alpha1"
kind: CiliumLoadBalancerIPPool
metadata:
    name: "pool"
spec:
    blocks:
        - cidr: "172.18.255.200/29"
```


## Training
The training step uses a PyTorch implementation of a Convolutional Neural Network (CNN) to learn from the MNIST dataset. The model is trained to recognize digits from images and save the trained weights for later inference.

1. **Build the training image and load it into the K8S cluster**
    - This step packages the training code and dependencies into a Docker image, then loads it into your Kubernetes cluster using kind.
    ```bash
    cd training
    docker build -t mnist:train .
    kind load docker-image mnist:train
    ```
2. **Deploy the training pod**
    - This creates a pod in your cluster that runs the training job. The pod will download the MNIST dataset, train the model, and optionally save the trained weights.
    ```bash
    kubectl apply -f train-pod.yaml
    ```
3. **Run the pod on a specific node (optional)**
    - You can label a node and use affinity rules to schedule the training pod on that node.
    ```bash
    kubectl label nodes kind-worker training=allowed
    kubectl apply -f train-pod-affinity.yaml
    ```
4. **Save the trained model to persistent storage (optional)**
    - Use a pod spec with a mounted volume to persist the trained model weights for later use in inference.
    ```bash
    kubectl apply -f train-pod-affinity-mount.yaml
    ```

## Inference
We are going to deploy a Python Flask server running the MNIST Inference in K8S.
- build the image & load it into the K8S cluster
    ```
    cd inference
    docker build -t mnist:inference
    kind load docker-image mnist:inference
    ```
- deploy in K8S
    ```
    kubectl apply -f inference.yaml
    ```
- get the external IP of the service
    ```
    kubectl get svc mnist-inference-service
    ```
- send a request to the server
    ```

        Example request:
        ```bash
        curl -X POST -F "file=@data/testing/7/7030.jpg" http://172.18.255.201:5000/predict
        ```
        Example response:
        ```json
        {
            "prediction": 7
        }
        ```
        ![](docs/img/request.png)

## Optional: Deploy Cluster and Install Cilium

You can optionally deploy your kind cluster and install Cilium with the correct settings before applying the LoadBalancer IP Pool and L2 Announcement Policy.

1. **Create your kind cluster**
    Save the following config to a file (e.g., `kind-config.yaml`):
    ```yaml
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
      - role: control-plane
      - role: worker
      - role: worker
    networking:
      disableDefaultCNI: true
    ```
    Create the cluster:
    ```bash
    kind create cluster --config kind-config.yaml
    ```

2. **Install Cilium with the required settings**
     You can install Cilium using Helm or the Cilium CLI.
     - **Cilium CLI:**
         ```bash
         cilium install --set kubeProxyReplacement=true --set l2announcements.enabled=true --set ipam.mode=kubernetes --set devices='{eth0}'
         ```
     - **Helm:**
         Save the sample values file from `samples/cilium-helm-values.yaml` and install with:
         ```bash
         helm repo add cilium https://helm.cilium.io/
         helm install cilium cilium/cilium --namespace kube-system --values samples/cilium-helm-values.yaml
         ```

3. **Apply the Cilium LoadBalancer IP Pool and L2 Announcement Policy**
    After Cilium is installed, apply the following manifests:
    ```bash
    kubectl apply -f <LB-pool-manifest>.yaml
    kubectl apply -f <L2-policy-manifest>.yaml
    ```
    ```
    Sample config:
    ```yaml
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
        - role: control-plane
        - role: worker
        - role: worker
    networking:
        disableDefaultCNI: true
    ```
# Optional: Sample kind cluster config
```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4

## Inference
The inference step deploys a Python Flask server in Kubernetes that loads the trained model and predicts the digit in uploaded images.

1. **Build the inference image and load it into the K8S cluster**
    - This packages the inference code and dependencies into a Docker image, then loads it into your cluster.
    ```bash
    cd inference
    docker build -t mnist:inference .
    kind load docker-image mnist:inference
    ```
2. **Deploy the inference server**
    - This creates a pod and service in your cluster that exposes the Flask API for digit prediction.
    ```bash
    kubectl apply -f inference.yaml
    ```
3. **Get the external IP of the service**
    - Use this command to find the IP address for accessing the inference API from outside the cluster.
    ```bash
    kubectl get svc mnist-inference-service
    ```
4. **Send an image to the server for prediction**
    - Use `curl` to POST an image file to the API. The server will return the predicted digit as JSON.
    ```bash
    curl -X POST -F "file=@data/testing/7/7030.jpg" http://172.18.255.201:5000/predict
    ```
    Example response:
    ```json
    {
      "prediction": 7
    }
    ```
    ![](docs/img/request.png)
