# Training & Inference in K8S

This project demonstrates how to train and deploy a neural network for handwritten digit recognition using the MNIST dataset on Kubernetes. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), commonly used for benchmarking machine learning models.

## Getting Started

First, clone this repository to your local machine:

```bash
git clone https://github.com/nvibert/ML-Tutorial.git
cd ML-Tutorial
```

## Prerequisites
- Docker
- kubectl
- a K8S cluster
    - You can simply create one using [kind](https://kind.sigs.k8s.io/).
    - Use [Cilium](https://cilium.io/) as the CNI and Cilium L2 Announcement for LoadBalancer IPs.

Example Cilium L2 Announcement Policy and LB IP Pool:

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

4. **Set the LoadBalancer IP as an environment variable**
   - Extract the external IP and store it for easy use in subsequent commands.
   ```bash
   export INFERENCE_IP=$(kubectl get svc mnist-inference-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
   echo "Inference service available at: $INFERENCE_IP"
   ```

5. **Send images to the server for prediction**
   - Use `curl` to POST image files to the API. The server will return the predicted digit as JSON.
   
   **Test with different digits:**
   
   Test digit 0:
   ```bash
   curl -X POST -F "file=@data/testing/0/10.jpg" http://$INFERENCE_IP:5000/predict
   ```
   
   Test digit 1:
   ```bash
   curl -X POST -F "file=@data/testing/1/1004.jpg" http://$INFERENCE_IP:5000/predict
   ```
   
   Test digit 2:
   ```bash
   curl -X POST -F "file=@data/testing/2/1.jpg" http://$INFERENCE_IP:5000/predict
   ```
   
   Test digit 3:
   ```bash
   curl -X POST -F "file=@data/testing/3/1020.jpg" http://$INFERENCE_IP:5000/predict
   ```
   
   Test digit 7:
   ```bash
   curl -X POST -F "file=@data/testing/7/0.jpg" http://$INFERENCE_IP:5000/predict
   ```
   
   Test digit 9:
   ```bash
   curl -X POST -F "file=@data/testing/9/1000.jpg" http://$INFERENCE_IP:5000/predict
   ```
   
   Example responses:
   ```json
   {"prediction": 0}
   {"prediction": 1}  
   {"prediction": 2}
   {"prediction": 3}
   {"prediction": 7}
   {"prediction": 9}
   ```

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
   kind create cluster --config samples/kind-config.yaml
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
   kubectl apply -f samples/cilium-lb-pool.yaml
   kubectl apply -f samples/cilium-l2-policy.yaml
   ```

4. **Optional: Enable Cilium Hubble (Network Observability)**
   Hubble provides deep network visibility for your Kubernetes cluster.
   
   Enable Hubble with the UI:
   ```bash
   cilium hubble enable --ui
   ```
   
   Open the Hubble UI (this will port-forward and open in your browser):
   ```bash
   cilium hubble ui
   ```


## TO DO

- Simulate an attack where a user would modify the folders - swapping the `6` folder with the `9` folder.
- Understand who executed the commands with Tetragon.
- Prevent data poisoning by adding a network policy?
- Let's suppose that access to the training machine is compromised. 
- Add a diagram of inter-communications between client and model.
- Add Hubble UI screenshot.
- Add Gateway API to introduce a new model. Refer back to the failed introduction of Chat GPT5 and the new to bring back Chat GPT 4-o model.
- Consider swapping MNIST to [Fashionm-Mnist](https://github.com/zalandoresearch/fashion-mnist)