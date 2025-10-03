#!/bin/bash

kubectl apply -f train-pod.yaml
kubectl wait --for=condition=Ready pod/mnist-train --timeout=300s
kubectl cp ./poison-data.sh mnist-train:/app/
kubectl exec mnist-train -- /app/poison-data.sh /data/MNIST/raw