#!/bin/bash

kubectl apply -f train-pod.yaml
kubectl cp ./poison-data.sh mnist-train:/app/
kubectl exec mnist-train -- /app/poison-data.sh /data/MNIST/raw