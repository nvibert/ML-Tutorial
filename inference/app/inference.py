import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import threading

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Global model and lock for thread safety
model = None
model_lock = threading.Lock()

def predict_class(image_path):
    # Acquire lock for the entire prediction to ensure model consistency
    with model_lock:
        if model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to grayscale
        image = Image.open(image_path).convert('L')  
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            output = model(input_tensor)

        # Get the predicted class
        _, predicted_class = output.max(1)
        return predicted_class.item()


def refresh_model():
    # Load new model instance
    new_model = Net()
    new_model.load_state_dict(torch.load('app/mnist_cnn.pt', map_location=torch.device('cpu')))
    new_model.eval()
    
    # Atomically replace the global model
    global model
    with model_lock:
        model = new_model

# Main: load the trained model
refresh_model()