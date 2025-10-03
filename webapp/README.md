# 🧠 MNIST Digit Predictor - Fun Web App

A fun, interactive web application that lets users draw digits and get AI predictions using the MNIST neural network model. Features two exciting game modes with animations, sound effects, and a responsive Bootstrap interface.

## ✨ Features

### 🎨 **Mode 1: Free Draw**
- Draw any digit (0-9) on the canvas
- Get instant AI predictions
- Fun animations and emoji reactions
- Perfect for exploring what the AI can recognize

### 🏆 **Mode 2: Challenge Mode**
- AI challenges you to draw specific digits
- Score tracking with accuracy percentage
- Success/failure animations and sound effects
- Confetti celebrations for correct answers!

### 🎪 **Fun Elements**
- **Animations**: Bouncing brain, pulsing elements, shake effects
- **Sound Effects**: Success chimes and failure sounds
- **Visual Feedback**: Glowing canvas, color-coded results
- **Responsive Design**: Works on desktop and mobile devices
- **Connection Status**: Real-time API connection monitoring

## 🚀 Quick Start

### Prerequisites
- Kubernetes cluster with the MNIST inference service running
- The inference service should be available at `mnist-inference-service:5000`

### Deploy the Web App

1. **Build and load the Docker image:**
   ```bash
   cd webapp
   docker build -t mnist:webapp .
   kind load docker-image mnist:webapp
   ```

2. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f webapp.yaml
   ```

3. **Get the web app URL:**
   ```bash
   export WEBAPP_IP=$(kubectl get svc mnist-webapp-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
   echo "Web app available at: http://$WEBAPP_IP"
   ```

4. **Open in your browser and start drawing!**

## 🎮 How to Play

### Free Draw Mode (Default)
1. Select "Free Draw" mode
2. Draw any digit (0-9) in the canvas
3. Click "Predict!" to see what the AI thinks you drew
4. Clear the canvas and try again!

### Challenge Mode
1. Select "Challenge Mode"
2. The app will show you a digit to draw
3. Draw that specific digit in the canvas  
4. Click "Predict!" to see if you got it right
5. Watch your score improve over time!
6. Click "New Challenge" for a different digit

## 🛠 Technical Details

### Architecture
- **Frontend**: HTML5 Canvas, Bootstrap 5, jQuery
- **Backend**: MNIST PyTorch inference service
- **Deployment**: Nginx serving static files with API proxy
- **Communication**: REST API with multipart/form-data for image upload

### Canvas Details
- **Resolution**: 280x280 pixels (display) → 28x28 pixels (model input)
- **Format**: JPEG images sent to the inference API
- **Drawing**: HTML5 Canvas with mouse and touch support

### API Integration
- **Endpoint**: `/api/predict` (proxied to inference service)
- **Method**: POST with multipart form data
- **Response**: JSON with prediction number (0-9)

## 🎨 Customization

### Styling
- Modify `assets/style.css` for custom themes
- Bootstrap 5 classes for responsive design
- CSS animations and transitions for smooth UX

### Functionality  
- Update `assets/app.js` for new features
- Easy to extend with additional game modes
- Modular class-based JavaScript architecture

### Deployment
- Customize `webapp.yaml` for different Kubernetes configurations
- Update `Dockerfile` for different web server setups
- Nginx configuration includes API proxy setup

## 🐛 Troubleshooting

### Connection Issues
- Ensure the inference service is running: `kubectl get pods -l app=mnist-inference`
- Check service connectivity: `kubectl get svc mnist-inference-service`
- Verify the web app can reach the API proxy

### Canvas Not Working
- Check browser console for JavaScript errors
- Ensure modern browser with HTML5 Canvas support
- Try refreshing the page

### Predictions Failing
- Verify the inference service is responding: `curl http://INFERENCE_IP:5000/index`
- Check that images are being created properly from canvas
- Monitor browser network tab for failed API calls

## 🌟 Fun Facts

- The neural network model recognizes 28x28 pixel grayscale images
- The same model architecture used in the original MNIST research
- Sound effects are generated using Web Audio API
- Confetti animations celebrate your drawing successes!
- The app works offline once loaded (except for predictions)

## 🔮 Future Enhancements

- [ ] Add drawing tutorials and tips
- [ ] Implement user profiles and leaderboards  
- [ ] Add more visual effects and animations
- [ ] Support for drawing multiple digits
- [ ] Integration with other ML models
- [ ] Offline prediction capability
- [ ] Drawing replay and sharing features

Enjoy drawing and testing your artistic skills against AI! 🎨🤖