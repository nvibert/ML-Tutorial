/**
 * MNIST Digit Predictor - Fun Edition
 * Interactive web app for drawing digits and getting AI predictions
 */

class MNISTPredictor {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.isDrawing = false;
        this.gameMode = 'free';
        this.currentChallenge = null;
        this.score = {
            correct: 0,
            incorrect: 0
        };
        this.apiEndpoint = '/api';
        this.hasDrawn = false;
        
        this.init();
    }

    init() {
        this.setupCanvas();
        this.setupEventListeners();
        this.setupSounds();
        this.checkConnection();
        this.updateGameMode();
    }

    setupCanvas() {
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set up canvas for drawing - optimized for MNIST 28x28 recognition
        this.ctx.lineWidth = 15;  // Slightly thinner for better detail when scaled to 28x28
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = '#FFFFFF';  // White foreground/digits (MNIST: 255)
        this.ctx.fillStyle = '#000000';   // Black background (MNIST: 0)
        
        // Improve anti-aliasing for smoother lines
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
        
        // Fill with white background
        this.clearCanvas();
    }

    setupEventListeners() {
        // Canvas drawing events
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            this.canvas.dispatchEvent(mouseEvent);
        });

        // UI Controls
        $('#clearCanvas').on('click', () => this.clearCanvas());
        $('#predictBtn').on('click', () => this.predict());
        $('#newChallengeBtn').on('click', () => this.generateNewChallenge());
        
        // Game mode switching
        $('input[name="gameMode"]').on('change', (e) => {
            this.gameMode = e.target.value;
            this.updateGameMode();
        });
    }

    setupSounds() {
        // Create success sound
        this.createSound('successSound', [523.25, 659.25, 783.99], [0.2, 0.2, 0.4]);
        // Create fail sound
        this.createSound('failSound', [196.00, 146.83], [0.3, 0.3]);
    }

    createSound(id, frequencies, durations) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audio = document.getElementById(id);
        
        // Store reference for later use
        this[id + 'Context'] = audioContext;
        this[id + 'Frequencies'] = frequencies;
        this[id + 'Durations'] = durations;
    }

    playSound(soundType) {
        try {
            const audioContext = this[soundType + 'Context'];
            const frequencies = this[soundType + 'Frequencies'];
            const durations = this[soundType + 'Durations'];

            let startTime = audioContext.currentTime;
            
            frequencies.forEach((freq, index) => {
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.setValueAtTime(freq, startTime);
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0, startTime);
                gainNode.gain.linearRampToValueAtTime(0.1, startTime + 0.01);
                gainNode.gain.exponentialRampToValueAtTime(0.001, startTime + durations[index]);
                
                oscillator.start(startTime);
                oscillator.stop(startTime + durations[index]);
                
                startTime += durations[index];
            });
        } catch (error) {
            console.log('Audio not supported, continuing without sound effects');
        }
    }

    startDrawing(e) {
        this.isDrawing = true;
        this.hasDrawn = true;
        this.updateCanvasState();
        this.draw(e);
        
        // Enable predict button
        $('#predictBtn').prop('disabled', false);
        
        // Add glow effect to canvas
        $(this.canvas).addClass('canvas-glow');
    }

    draw(e) {
        if (!this.isDrawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        this.ctx.lineTo(x, y);
        this.ctx.stroke();
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
    }

    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.ctx.beginPath();
            $(this.canvas).removeClass('canvas-glow');
        }
    }

    clearCanvas() {
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.beginPath();
        this.hasDrawn = false;
        this.updateCanvasState();
        
        // Reset UI state
        $('#predictBtn').prop('disabled', true);
        $('#predictionResult').addClass('d-none');
        $('#initialState').removeClass('d-none');
        $('#loadingState').addClass('d-none');
        $(this.canvas).removeClass('canvas-success canvas-error');
        
        // Add bounce animation to clear button
        $('#clearCanvas').addClass('bounce');
        setTimeout(() => $('#clearCanvas').removeClass('bounce'), 1000);
    }

    updateCanvasState() {
        const container = $('.canvas-container');
        if (this.hasDrawn) {
            container.removeClass('canvas-empty').addClass('canvas-drawing');
        } else {
            container.removeClass('canvas-drawing').addClass('canvas-empty');
        }
    }

    async predict() {
        if (!this.hasDrawn) {
            this.showAlert('Please draw something first!', 'warning');
            return;
        }

        // Show loading state
        $('#initialState').addClass('d-none');
        $('#predictionResult').addClass('d-none');
        $('#loadingState').removeClass('d-none');

        try {
            // Convert canvas to blob - use PNG for lossless quality (better for digit recognition)
            const blob = await new Promise(resolve => {
                this.canvas.toBlob(resolve, 'image/png');
            });

            // Create form data
            const formData = new FormData();
            formData.append('file', blob, 'drawing.png');

            // Make prediction request
            const response = await fetch(`${this.apiEndpoint}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }

            this.displayPrediction(result.prediction);
            
            // Optional: Show what the model "sees" for debugging
            this.showModelPreview(blob);
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showConnectionError();
            this.showAlert('Prediction failed. Please check the connection.', 'danger');
        }
    }

    displayPrediction(prediction) {
        // Hide loading state
        $('#loadingState').addClass('d-none');
        
        // Show prediction
        $('#predictedDigit').text(prediction).addClass('digit-reveal');
        $('#predictionResult').removeClass('d-none');
        
        // Handle game modes
        if (this.gameMode === 'challenge') {
            this.handleChallengeResult(prediction);
        } else {
            // Add some fun reactions for free mode
            this.addEmojiReaction(this.getRandomReaction());
        }

        // Remove digit reveal animation after it completes
        setTimeout(() => {
            $('#predictedDigit').removeClass('digit-reveal');
        }, 800);
    }

    handleChallengeResult(prediction) {
        const isCorrect = prediction === this.currentChallenge;
        
        if (isCorrect) {
            this.score.correct++;
            $('#successResult').removeClass('d-none');
            $('#failResult').addClass('d-none');
            $(this.canvas).addClass('canvas-success');
            this.playSound('successSound');
            this.createConfetti();
            this.addEmojiReaction('🎉');
        } else {
            this.score.incorrect++;
            $('#successResult').addClass('d-none');
            $('#failResult').removeClass('d-none');
            $(this.canvas).addClass('canvas-error shake');
            this.playSound('failSound');
            this.addEmojiReaction('😅');
            
            // Remove shake animation
            setTimeout(() => {
                $(this.canvas).removeClass('shake');
            }, 500);
        }
        
        $('#challengeResult').removeClass('d-none');
        this.updateScore();
    }

    updateGameMode() {
        if (this.gameMode === 'challenge') {
            $('body').removeClass('free-mode').addClass('challenge-mode');
            $('#challengeInstructions').removeClass('d-none');
            $('#challengeControls').removeClass('d-none');
            $('#scorePanel').removeClass('d-none');
            this.generateNewChallenge();
        } else {
            $('body').removeClass('challenge-mode').addClass('free-mode');
            $('#challengeInstructions').addClass('d-none');
            $('#challengeControls').addClass('d-none');
            $('#scorePanel').addClass('d-none');
        }
        
        this.clearCanvas();
    }

    generateNewChallenge() {
        this.currentChallenge = Math.floor(Math.random() * 10);
        $('#targetDigit').text(this.currentChallenge).addClass('pulse');
        
        // Remove pulse animation after it completes
        setTimeout(() => {
            $('#targetDigit').removeClass('pulse');
        }, 2000);
        
        this.clearCanvas();
    }

    updateScore() {
        const total = this.score.correct + this.score.incorrect;
        const accuracy = total > 0 ? Math.round((this.score.correct / total) * 100) : 0;
        
        $('#correctCount').text(this.score.correct).addClass('score-update');
        $('#incorrectCount').text(this.score.incorrect).addClass('score-update');
        $('#accuracy').text(accuracy + '%').addClass('score-update');
        
        // Remove animation after it completes
        setTimeout(() => {
            $('.score-update').removeClass('score-update');
        }, 500);
    }

    async checkConnection() {
        try {
            const response = await fetch(`${this.apiEndpoint}/index`);
            if (response.ok) {
                this.showConnectionSuccess();
            } else {
                throw new Error('Service unavailable');
            }
        } catch (error) {
            this.showConnectionError();
        }
    }

    showConnectionSuccess() {
        $('#connectionStatus')
            .removeClass('alert-info connection-error')
            .addClass('connection-success')
            .html('<i class="fas fa-check-circle"></i> Connected to inference service successfully!')
            .fadeOut(3000);
    }

    showConnectionError() {
        $('#connectionStatus')
            .removeClass('alert-info connection-success')
            .addClass('connection-error')
            .html('<i class="fas fa-exclamation-triangle"></i> Cannot connect to inference service. Please check if the service is running.');
    }

    showAlert(message, type = 'info') {
        const alert = $(`
            <div class="alert alert-${type} alert-dismissible fade show position-fixed" 
                 style="top: 20px; right: 20px; z-index: 9999; max-width: 300px;">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `);
        
        $('body').append(alert);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alert.alert('close');
        }, 5000);
    }

    getRandomReaction() {
        const reactions = ['🎯', '🤖', '✨', '🎨', '🧠', '💫', '🌟', '🎪', '🎭', '🎪'];
        return reactions[Math.floor(Math.random() * reactions.length)];
    }

    addEmojiReaction(emoji) {
        const reaction = $(`<div class="emoji-reaction">${emoji}</div>`);
        
        // Position randomly around the prediction area
        const container = $('#predictionResult');
        const containerOffset = container.offset();
        
        reaction.css({
            left: containerOffset.left + Math.random() * container.width(),
            top: containerOffset.top + Math.random() * container.height()
        });
        
        $('body').append(reaction);
        
        // Remove after animation
        setTimeout(() => {
            reaction.remove();
        }, 1000);
    }

    createConfetti() {
        const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
        
        for (let i = 0; i < 20; i++) {
            const confetti = $('<div class="confetti"></div>');
            confetti.css({
                left: Math.random() * window.innerWidth + 'px',
                backgroundColor: colors[Math.floor(Math.random() * colors.length)],
                animationDelay: Math.random() * 2 + 's',
                animationDuration: (Math.random() * 2 + 2) + 's'
            });
            
            $('body').append(confetti);
            
            // Remove after animation
            setTimeout(() => {
                confetti.remove();
            }, 5000);
        }
    }

    showModelPreview(blob) {
        // Create a preview of what the model sees (28x28 version)
        const img = new Image();
        img.onload = () => {
            // Create a temporary canvas for 28x28 preview
            const previewCanvas = document.createElement('canvas');
            previewCanvas.width = 28;
            previewCanvas.height = 28;
            const previewCtx = previewCanvas.getContext('2d');
            
            // Draw the image scaled down to 28x28
            previewCtx.drawImage(img, 0, 0, 28, 28);
            
            // Create a larger preview for display (280x280)
            const displayCanvas = document.createElement('canvas');
            displayCanvas.width = 140;
            displayCanvas.height = 140;
            const displayCtx = displayCanvas.getContext('2d');
            displayCtx.imageSmoothingEnabled = false; // Pixelated look
            displayCtx.drawImage(previewCanvas, 0, 0, 140, 140);
            
            // Show in console for debugging
            console.log('🔍 Model Preview: What the AI sees (28x28 scaled up)');
            console.log('Canvas Data URL:', displayCanvas.toDataURL());
            
            // Optional: Add to page for visual debugging (uncomment if needed)
            /*
            const preview = $('<div class="model-preview position-fixed" style="top: 10px; right: 10px; z-index: 9999; background: white; padding: 10px; border: 2px solid #007bff; border-radius: 8px;"><h6>Model Preview (28x28)</h6></div>');
            preview.append(displayCanvas);
            $('body').append(preview);
            setTimeout(() => preview.remove(), 5000);
            */
        };
        
        img.src = URL.createObjectURL(blob);
    }
}

// Initialize the app when the document is ready
$(document).ready(function() {
    console.log('🧠 MNIST Digit Predictor loading...');
    
    // Add loading animation to the brain icon
    $('.fa-brain').addClass('bounce');
    
    // Initialize the predictor
    window.mnistPredictor = new MNISTPredictor();
    
    console.log('✨ App ready! Start drawing digits!');
    
    // Add welcome message
    setTimeout(() => {
        if (window.mnistPredictor) {
            window.mnistPredictor.showAlert('Welcome! Draw a digit and let the AI guess what it is! 🎨', 'success');
        }
    }, 1000);
});