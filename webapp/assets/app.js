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
        
        // Fill with black background
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
            // Convert canvas to blob
            const blob = await new Promise(resolve => {
                this.canvas.toBlob(resolve, 'image/jpeg', 0.8);
            });

            // Create form data
            const formData = new FormData();
            formData.append('file', blob, 'drawing.jpg');

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
            
            // Random success messages
            const successMessages = [
                { title: "Brilliant! 🧠✨", text: "You got it right! Smart cookie! 🍪" },
                { title: "Fantastic! 🌟", text: "Your brain is on fire! 🔥" },
                { title: "Awesome! 🎯", text: "Bulls-eye! Perfect answer! 🎪" },
                { title: "Genius! 🤓", text: "You're a digit-drawing master! 🎨" },
                { title: "Excellent! 🏆", text: "That's exactly right! Champion! 👑" },
                { title: "Amazing! ⭐", text: "Your drawing skills are incredible! 🎭" }
            ];
            
            const success = successMessages[Math.floor(Math.random() * successMessages.length)];
            $('#successResult h4').text(success.title);
            $('#successResult p').text(success.text);
            $('#successResult').removeClass('d-none');
            $('#failResult').addClass('d-none');
            $(this.canvas).addClass('canvas-success');
            this.playSound('successSound');
            this.createConfetti();
            this.addEmojiReaction(['🎉', '🎊', '🌟', '✨', '🏆', '👏'][Math.floor(Math.random() * 6)]);
        } else {
            this.score.incorrect++;
            
            // Random failure messages
            const failMessages = [
                { title: "Hmm, not quite! 🤔", text: "Think again or try a new brain teaser!" },
                { title: "Oops! 😊", text: "That's not it, but keep trying!" },
                { title: "Almost! 😉", text: "You're getting warmer! Try again!" },
                { title: "Not this time! 🙃", text: "Don't give up, you've got this!" },
                { title: "Close! 🎯", text: "Think a bit more and try again!" },
                { title: "Nope! 😄", text: "That's okay, everyone learns differently!" }
            ];
            
            const fail = failMessages[Math.floor(Math.random() * failMessages.length)];
            $('#failResult h4').text(fail.title);
            $('#failResult p').text(fail.text);
            $('#successResult').addClass('d-none');
            $('#failResult').removeClass('d-none');
            $(this.canvas).addClass('canvas-error shake');
            this.playSound('failSound');
            this.addEmojiReaction(['😅', '🤷', '😊', '🙃', '😉'][Math.floor(Math.random() * 5)]);
            
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
        // Fun questions with answers 0-9
        const challenges = [
            { question: "How many lives does a cat have minus one? 🐱", answer: 8 },
            { question: "How many legs does an ant have? 🐜", answer: 6 },
            { question: "How many fingers on one hand? ✋", answer: 5 },
            { question: "What comes before 1? 🤔", answer: 0 },
            { question: "How many sides does a triangle have? 📐", answer: 3 },
            { question: "How many wheels on a bicycle? 🚲", answer: 2 },
            { question: "How many days in a week? 📅", answer: 7 },
            { question: "How many legs does a spider have? 🕷️", answer: 8 },
            { question: "Lucky number that rhymes with 'mine'? 🍀", answer: 9 },
            { question: "How many sides does a square have? ⬜", answer: 4 },
            { question: "What's half of two? ➗", answer: 1 },
            { question: "How many eyes do you have? 👀", answer: 2 },
            { question: "How many noses does a human have? 👃", answer: 1 },
            { question: "How many sides does a pentagon have? ⭐", answer: 5 },
            { question: "How many strings on a guitar? 🎸", answer: 6 },
            { question: "How many colors in a rainbow? 🌈", answer: 7 },
            { question: "How many planets from the sun is Earth? 🌍", answer: 3 },
            { question: "How many minutes in half an hour divided by 10? ⏰", answer: 3 },
            { question: "How many seasons in a year? 🍂", answer: 4 },
            { question: "What digit looks like a snake? 🐍", answer: 2 },
            { question: "How many thumbs on both hands? 👍", answer: 2 },
            { question: "What number rhymes with 'heaven'? ☁️", answer: 7 },
            { question: "How many lives does a cat really have? (mythology) 🐾", answer: 9 },
            { question: "What comes after 8? ➡️", answer: 9 },
            { question: "How many wheels on a tricycle? 🚴", answer: 3 },
            { question: "What's the loneliest number? 😢", answer: 1 },
            { question: "How many tentacles does an octopus have? 🐙", answer: 8 },
            { question: "What do you get when you eat zero cookies? 🍪", answer: 0 },
            { question: "How many horns does a unicorn have? 🦄", answer: 1 },
            { question: "How many lives does a video game character usually have? 🎮", answer: 3 },
            { question: "How many points does a star usually have? ⭐", answer: 5 },
            { question: "How many wings does a butterfly have? 🦋", answer: 4 },
            { question: "What's the first digit in your phone number? (just kidding!) What comes after 5? 📱", answer: 6 },
            { question: "How many continents are there? 🌍", answer: 7 },
            { question: "What's 5 + 4? 🧮", answer: 9 },
            { question: "How many holes in a donut? 🍩", answer: 1 },
            { question: "What's 10 - 10? ➖", answer: 0 },
            { question: "How many beats in a waltz? 🎵", answer: 3 },
            { question: "How many chambers in a human heart? ❤️", answer: 4 },
            { question: "How many Olympic rings? 🏅", answer: 5 },
            { question: "What's the square root of 4? 📏", answer: 2 },
            { question: "How many deadly sins? 😈", answer: 7 },
            { question: "What's 2 × 4? ✖️", answer: 8 },
            { question: "How many letters in 'cat'? 🐱", answer: 3 },
            { question: "What's the loneliest number that you'll ever do? 🎵", answer: 1 },
            { question: "How many reindeer pull Santa's sleigh? (including Rudolph) 🦌", answer: 9 },
            { question: "What comes before 'teen' numbers? 🔢", answer: 9 },
            { question: "How many sides on a dice? 🎲", answer: 6 },
            { question: "What's half of 16? ➗", answer: 8 },
            { question: "How many musicians in a quartet? 🎼", answer: 4 },
            { question: "How many players on a basketball team on court? 🏀", answer: 5 }
        ];
        
        const challenge = challenges[Math.floor(Math.random() * challenges.length)];
        this.currentChallenge = challenge.answer;
        const displayAnswer = challenge.displayAnswer !== undefined ? challenge.displayAnswer : challenge.answer;
        
        // Update the UI with the question
        $('#challengeInstructions .alert-heading').html('<i class="fas fa-brain"></i> Brain Teaser:');
        $('#challengeInstructions p').html(`${challenge.question}<br><strong>Draw your answer:</strong>`);
        $('#targetDigit').text(displayAnswer).addClass('pulse');
        
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