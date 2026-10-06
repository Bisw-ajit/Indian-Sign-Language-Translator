import os
import sys
import json
import base64
import time
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# Ensure App directory is in path for imports
app_dir = os.path.join(os.path.dirname(__file__), 'App')
if app_dir not in sys.path:
    sys.path.append(app_dir)

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

from preprocessing import skinDetector, resize

app = Flask(__name__)

# Load Model & Classes
MODEL_PATH = os.path.join(app_dir, 'final_model.h5') if os.path.exists(os.path.join(app_dir, 'final_model.h5')) else os.path.join(app_dir, 'final_model')
CLASS_JSON_PATH = os.path.join(app_dir, 'model_class.json')

print("Loading ISL SqueezeNet model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_JSON_PATH, 'r') as f:
    CLASS_INDEX = json.load(f)
print("Model loaded successfully! Supported classes:", list(CLASS_INDEX.values()))

# Check for YOLO
yolo_detector = None
weights_path = os.path.join(app_dir, 'yolo_models', 'cross-hands.weights')
cfg_path = os.path.join(app_dir, 'yolo_models', 'cross-hands.cfg')

if os.path.exists(weights_path) and os.path.exists(cfg_path):
    try:
        from yolo import YOLO
        yolo_detector = YOLO(cfg_path, weights_path, ["hand"])
        print("YOLO hand detector initialized successfully.")
    except Exception as e:
        print(f"YOLO initialization skipped: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indian Sign Language Translator - Web Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --panel-bg: rgba(18, 26, 43, 0.7);
            --border-glow: rgba(56, 189, 248, 0.2);
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.4);
            --accent: #818cf8;
            --accent-pink: #f472b6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --card-border: rgba(255, 255, 255, 0.08);
            --success: #34d399;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(129, 140, 248, 0.12) 0%, transparent 40%);
            overflow-x: hidden;
        }

        header {
            padding: 1.25rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--card-border);
            backdrop-filter: blur(12px);
            background: rgba(9, 13, 22, 0.8);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 700;
            font-size: 1.35rem;
            letter-spacing: -0.02em;
        }

        .logo-icon {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px var(--primary-glow);
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            background: rgba(52, 211, 153, 0.1);
            border: 1px solid rgba(52, 211, 153, 0.3);
            color: var(--success);
            font-size: 0.85rem;
            font-weight: 500;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
        }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .glass-panel {
            background: var(--panel-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .video-container {
            position: relative;
            width: 100%;
            aspect-ratio: 16 / 9;
            border-radius: 16px;
            overflow: hidden;
            background: #000;
            border: 1px solid var(--border-glow);
            box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.8);
        }

        video, canvas {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        #outputCanvas {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }

        .controls-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .btn {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: #fff;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-family: inherit;
            font-size: 0.95rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px var(--primary-glow);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            box-shadow: none;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.12);
            box-shadow: none;
        }

        .btn-danger {
            background: linear-gradient(135deg, #ef4444, #f43f5e);
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
        }

        .prediction-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 16px;
            padding: 1.5rem;
        }

        .hero-sign {
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }

        .sign-badge {
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.2rem;
            font-weight: 700;
            box-shadow: 0 0 25px var(--primary-glow);
        }

        .sign-info h2 {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .sign-info p {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .confidence-circle {
            text-align: right;
        }

        .confidence-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
        }

        .confidence-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .segmented-preview {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 12px;
            border: 1px solid var(--card-border);
        }

        .segmented-img {
            width: 80px;
            height: 80px;
            border-radius: 8px;
            object-fit: cover;
            border: 1px solid var(--primary-glow);
        }

        .prob-bars {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }

        .prob-row {
            display: grid;
            grid-template-columns: 30px 1fr 50px;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.9rem;
        }

        .prob-bar-bg {
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            overflow: hidden;
        }

        .prob-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            border-radius: 9999px;
            transition: width 0.3s ease;
        }

        .sentence-box {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .sentence-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .sentence-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            min-height: 48px;
            background: rgba(0, 0, 0, 0.4);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            border: 1px dashed var(--border-glow);
            color: var(--primary);
            word-break: break-all;
        }

        .dict-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.6rem;
        }

        .dict-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 0.75rem;
            text-align: center;
            transition: all 0.2s ease;
        }

        .dict-card.active {
            border-color: var(--primary);
            background: rgba(56, 189, 248, 0.15);
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .dict-char {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .dict-name {
            font-size: 0.7rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">
            <div class="logo-icon">✋</div>
            <span>Indian Sign Language AI Studio</span>
        </div>
        <div class="status-badge">
            <div class="pulse-dot"></div>
            <span>Model Active (SqueezeNet + Skin Segmentation)</span>
        </div>
    </header>

    <main>
        <!-- Left Column: Video Feed & Controls -->
        <section class="glass-panel">
            <div class="video-container">
                <video id="webcam" autoplay playsinline muted></video>
                <canvas id="canvas" style="display:none;"></canvas>
            </div>

            <div class="controls-bar">
                <button id="toggleCamBtn" class="btn">
                    <span>📹 Stop Camera</span>
                </button>
                <button id="toggleDetectBtn" class="btn btn-secondary">
                    <span>⚡ Real-Time Predict: ON</span>
                </button>
                <button id="captureBtn" class="btn btn-secondary">
                    <span>📸 Capture Frame</span>
                </button>
            </div>

            <!-- Skin Segmentation Preview -->
            <div class="segmented-preview">
                <img id="segImg" class="segmented-img" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80'><rect width='80' height='80' fill='%23111'/><text x='50%' y='50%' fill='%23666' dominant-baseline='middle' text-anchor='middle' font-size='10'>Filter Preview</text></svg>" alt="Segmented Skin">
                <div>
                    <h4 style="font-size:0.95rem; font-weight:600;">HSV + YCbCr Skin Filter</h4>
                    <p style="font-size:0.8rem; color:var(--text-muted);">Real-time background noise removal pipeline</p>
                </div>
            </div>
        </section>

        <!-- Right Column: Predictions & Output -->
        <section class="glass-panel">
            <!-- Hero Prediction -->
            <div class="prediction-hero">
                <div class="hero-sign">
                    <div id="signBadge" class="sign-badge">-</div>
                    <div class="sign-info">
                        <h2 id="signTitle">Waiting...</h2>
                        <p id="signSubtitle">Hold up an ISL hand gesture</p>
                    </div>
                </div>
                <div class="confidence-circle">
                    <div id="confValue" class="confidence-value">0%</div>
                    <div class="confidence-label">Confidence</div>
                </div>
            </div>

            <!-- Probability Bars -->
            <div>
                <h4 style="font-size:0.9rem; color:var(--text-muted); margin-bottom:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">Top Class Probabilities</h4>
                <div id="probBars" class="prob-bars">
                    <div class="prob-row"><span>G</span><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:0%"></div></div><span>0%</span></div>
                    <div class="prob-row"><span>I</span><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:0%"></div></div><span>0%</span></div>
                    <div class="prob-row"><span>K</span><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:0%"></div></div><span>0%</span></div>
                    <div class="prob-row"><span>O</span><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:0%"></div></div><span>0%</span></div>
                    <div class="prob-row"><span>P</span><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:0%"></div></div><span>0%</span></div>
                </div>
            </div>

            <!-- Recognized Sentence Builder -->
            <div class="sentence-box">
                <div class="sentence-header">
                    <h4 style="font-size:0.9rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em;">Translated Sentence</h4>
                    <div style="display:flex; gap:0.5rem;">
                        <button id="speakBtn" class="btn btn-secondary" style="padding:0.4rem 0.75rem; font-size:0.8rem;">🔊 Speak</button>
                        <button id="clearBtn" class="btn btn-secondary" style="padding:0.4rem 0.75rem; font-size:0.8rem;">🧹 Clear</button>
                    </div>
                </div>
                <div id="sentenceText" class="sentence-text"></div>
            </div>

            <!-- Supported Signs Grid -->
            <div>
                <h4 style="font-size:0.9rem; color:var(--text-muted); margin-bottom:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">Supported ISL Vocabulary (10 Signs)</h4>
                <div id="dictGrid" class="dict-grid">
                    <div class="dict-card" data-sign="G"><div class="dict-char">G</div><div class="dict-name">Sign G</div></div>
                    <div class="dict-card" data-sign="I"><div class="dict-char">I</div><div class="dict-name">Sign I</div></div>
                    <div class="dict-card" data-sign="K"><div class="dict-char">K</div><div class="dict-name">Sign K</div></div>
                    <div class="dict-card" data-sign="O"><div class="dict-char">O</div><div class="dict-name">Sign O</div></div>
                    <div class="dict-card" data-sign="P"><div class="dict-char">P</div><div class="dict-name">Sign P</div></div>
                    <div class="dict-card" data-sign="S"><div class="dict-char">S</div><div class="dict-name">Sign S</div></div>
                    <div class="dict-card" data-sign="U"><div class="dict-char">U</div><div class="dict-name">Sign U</div></div>
                    <div class="dict-card" data-sign="V"><div class="dict-char">V</div><div class="dict-name">Sign V</div></div>
                    <div class="dict-card" data-sign="X"><div class="dict-char">X</div><div class="dict-name">Sign X</div></div>
                    <div class="dict-card" data-sign="Y"><div class="dict-char">Y</div><div class="dict-name">Sign Y</div></div>
                </div>
            </div>
        </section>
    </main>

    <script>
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const toggleCamBtn = document.getElementById('toggleCamBtn');
        const toggleDetectBtn = document.getElementById('toggleDetectBtn');
        const captureBtn = document.getElementById('captureBtn');
        const segImg = document.getElementById('segImg');
        
        const signBadge = document.getElementById('signBadge');
        const signTitle = document.getElementById('signTitle');
        const signSubtitle = document.getElementById('signSubtitle');
        const confValue = document.getElementById('confValue');
        const probBars = document.getElementById('probBars');
        
        const sentenceText = document.getElementById('sentenceText');
        const speakBtn = document.getElementById('speakBtn');
        const clearBtn = document.getElementById('clearBtn');

        let isCamOn = false;
        let isDetecting = true;
        let stream = null;
        let lastPredicted = '';
        let lastPredictTime = 0;
        let predictedSequence = [];

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 864, height: 480, facingMode: 'user' }
                });
                video.srcObject = stream;
                isCamOn = true;
                toggleCamBtn.innerHTML = '<span>📹 Stop Camera</span>';
                toggleCamBtn.classList.remove('btn-secondary');
            } catch (err) {
                console.error("Camera access error:", err);
                alert("Webcam access denied or unavailable: " + err.message);
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                isCamOn = false;
                toggleCamBtn.innerHTML = '<span>📹 Start Camera</span>';
                toggleCamBtn.classList.add('btn-secondary');
            }
        }

        toggleCamBtn.addEventListener('click', () => {
            if (isCamOn) stopCamera();
            else startCamera();
        });

        toggleDetectBtn.addEventListener('click', () => {
            isDetecting = !isDetecting;
            toggleDetectBtn.innerHTML = isDetecting ? '<span>⚡ Real-Time Predict: ON</span>' : '<span>⏸️ Real-Time Predict: OFF</span>';
            toggleDetectBtn.classList.toggle('btn-secondary', !isDetecting);
        });

        let isSending = false;
        async function sendFrame() {
            if (!isCamOn || !isDetecting || video.videoWidth === 0 || isSending) return;
            isSending = true;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });

                const data = await res.json();
                if (data.success) {
                    updateUI(data);
                }
            } catch (err) {
                console.error("Prediction error:", err);
            } finally {
                isSending = false;
            }
        }

        function updateUI(data) {
            const topSign = data.sign;
            const topConf = data.confidence;

            signBadge.innerText = topSign;
            signTitle.innerText = `Sign "${topSign}" Recognized`;
            signSubtitle.innerText = `Matched ISL Class ${topSign}`;
            confValue.innerText = `${topConf}%`;

            if (data.segmented_image) {
                segImg.src = data.segmented_image;
            }

            // Draw bounding box overlay on video
            const outCanvas = document.getElementById('outputCanvas');
            if (outCanvas) {
                outCanvas.width = video.videoWidth;
                outCanvas.height = video.videoHeight;
                const outCtx = outCanvas.getContext('2d');
                outCtx.clearRect(0, 0, outCanvas.width, outCanvas.height);
                if (data.box) {
                    outCtx.strokeStyle = '#34d399';
                    outCtx.lineWidth = 3;
                    outCtx.strokeRect(data.box.x, data.box.y, data.box.width, data.box.height);
                    outCtx.fillStyle = '#34d399';
                    outCtx.font = 'bold 16px Outfit, sans-serif';
                    outCtx.fillText(`${data.sign} (${data.confidence}%)`, data.box.x + 4, Math.max(22, data.box.y - 8));
                }
            }

            // Update top probability bars
            if (data.top_predictions) {
                probBars.innerHTML = data.top_predictions.map(p => `
                    <div class="prob-row">
                        <span>${p.class}</span>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${p.score}%"></div>
                        </div>
                        <span>${p.score}%</span>
                    </div>
                `).join('');
            }

            // Highlight in Dictionary Grid
            document.querySelectorAll('.dict-card').forEach(card => {
                card.classList.toggle('active', card.dataset.sign === topSign);
            });

            // Sentence Builder logic with thresholding
            const now = Date.now();
            if (topConf > 70 && topSign !== lastPredicted && (now - lastPredictTime > 1200)) {
                predictedSequence.push(topSign);
                sentenceText.innerText = predictedSequence.join(' ');
                lastPredicted = topSign;
                lastPredictTime = now;
            }
        }

        captureBtn.addEventListener('click', () => {
            sendFrame();
        });

        clearBtn.addEventListener('click', () => {
            predictedSequence = [];
            sentenceText.innerText = '';
            lastPredicted = '';
        });

        speakBtn.addEventListener('click', () => {
            const text = sentenceText.innerText;
            if (!text) return;
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-IN';
            window.speechSynthesis.speak(utterance);
        });

        // Initialize camera & start prediction loop
        startCamera();
        setInterval(sendFrame, 400);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        # Decode base64 image
        header, encoded = data['image'].split(',', 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400

        ih, iw = img_bgr.shape[:2]
        cropped_img = img_bgr
        box_data = None

        # Optional YOLO hand detection & bounding box crop (mirrors App/main.py pipeline)
        if yolo_detector is not None:
            try:
                _, _, _, results = yolo_detector.inference(img_bgr)
                if len(results) == 1:
                    _, _, _, bx, by, bw, bh = results[0]
                    bx1, by1 = bx + bw, by + bh
                    x = max(0, bx - 54)
                    y = max(0, by - 30)
                    x1 = min(iw, bx1 + 54)
                    y1 = min(ih, by1 + 30)
                    if x1 > x and y1 > y:
                        cropped_img = img_bgr[y:y1, x:x1]
                        box_data = {'x': int(x), 'y': int(y), 'width': int(x1 - x), 'height': int(y1 - y)}
                elif len(results) >= 2:
                    _, _, _, bx1_a, by1_a, bw1_a, bh1_a = results[0]
                    _, _, _, bx2_a, by2_a, bw2_a, bh2_a = results[1]
                    bx = min(bx1_a, bx2_a)
                    by = min(by1_a, by2_a)
                    bx1 = max(bx1_a + bw1_a, bx2_a + bw2_a)
                    by1 = max(by1_a + bh1_a, by2_a + bh2_a)
                    x = max(0, bx - 54)
                    y = max(0, by - 30)
                    x1 = min(iw, bx1 + 54)
                    y1 = min(ih, by1 + 30)
                    if x1 > x and y1 > y:
                        cropped_img = img_bgr[y:y1, x:x1]
                        box_data = {'x': int(x), 'y': int(y), 'width': int(x1 - x), 'height': int(y1 - y)}
            except Exception as e:
                print(f"YOLO detection warning during frame prediction: {e}")

        # Skin segmentation & preprocessing
        skin_detector = skinDetector(cropped_img)
        segmented_img = skin_detector.find_skin()

        # Resize to 224x224
        resized_img = resize.resize_image(segmented_img, (224, 224))

        # Prepare for Keras SqueezeNet model
        prep_img = resized_img.astype(np.float32) * (1.0 / 255.0)
        prep_img = np.expand_dims(prep_img, axis=0)

        # Run prediction
        predictions = model.predict(prep_img, verbose=0)[0]

        # Process prediction results
        top_indices = predictions.argsort()[-5:][::-1]
        top_preds = []
        for i in top_indices:
            top_preds.append({
                'class': CLASS_INDEX[str(i)],
                'score': round(float(predictions[i]) * 100, 1)
            })

        top_sign = top_preds[0]['class']
        top_conf = top_preds[0]['score']

        # Encode segmented image to base64 for live UI preview
        _, buffer = cv2.imencode('.jpg', resized_img)
        seg_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'sign': top_sign,
            'confidence': top_conf,
            'box': box_data,
            'top_predictions': top_preds,
            'segmented_image': seg_base64
        })

    except Exception as e:
        print("Prediction API Error:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting ISL Web App on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
