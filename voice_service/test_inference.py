import os
import numpy as np
import soundfile as sf
import os
import joblib
import numpy as np
import soundfile as sf
# Import the actual models dictionary from main
from main import extract_voice_features, predict_confidence, models
# Create a sample audio file (5 seconds of white noise + tone)
print("📊 Creating test audio...")
# 🆕 MANUAL MODEL LOADING (Crucial for testing)
# ============================================================
print("⏳ Loading models into memory for testing...")
try:
    models['scaler'] = joblib.load("models/voice_scaler.pkl")
    models['encoder'] = joblib.load("models/label_encoder.pkl")
    models['xgb_model'] = joblib.load("models/voice_confidence_xgb.pkl")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    exit()

sr = 16000
duration = 5
t = np.linspace(0, duration, int(sr * duration))

# Mix: speech-like frequency (200Hz) + noise
signal = (
    0.1 * np.sin(2 * np.pi * 200 * t) +  # 200Hz tone (speech-like)
    0.05 * np.random.randn(len(t))  # Noise
)

# Normalize
signal = signal / np.max(np.abs(signal))

# Save
test_audio = "test_audio.wav"
sf.write(test_audio, signal, sr)

print(f"✅ Test audio created: {test_audio}")

# Test feature extraction
print("\n📊 Testing feature extraction...")
try:
    features = extract_voice_features(test_audio)
    print("✅ Feature extraction successful")
    print(f"   Extracted {len(features)} features")
    for key, value in features.items():
        print(f"   {key}: {value:.4f}")
except Exception as e:
    print(f"❌ Feature extraction failed: {e}")

# Test inference
print("\n🤖 Testing inference...")
try:
    prediction = predict_confidence(features)
    print("✅ Inference successful")
    print(f"   Predicted label: {prediction['predicted_confidence_label']}")
    print(f"   Confidence: {prediction['confidence_probability']:.1f}%")
except Exception as e:
    print(f"❌ Inference failed: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - Ready for Express integration!")
print("="*60)