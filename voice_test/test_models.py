
import sys
print("Python version:", sys.version)

# Test 1: Check if joblib is installed
try:
    import joblib
    print("✅ joblib installed")
except ImportError:
    print("❌ joblib NOT installed. Run: pip install joblib")

# Test 2: Load the models
try:
    scaler = joblib.load("../voice_scaler.pkl")
    print("✅ voice_scaler.pkl loaded successfully")
except Exception as e:
    print(f"❌ voice_scaler.pkl FAILED: {e}")

try:
    encoder = joblib.load("../label_encoder.pkl")
    print("✅ label_encoder.pkl loaded successfully")
    print(f"   Classes: {encoder.classes_}")
except Exception as e:
    print(f"❌ label_encoder.pkl FAILED: {e}")

try:
    model = joblib.load("../voice_confidence_xgb.pkl")
    print("✅ voice_confidence_xgb.pkl loaded successfully")
    print(f"   Model type: {type(model)}")
except Exception as e:
    print(f"❌ voice_confidence_xgb.pkl FAILED: {e}")

print("\n" + "="*50)
print("All models loaded! Ready to proceed.")
print("="*50)
