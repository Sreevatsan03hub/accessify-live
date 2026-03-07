from services.sound_detection_service import get_sound_detector
import numpy as np

def test_sound_init():
    print("Testing SoundDetector Service...")
    
    # 1. Test Initialization (This will download YAMNet if not present)
    detector = get_sound_detector()
    
    if detector:
        print("✅ SoundDetector initialized successfully.")
    else:
        print("❌ SoundDetector failed to initialize.")
        return

    # 2. Test Processing Dummy Audio
    # YAMNet expects ~1s of 16kHz audio for optimal results (16000 samples)
    # We'll use random noise just to check execution flow
    dummy_audio = np.random.uniform(-1, 1, 16000).astype(np.float32)
    
    print("Running detection on dummy audio...")
    try:
        # Note: Random noise won't match any class with high confidence usually
        # But we just want to ensure NO CRASH
        result = detector.detect_sound(dummy_audio)
        
        if result:
            print(f"Result: {result}")
        else:
            print("No high-confidence sound detected (Expected for random noise).")
            
        print("✅ detector.detect_sound() executed without error.")
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")

if __name__ == "__main__":
    test_sound_init()
