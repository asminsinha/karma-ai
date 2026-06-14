import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.services.ml_service import ml_service
    print("\n Verification Successful! FastAPI model loading system maps perfectly to binaries.")
except Exception as e:
    print(f"\n Verification Failed: {str(e)}")