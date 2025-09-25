# tests/test_xray_ml_model.py
# predict.py
import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

# Import our model-creation function from the other file
from features.xray_predictor.model import create_model, DISEASE_LABELS

BASE_DIR = os.path.dirname(__file__)

def predict_image(image_path, model_weights_path):
    """
    Loads the trained model, processes an image, and returns the predictions.
    """
    # --- 1. Setup ---
    device = torch.device("cpu") # Run on CPU
    model = create_model()
    
    # Load the saved weights into the model structure
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    
    # Set the model to evaluation mode (very important!)
    model.eval()

    # --- 2. Image Transformations ---
    # These must be the SAME as the validation transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 3. Prediction ---
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply sigmoid to get probabilities between 0 and 1
        probabilities = torch.sigmoid(outputs)

    # --- 4. Format Results ---
    results = {DISEASE_LABELS[i]: f"{probabilities[0][i].item()*100:.2f}%" for i in range(len(DISEASE_LABELS))}
    return results

if __name__ == "__main__":
    # This allows us to run the script from the command line
    parser = argparse.ArgumentParser(description="Predict diseases from an X-ray image.")
    parser.add_argument("image", help="Path to the X-ray image file.")
    args = parser.parse_args()


    model_path = os.path.join(BASE_DIR, 'weights', 'best_model.pth')
    
    print(f"🔬 Analyzing image: {args.image}")
    predictions = predict_image(args.image, model_path)
    
    print("\n--- 🩺 Prediction Results ---")
    for disease, prob in predictions.items():
        print(f"{disease}: {prob}")
