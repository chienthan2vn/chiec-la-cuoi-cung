import torch
import torchvision.transforms as transforms
from PIL import Image
import mlflow.pytorch
import os
from dotenv import load_dotenv
import io

load_dotenv()

class InferenceHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = os.getenv("MLFLOW_MODEL_NAME")
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Load the model using the alias 'lastest' (as seen in your UI screenshot)
        # For aliases, use the '@' syntax: models:/<model_name>@<alias>
        model_uri = f"models:/{self.model_name}@lastest"
        print(f"Loading model from: {model_uri}")
        
        try:
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # Standard ResNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Placeholder for class names - in production, these should be loaded from an artifact
        self.class_names = None 

    def preprocess(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image_bytes):
        if self.model is None:
            return {"error": "Model not loaded"}
            
        tensor = self.preprocess(image_bytes)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred = torch.max(probabilities, 0)
            
        result = {
            "class_id": int(pred.item()),
            "confidence": float(conf.item())
        }
        
        if self.class_names:
            result["class_name"] = self.class_names[result["class_id"]]
            
        return result
