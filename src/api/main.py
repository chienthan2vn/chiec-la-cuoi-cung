from fastapi import FastAPI, File, UploadFile, HTTPException
from src.model.inference_handler import InferenceHandler
from src.api.schemas import PredictionResponse, ErrorResponse
import uvicorn

app = FastAPI(
    title="Plant Disease Classification API",
    description="API to predict plant diseases using a model registered in MLflow",
    version="1.0.0"
)

# Initialize handler
handler = InferenceHandler()

@app.get("/")
async def root():
    return {"message": "Plant Disease Classification API is running"}

@app.get("/health")
async def health():
    if handler.model is not None:
        return {"status": "healthy", "model": handler.model_name}
    raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict", response_model=PredictionResponse, responses={500: {"model": ErrorResponse}})
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    try:
        content = await file.read()
        result = handler.predict(content)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
