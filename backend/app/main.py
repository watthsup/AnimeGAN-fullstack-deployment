from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from app.inferencer import ModelInferencer
from app.config import api_config

import time
import io
from PIL import Image
import numpy as np

app = FastAPI(title = "Inferencing API Gateway", version="0.1", docs_url= api_config['api_prefix'] + '/docs')
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
)

if api_config['inference_engine'] == 'triton':
    model_url = f"{api_config['model_ip']}:{api_config['model_port']}"
    inferencer = ModelInferencer(model_name=api_config['model_name'], engine=api_config['inference_engine'], url=model_url)
if api_config['inference_engine'] == 'onnxruntime' or api_config['inference_engine'] == 'onnxruntime-gpu':    
    inferencer = ModelInferencer(model_name=api_config['model_name'], engine=api_config['inference_engine'])

@app.post('/models/infer', tags=["inferencing"])
def inference_request(file: UploadFile = File(...), model_name:str=None):
    try:
        content = file.file.read()
        np_array = np.fromstring(content, np.uint8)
        inputImage = Image.open(io.BytesIO(np_array))
        outputImage = inferencer.infer(inputImage, model_name)
        bytesImage = io.BytesIO()
        outputImage.save(bytesImage, format='PNG')

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
        
    return Response(content = bytesImage.getvalue(), media_type="image/png")