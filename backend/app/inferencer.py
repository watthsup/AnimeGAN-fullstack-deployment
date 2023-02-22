import numpy as np
import onnxruntime as rt
import tritonclient.grpc as client
import torch
import cv2
from .utils import *
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

class ModelInferencer():
    def __init__(self, model_name:str, engine:str = 'triton', url:str=None):
        self.engine = engine
        self.model_name = model_name
        self.url = url
        if self.engine == 'onnxruntime' or self.engine == 'onnxruntime-gpu':
            sessOpt = rt.SessionOptions()
            sessOpt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessOpt.add_session_config_entry("session.set_denormal_as_zero", "1")
            
            if self.engine == 'onnxruntime':
                providers=['CPUExecutionProvider']
            else:
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']

            self.inferencer = rt.InferenceSession(f'model_repository/{self.model_name}/1/model.onnx', sessOpt, providers=providers)
        elif self.engine == 'triton':
            self.tritonConnector = client.InferenceServerClient(url=self.url)
            
        self.face_detector = get_dlib_face_detector()
    
    def infer(self, pil_img, model_name=None):
        if self.engine == 'onnxruntime' or self.engine == 'onnxruntime-gpu':
            input_batch, face_im = self.preprocess(pil_img)
            onnx_output = self.inferencer.run(None, {'input' : input_batch.numpy().astype(np.float32)})
            onnx_output = torch.FloatTensor(np.array(onnx_output)).reshape(1,-1)
            results = self.postprocess(onnx_output, face_im)
            
        elif self.engine == 'triton':
            input_batch, face_im = self.preprocess(pil_img)
            input0 = client.InferInput('input', (1, 3, 512, 512), "FP32")
            input0.set_data_from_numpy(input_batch.numpy())
            output = self.tritonConnector.infer(model_name=model_name, inputs=[input0])
            triton_output = output.as_numpy('output')
            results = self.postprocess(triton_output, face_im, model_name)
            
        return results
        
    def preprocess(self, pil_img, size=512):
        landmarks = self.face_detector(pil_img)
        face = align_and_crop_face(pil_img, landmarks[0], expand=1.3)
        w, h = face.size
        s = min(w, h)
        img = face.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.LANCZOS)
        input_batch = to_tensor(img).unsqueeze(0) * 2 - 1
        return input_batch, img
        
    def postprocess(self, output, face_im, model_name):
        if self.engine == 'onnxruntime' or self.engine == 'onnxruntime-gpu':
            onnx_output = output
            onnx_output = torch.FloatTensor(np.array(onnx_output))
            onnx_output = (onnx_output * 0.5 + 0.5).clip(0, 1)
            onnx_output = onnx_output.reshape(3,512,512)

            anime_im = to_pil_image(onnx_output)
            concat = Image.new('RGB', (face_im.width + anime_im.width, face_im.height))
            concat.paste(face_im, (0, 0))
            concat.paste(anime_im, (face_im.width, 0))
            return concat           
        elif self.engine == 'triton':
            triton_output = output
            triton_output = (triton_output * 0.5 + 0.5).clip(0, 1)
            triton_output = triton_output.reshape(3,512,512)
            triton_output = torch.FloatTensor(triton_output)

            anime_im = to_pil_image(triton_output)
            concat = Image.new('RGB', (face_im.width + anime_im.width, face_im.height))
            concat.paste(face_im, (0, 0))
            concat.paste(anime_im, (face_im.width, 0))
            return concat
            