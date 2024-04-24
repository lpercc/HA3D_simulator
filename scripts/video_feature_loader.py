# Now 我们构建一个 Video Feature Loader 类, 这个类的作用是读取原视频, 并且提取视频的特征. 
# Firstly we load video, then we use the model to extract the features. 

from urllib.request import urlretrieve
import torch
import torch.nn as nn
import torchvision.models as models
import timm 
import torchvision.transforms as transforms
import torchvision
import tempfile
import os
from pathlib import Path
from abc import ABC, abstractmethod
from torchvision.transforms import Compose
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import cv2
import numpy as np
from PIL import Image
from transformers import VivitImageProcessor, VivitModel
#from models.transforms import (CenterCrop, Normalize, Resize, ToFloatTensorInZeroOne) #TODO for R12DExtractor

class BaseFrameExtractor(ABC):
    def __init__(self, model_name, fps, device, fuse='nomean') -> None:
        self.model_name = model_name
        self.fps = fps
        self.device = device
        self.fuse = fuse
        #self.video = None
        #self.model = self._load_model()
    @abstractmethod
    def _load_model(self):
        raise NotImplementedError
    
    
    def _load_video(self):
        try:
            self.video = self.video.transpose(0, 2, 1, 3) # (T, W, H, C) -> (T, H, W, C) numpy array use transpose, torch use permute
        except AttributeError:
            raise ValueError("Please load the video first.")
    
    def load_video(self, video):
        # input video is a numpy array with (T, W, H ,C)
        self.video = video
    
    def _clip(self):
        #self._load_video()
        # Get the original dimensions
        T, H, W, C = self.video.shape
        
        if T == self.fps:
            return self.video
        
        # Calculate the number of original frames to combine for one target frame
        group_size = T // self.fps
        
        # Initialize an empty array for the compressed video
        compressed_video = np.zeros((self.fps, H, W, C), dtype=np.uint8)
        
        if self.fuse == 'mean':
            print('Warning: Now you are using fuse method: {} to extract frames.'.format(self.fuse))
            for i in range(self.fps):
                # Determine the start and end frame indices for averaging
                
                start_idx = i * group_size
                end_idx = start_idx + group_size
                
                # Average the frames in the group and assign to the compressed video
                # Note: Ensure the result is of type uint8 to match image data type
                # Fuse to extract 
                compressed_video[i] = np.mean(self.video[start_idx:end_idx], axis=0).astype(np.uint8)
        else:
            # 等间隔抽取出指定的帧数
            # 使用 np.linspace 函数生成等间隔的索引
            # 使用索引从原始视频中抽取帧
            #print("Use intervals to extract frames.")
            #indices = np.linspace(0, T, self.fps, endpoint=False ,dtype=np.uint8)
            compressed_video = self.video
        
        return compressed_video

    @torch.no_grad()
    def extract_features(self, keep_T=False):
        frames = self._clip() # numpy array 
        # convert to torch tennsor 
        #frames = torch.tensor(frames, dtype=torch.float32).to(self.device)
        transformed_frames = torch.stack([self.transforms(frame) for frame in frames])
        self.features = self.model(transformed_frames.to(self.device)).squeeze()
        if keep_T:
            return self.features.cpu().numpy()
        else:   
            return self.features.mean(0).cpu().numpy()

class TorchVisionExtractor(BaseFrameExtractor):
    def __init__(self, model_name, weights_key, fps, device) -> None:
        super().__init__(model_name, fps, device)
        self.weights_key = weights_key
        self.model = self._load_model()
        self._test_model()
    
    def _load_model(self):
        model = models.get_model(self.model_name, weights=self.weights_key)
        self.transforms = transforms.Compose([
            # For torch vision, we should use a tensor image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,  0.456,  0.406], std=[0.229,  0.224,  0.225])
        ])
        model = model.to(self.device)
        model.eval()
        return model



class TimmExtractor(BaseFrameExtractor):
    def __init__(self, model_name, fps, device, fuse) -> None:
        super().__init__(model_name, fps, device, fuse)
        self.model = self._load_model()
    
    def _load_model(self):
        model = timm.create_model(self.model_name, pretrained=True)
        self.transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model, verbose=True))
        # for timm we should use a PIL image
        self.transforms = Compose([transforms.ToPILImage(), self.transforms])
        model.reset_classifier(0, '')
        model = model.to(self.device)
        model.eval()
        return model
    def _test_model(self):
        # Test the model with a dummy input
        dummy_input = torch.randn(1,4, 3, 224, 224)
        output = self.model(dummy_input)
        print(output.shape)
    
    def _test_model(self):
        # Test the model with a random input
        x = torch.randn(1, 3, 224, 224)
        outputs = self.model(x)
        assert outputs.dim() == 4, "Model output should be 4D tensor, (Batch, C, H, W)"
        assert outputs.size(1) == 2048, "Resnet152 should output 2048 channels"
        
    
    
class HieraExtractor(BaseFrameExtractor):
    # github model page: https://github.com/facebookresearch/hiera?tab=readme-ov-file
    def __init__(self, model_name, fps, device) -> None:
        super().__init__(model_name, fps, device)
        self.fps = fps
        assert self.fps == 16, "Hiera model only support 16 fps"
        self.model = self._load_model()
    
    def _load_model(self):
        model = torch.hub.load("facebookresearch/hiera", model=self.model_name, pretrained=True, checkpoint="mae_k400_ft_k400")
        # each hiera model inputs 16 224x224 frames with a temporal stride of 4
        # input should be (1, C, T, H, W))
        # we use middle stages here 
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]) #TODO we need to check the mean and std
        ])
        
        
        model = model.to(self.device)
        model.eval()
        
        return model 
        
    @torch.no_grad()
    def extract_features(self, keep_T=False):
        frames = self._clip() # numpy array
        assert len(frames) == 16, "Hiera model only support 16 fps"
        frames = torch.Tensor(frames) #we need to convert to tensor to use the transforms because we use Resize
        transformed_frames = torch.stack([self.transforms(frame.permute(2,0,1)) for frame in frames])
        # we need to permute the tensor to (1, C, T, H, W), orginal is (T, C, H, W)
        transformed_frames = transformed_frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        
        _, intermediate_outputs = self.model(transformed_frames, return_intermediates=True)
        
        # we use middle stages here 
        # The output will be torch.Size([1, 8, 14, 14, 384]), B, T, H, W, C
        self.features = intermediate_outputs[2].mean(2).mean(2).squeeze()
        
        if keep_T:
            return self.features.cpu().numpy()
        else:
            return self.features.mean(0).cpu().numpy() # output a 384 dim vector
        
        
class VivitExtractor(BaseFrameExtractor):
    # We use all transforer models from timm 
    # model name here should be "google/vivit-b-16x2-kinetics400"
    def __init__(self, model_name, fps, device) -> None:
        super().__init__(model_name, fps, device)
        assert self.fps == 32, "Vivit model orignally support 32 fps, can modify to suppoet lower or higher fps"
        self.model = self._load_model()
        
    def _load_model(self):
        model = VivitModel.from_pretrained(self.model_name, add_pooling_layer=True)
        image_processor = VivitImageProcessor.from_pretrained(self.model_name)
        
        self.transforms = image_processor 
        
        return model 
    
    @torch.no_grad()
    def extract_features(self, keep_T=False):
        frames = self._clip()
        inputs = self.transforms(list(frames), return_tensors="pt") #TODO 
        # input should be (batch_size, num_frames, num_channels, height, width), orginal is (T, C, H, W)
        
        # forward pass 
        outputs = self.model(**inputs)
        self.features = outputs.last_hidden_state
        
        return self.features.mean(1).squeeze().cpu().numpy()
        
        
class R12DExtractor(TorchVisionExtractor):
    # TODO 
    # We use all transforer models from timm 
    # model name here should be "facebookresearch/R2plus1D_18"
    def __init__(self, model_name, fps, device) -> None:
        super().__init__(model_name, fps, device)
        self.model = self._load_model()
        
    def _load_model(self):
        model = models.get_model(self.model_name, weights=self.weights_key)
        self.transforms = torchvision.transforms.Compose([
            ToFloatTensorInZeroOne(),
            Resize((128, 171)),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            CenterCrop((112, 112)),
        ])
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(self.device)
        model.eval()
        model.fc = torch.nn.Identity()
        return model
    
    @torch.no_grad()
    def extract_features(self, keep_T=False):
        frames = self._clip()
        inputs = frames.transpose(0, 3, 1, 2)
        inputs = self.transforms(frames).unsqueeze(0)
        # read video 1, T, H, W, C here, original is T, C, H, W
        # forward pass 
        outputs = self.model(inputs)
        self.features = outputs.mean(2).mean(2).mean(2).squeeze()
        
        return self.features.cpu().numpy()
    

if __name__ == '__main__':
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
    video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
    _ = urlretrieve(video_url, video_path)
    
    # Test a dummy numpy video array
    frame_num = 100
    WIDTH = 640
    HEIGHT = 480
    channels = 3
    
    # Create a dummy video as a NumPy array with random values
    # Shape: (frame_num, HEIGHT, WIDTH, channels)
    dummy_video = np.random.randint(60, 256, (frame_num, WIDTH, HEIGHT, channels), dtype=np.uint8)
    print(f'input video shape(Frames, H, W, C): {dummy_video.shape}')
    extractor = TimmExtractor('resnet152', 16, device, fuse='mean')
    extractor.load_video(dummy_video)
    output = extractor.extract_features(keep_T=True)
    print(f'output feature shape (Frames, C, H, W): {output.shape}')
    # IF you have a batch of video like (B,T,H,W,C). First reshapa to B*T, H, W, C then do not use mean fuse. 
    # After get output, you can reshape to B,T, C, H, W. then use your now clip frames method to get target fps
    # Your clip method can like _clip in BaseFrameExtractor class.
    
    '''extractor = HieraExtractor('hiera_base_16x224', 16, device)
    extractor.load_video(dummy_video)
    print(extractor.extract_features().shape)
    
    # Test VivitExtractor
    extractor = VivitExtractor('google/vivit-b-16x2-kinetics400', 32, device)
    extractor.load_video(dummy_video)
    #print(extractor.extract_features().shape)
    output = extractor.extract_features()
    print(output.shape)'''
    
    # Test R12DExtractor 
    #extractor = R12DExtractor('r2plus1d_18_16_kinetics', 16, device)
    #extractor.load_video(dummy_video)
    #print(extractor.extract_features().shape)