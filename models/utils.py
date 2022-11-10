import segmentation_models_pytorch as smp
from models.encoder import SwinEncoder

def register_encoder():
    smp.encoders.encoders["swin_encoder"] = {
    "encoder": SwinEncoder, # encoder class here
    "pretrained_settings": { # pretrained 값 설정
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": { # 기본 파라미터
        "pretrain_img_size": 384,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        "window_size": 12,
        "drop_path_rate": 0.3,
    }
}

def get_model(model_str: str, config):
    """모델 클래스 변수 설정
    Args:
        model_str (str): 모델 클래스명
    Note:
        model_str 이 모델 클래스명으로 정의돼야함
        `model` 변수에 모델 클래스에 해당하는 인스턴스 할당
    """


    if model_str == 'Unet':
        return smp.Unet
    
    elif model_str == 'FPN':
        return smp.FPN

    elif model_str == 'DeepLabV3Plus':
        return smp.DeepLabV3Plus
    
    elif model_str == 'UnetPlusPlus':
        return smp.UnetPlusPlus

    elif model_str == 'PAN':
        return smp.PAN

    elif model_str == 'MAnet':
        return smp.MAnet

    elif model_str == 'PSPNet':
        return smp.PSPNet
    
    elif model_str == 'Swin':
        register_encoder()

        model = smp.PAN(
                encoder_name="swin_encoder",
                encoder_weights=config['encoder_weight'],
                encoder_output_stride=32,
                in_channels=3,
                classes=config['n_classes']
            )
	
        return model