CLIP_EMBEDDING_DIMS = {
    'ViT-B/32': 512,
    'ViT-B/16': 512,
    'RN50': 1024,
    'RN101': 512,
    'RN50x4': 640,
    'RN50x16': 768,
}

TXT_EMBEDDING_DIMS = {
    'ViT-B/32': 512,
    'ViT-B/16': 512,
    'RN50': 1024,
    'RN101': 512,
    'RN50x4': 640,
    'RN50x16': 768,
    'gpt2': 768,
    'gpt2-medium': 1024,
    'gpt2-large': 1024,
    'gpt2-xl': 1600,
    'bert-base-cased': 768,
    'bert-large-cased': 1024,
    'bert-base-uncased': 768,
    'bert-large-uncased': 1024,
    'roberta-base': 768,
    'roberta-large': 1024,
    'sgpt': 2048,
}

CLIP_IMAGE_RESOLUTION = {
    'ViT-B/32': 224,
    'ViT-B/16': 224,
    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
}

MLFLOW_TRACKING_URI = 'http://127.0.0.1:5001' #5000
MLFLOW_ARTIFACT_STORE = '/home/ssd_storage/experiments/clip_decoder/mlflow/artifact_store' #'/home/hdd_storage/mlflow/artifact_store'

PREDICT_WITH_RANDOM_Z = False
TRUNCATE_LIMIT = 1.0

model_paths = {
    'ir_se50': 'models/psp/encoders/model_ir_se50.pth',
    'stylegan_weights': 'models/stylegan2/stylegan2-ffhq-config-f.pt',
    'psp_checkpoint_path': '/home/administrator/PycharmProjects/clip_sg_decoder/models/psp/psp_ffhq_encode_new.pt',
    'e4e_checkpoint_path': '/home/ssd_storage/experiments/clip_decoder/e4e_ffhq_encode.pt'
}
