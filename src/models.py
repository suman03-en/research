import torch.nn as nn
import timm

def get_model(model_name, num_classes=27, pretrained=True):
    """
    Factory function to instantiate models from timm.
    Supported models:
        - resnet50
        - mobilenetv2_100
        - vit_tiny_patch16_224
        - edgenext_small
        - faster_vit_0_224
        - mobilenetv4_conv_small
    """
    # Some models might need slight name adjustments in timm, but these should be standard
    print(f"Initializing {model_name}...")
    
    # Check if model exists in timm
    if model_name not in timm.list_models():
        # Fallback to closest match if exact name not found
        matches = timm.list_models(f"*{model_name}*")
        if not matches:
            # Let's handle specific edge cases in timm names
            if 'edgenext_small' in model_name:
                model_name = 'edgenext_small'
            elif 'faster_vit_0_224' in model_name:
                model_name = 'faster_vit_0_224'
            elif 'mobilenetv4_conv_small' in model_name:
                model_name = 'mobilenetv4_conv_small.e2400_r224_in1k'
            elif 'mobilenetv2_100' in model_name:
                model_name = 'mobilenetv2_100'
            else:
                raise ValueError(f"Model {model_name} not found in timm. Close matches: {matches}")
        else:
            print(f"Warning: Exact name '{model_name}' not found, using closest match: {matches[0]}")
            model_name = matches[0]

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

if __name__ == "__main__":
    # Test initialization
    models = ['resnet50', 'mobilenetv2_100', 'vit_tiny_patch16_224', 
              'edgenext_small', 'faster_vit_0_224', 'mobilenetv4_conv_small.e2400_r224_in1k']
    for m in models:
        try:
            model = get_model(m, num_classes=27)
            print(f"Successfully initialized {m}")
        except Exception as e:
            print(f"Failed to initialize {m}: {e}")
