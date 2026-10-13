def get_model(config, num_classes):
    """Factory function to instantiate models dynamically based on the YAML config."""
    model_name = config['model']['name']
    
    if model_name == "roberta_mapping":
        # lazy import, only load when requested
        from models.roberta_mapping import RobertaMappingModel
        backbone = config['model']['pretrained_backbone']
        return RobertaMappingModel(backbone_name=backbone, num_classes=num_classes)
        
    elif model_name == "dual_encoder":
        from models.dual_encoder import DualEncoderModel
        backbone = config['model']['pretrained_backbone']
        # Allow temperature to be configured via yaml, default to 0.05
        temperature = config['model'].get('temperature', 0.05)
        return DualEncoderModel(backbone_name=backbone, temperature=temperature)
        
    elif model_name == "simple_linear":
        #from models.simple_linear import SimpleLinearBaseline
        #return SimpleLinearBaseline(num_classes=num_classes)
        pass
        
    else:
        raise ValueError(f"Unknown model name configured: {model_name}")