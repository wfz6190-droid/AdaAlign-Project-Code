"""
Text utilities for dynamic class embedding generation
"""
import torch
import torch.nn.functional as F


def article(name):
    """Return 'a' or 'an' based on the first letter"""
    return 'an' if name[0].lower() in 'aeiou' else 'a'


# Text prompt templates
SINGLE_PHOTO_TEMPLATE = ['a photo of {article} {}.']
SINGLE_SKETCH_TEMPLATE = ['a sketch of {article} {}.']
CLASS_NAME_TEMPLATE = ['{article} {}.']

MULTIPLE_COMMON_TEMPLATES = [
    'a photo of {article} {}.',
    'a photo of the {}.',
    'a sketch of {article} {}.',
    'a sketch of the {}.',
]


def get_text_templates(prompt_type='naive'):
    """Get text templates based on prompt type"""
    if prompt_type == 'photo':
        return SINGLE_PHOTO_TEMPLATE
    elif prompt_type == 'sketch':
        return SINGLE_SKETCH_TEMPLATE
    elif prompt_type == 'class_name':
        return CLASS_NAME_TEMPLATE
    elif prompt_type == 'naive':
        return SINGLE_PHOTO_TEMPLATE
    elif prompt_type == 'sk-im':
        return MULTIPLE_COMMON_TEMPLATES
    else:
        return SINGLE_PHOTO_TEMPLATE


def generate_text_prompts(class_names, prompt_type='naive'):
    """
    Generate text prompts for all class names
    
    Args:
        class_names: List of class names
        prompt_type: Type of prompt template
    
    Returns:
        List of text prompts
    """
    templates = get_text_templates(prompt_type)
    
    all_prompts = []
    for class_name in class_names:
        for template in templates:
            if '{article}' in template:
                prompt = template.format(class_name, article=article(class_name))
            else:
                prompt = template.format(class_name)
            all_prompts.append(prompt)
    
    return all_prompts


def generate_class_embeddings(class_names, text_encoder, tokenizer, prompt_type='naive', 
                              normalize=True, device='cuda'):
    """
    Generate class embeddings dynamically using text encoder
    
    Args:
        class_names: List of class names
        text_encoder: Text encoder model
        tokenizer: Tokenizer for text encoder
        prompt_type: Type of prompt template
        normalize: Whether to normalize embeddings
        device: Device to run on
    
    Returns:
        Tensor of class embeddings [num_classes, embed_dim]
    """
    try:
        templates = get_text_templates(prompt_type)
    except Exception as e:
        print(f"Warning: Failed to get templates for '{prompt_type}', using default")
        templates = get_text_templates('naive')
    
    all_embeddings = []
    
    # Process each class
    for class_name in class_names:
        class_prompts = []
        for template in templates:
            if '{article}' in template:
                prompt = template.format(class_name, article=article(class_name))
            else:
                prompt = template.format(class_name)
            class_prompts.append(prompt)
        
        # Encode text prompts
        embeddings = text_encoder.encode_text(class_prompts, tokenizer, device=device)
        
        # Normalize if specified
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        
        # Average embeddings for this class (ensemble)
        class_embedding = embeddings.mean(dim=0)
        all_embeddings.append(class_embedding)
    
    # Stack into [num_classes, embed_dim]
    class_embeddings = torch.stack(all_embeddings, dim=0)
    
    return class_embeddings


def update_class_embeddings_inplace(class_embeddings, class_names, text_encoder, tokenizer, 
                                   prompt_type='naive', normalize=True, device='cuda'):
    """
    Update class embeddings in-place during training
    
    Args:
        class_embeddings: Existing embeddings tensor to update
        class_names: List of class names
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        prompt_type: Prompt template type
        normalize: Whether to normalize
        device: Device to run on
    
    Returns:
        Updated embeddings tensor
    """
    new_embeddings = generate_class_embeddings(
        class_names, text_encoder, tokenizer, 
        prompt_type, normalize, device
    )
    
    class_embeddings.data = new_embeddings.data
    return class_embeddings

