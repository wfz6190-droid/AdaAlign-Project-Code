from easydict import EasyDict
import torch
# import clip

from transformers import SiglipModel, AutoProcessor, AutoTokenizer
import pickle




def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'

# def processed_name(name, rm_dot=False):
#   # _ for lvis
#   # / for obj365
#     res = name.replace('_', ' ').replace('/', ' or ').lower()
#     if rm_dot:
#         res = res.rstrip('.')
#     return res

class_name_template = [
    '{article} {}.'
]

single_photo_template = [
    'a photo of {article} {}.'
]
single_sketch_template = [
    'a sketch of {article} {}.'
]

multiple_common_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',


    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a plushie {}.',
    'the plushie {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]

multiple_photo_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',

    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',

    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a bright photo of {article} {}.',
    'a bright photo of the {}.',

    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',

    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
]

multiple_sketch_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a sketch of {article} {} in the scene.',
    'a sketch of the {} in the scene.',
    'a sketch of one {} in the scene.',

    'a sketch of my {}.',
    'a sketch of the {}.',
    'a sketch of one {}.',

    'a good sketch of {article} {}.',
    'a good sketch of the {}.',
    'a bad sketch of {article} {}.',
    'a bad sketch of the {}.',

    'a sketch of a nice {}.',
    'a sketch of the nice {}.',
    'a sketch of a cool {}.',
    'a sketch of the cool {}.',
    'a sketch of a weird {}.',
    'a sketch of the weird {}.',

    'a sketch of a small {}.',
    'a sketch of the small {}.',
    'a sketch of a large {}.',
    'a sketch of the large {}.',

    'a black and white sketch of the {}.',
    'a black and white sketch of {article} {}.',

    'a painting of the {}.',     
    'a painting of a {}.',
    'a close-up sketch of {article} {}.',
    'a close-up sketch of the {}.',
]


def buding_text_embedding(flag, all_class_name, model, tokenizer):
    """buding text embedding
    args:
        flag: photo and sketch have different templates
        categories: all categories of dataset
    """
    categories = [{'name': item} for item in all_class_name]
    if FLAGS.prompt_ensembling:
        if flag == 'photo':
            templates = multiple_photo_templates
        elif flag == 'sketch':
            templates = multiple_sketch_templates
        elif flag == 'naive':
            templates = multiple_common_templates
        else:
            print('unknown type for text_embedding initialization:')
            return
        with torch.no_grad():
            all_text_embeddings = []
            for category in categories:
                # the length of templates
                texts = [
                    template.format(category['name'], article=article(category['name'])) for template in templates
                ]
                if FLAGS.this_is:
                    texts = [
                        'This is ' + text if text.startswith('a') or text.startswith('the') else text for text in texts
                    ]
                texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                text_embeddings = model(**texts)[1]
                # texts = clip.tokenize(texts).to(device=)  # tokenize
                # text_embeddings = clip_model.encode_text(texts)
                if FLAGS.normalization:
                    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                    text_embedding = text_embeddings.mean(dim=0)
                    # text_embedding /= text_embedding.norm()
                else:
                    text_embedding = text_embeddings.mean(dim=0)
                all_text_embeddings.append(text_embedding)
            all_text_embeddings = torch.stack(all_text_embeddings)
        # return all categories text_embeddings
        return all_text_embeddings.cpu().numpy()
    else:
    # single template
        if flag == 'photo':
            templates = single_photo_template
        elif flag == 'sketch':
            templates = single_sketch_template
        elif flag == 'naive': 
            templates = single_photo_template
        elif flag == 'class_name':
            templates = class_name_template
        else:
            print('unknown type for text_embedding initialization:')
            return
        with torch.no_grad():
            all_text_embeddings = []
            for category in categories:
                texts = [
                    template.format(category['name'], article=article(category['name'])) for template in templates
                ]
                # texts = clip.tokenize(texts).to(args.device)  # tokenize
                # text_embeddings = clip_model.encode_text(texts)
                texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                # text_embeddings = model(**texts)[1]
                results = model(**texts)
                text_embeddings = results[1]
                if FLAGS.normalization:
                    text_embeddings /= text_embeddings.norm(p=2, dim=-1, keepdim=True)
                # all_text_embeddings.append(text_embeddings.mean(dim=0))
                all_text_embeddings.append(text_embeddings)
            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

        return all_text_embeddings.cpu().numpy()


# multiple_templates = [
#     'There is {article} {} in the scene.',
#     'There is the {} in the scene.',
#     'a photo of {article} {} in the scene.',
#     'a photo of the {} in the scene.',
#     'a photo of one {} in the scene.',


#     'itap of {article} {}.',
#     'itap of my {}.',  # itap: I took a picture of
#     'itap of the {}.',
#     'a photo of {article} {}.',
#     'a photo of my {}.',
#     'a photo of the {}.',
#     'a photo of one {}.',
#     'a photo of many {}.',

#     'a good photo of {article} {}.',
#     'a good photo of the {}.',
#     'a bad photo of {article} {}.',
#     'a bad photo of the {}.',
#     'a photo of a nice {}.',
#     'a photo of the nice {}.',
#     'a photo of a cool {}.',
#     'a photo of the cool {}.',
#     'a photo of a weird {}.',
#     'a photo of the weird {}.',

#     'a photo of a small {}.',
#     'a photo of the small {}.',
#     'a photo of a large {}.',
#     'a photo of the large {}.',

#     'a photo of a clean {}.',
#     'a photo of the clean {}.',
#     'a photo of a dirty {}.',
#     'a photo of the dirty {}.',

#     'a bright photo of {article} {}.',
#     'a bright photo of the {}.',
#     'a dark photo of {article} {}.',
#     'a dark photo of the {}.',

#     'a photo of a hard to see {}.',
#     'a photo of the hard to see {}.',
#     'a low resolution photo of {article} {}.',
#     'a low resolution photo of the {}.',
#     'a cropped photo of {article} {}.',
#     'a cropped photo of the {}.',
#     'a close-up photo of {article} {}.',
#     'a close-up photo of the {}.',
#     'a jpeg corrupted photo of {article} {}.',
#     'a jpeg corrupted photo of the {}.',
#     'a blurry photo of {article} {}.',
#     'a blurry photo of the {}.',
#     'a pixelated photo of {article} {}.',
#     'a pixelated photo of the {}.',

#     'a black and white photo of the {}.',
#     'a black and white photo of {article} {}.',

#     'a plastic {}.',
#     'the plastic {}.',

#     'a toy {}.',
#     'the toy {}.',
#     'a plushie {}.',
#     'the plushie {}.',
#     'a cartoon {}.',
#     'the cartoon {}.',

#     'an embroidered {}.',
#     'the embroidered {}.',

#     'a painting of the {}.',
#     'a painting of a {}.',
# ]

FLAGS = {
    'prompt_ensembling': False,
    'this_is': False,
    'temperature': 100.0,
    'use_softmax': False,
    'normalization':True,
}
FLAGS = EasyDict(FLAGS)

def main():
    path = "checkpoints/huggingface/google/siglip-base-patch16-224"
    tokenizer = AutoTokenizer.from_pretrained(path)
    siglip_model = SiglipModel.from_pretrained(path)
    model = siglip_model.text_model
    class_labels_file = '/data/sydong/datasets/SBIR/Sketchy/zeroshot1/cname_cid.txt'
    save_path = 'data/Sketchy/zeroshot1/class_embs_siglip-base-p16_naive.pickle'
    with open(class_labels_file) as fp:
        all_class_name = [c.split()[0] for c in fp.readlines()] # 读取训练的类别标签
    text_beddings = buding_text_embedding('naive', all_class_name, model, tokenizer)
    # print(text_beddings.shape)
    # save text embeddings
    with open(save_path, 'wb') as f:
        pickle.dump(text_beddings, f)

main()






