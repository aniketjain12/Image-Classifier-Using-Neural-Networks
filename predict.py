import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from train import DenseNet
import os
import argparse
import json
from PIL import Image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Load and process the image
    img = Image.open(image)

    # Resize the image to 224x224 pixels
    img = img.resize((224, 224))

    # Convert to NumPy array and normalize
    np_image = np.array(img) / 255.0

    # Standardize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose the image to the correct format (channels, height, width)
    np_image = np_image.transpose((2, 0, 1))

    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(np_image).float()

    return img_tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model_checkpoint, category_names='cat_to_name.json', topk=5, device=None):
    # Load the checkpoint and extract the model state dict
    checkpoint = torch.load(model_checkpoint)
    model_state_dict = checkpoint['state_dict']

    # Assuming your model class is called DenseNet and you've defined it
    model = DenseNet()
    #model = torchvision.models.densenet121(pretrained=True)
    #print(model.features[0].in_channels)
    model.load_state_dict(model_state_dict)

    # Preprocess the image
    img = process_image(image_path)
    #img = img.unsqueeze(0)  # Add batch dimension
    #img = torch.randn(1, 224, 224, 3)
    #img = torch.randn(1, 3, 224, 224)
    img = torch.FloatTensor(img)

    img.unsqueeze_(0)
    # Set the model to evaluation mode and move to the appropriate device
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #img = torch.from_numpy(img).to(device)
    img = img.to(device)

    # Perform the forward pass
    with torch.no_grad():
        logps = model.forward(img)
 
    # Calculate probabilities and classes
    probabilities = torch.exp(logps)
    #print("probabilities:",probabilities.topk(topk))
    topk_prob, topk_class = probabilities.topk(topk, dim=1)
    #print("topk_prob",topk_prob)
    #print("topk_classe",topk_class)
    
    topk_probs = topk_prob.tolist()[0]
    topk_classes = topk_class.tolist()[0]
    # Convert indices to class labels using class_to_idx
    idx_to_class = {val: key for key, val in checkpoint['class_to_idx'].items()}
    #idx_class={i: k for k, i in checkpoint['class_to_idx'].items()}
    topk_classes = [idx_to_class[i] for i in topk_classes]
    if category_names:
       with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    topk_names = [cat_to_name[i] for i in topk_classes if i is not None]


    return topk_probs, topk_classes, topk_names
if __name__ == "__main__":
    # python predict.py flowers/test/1/image_06743.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help='Enter Image', action='store')
    parser.add_argument("--checkpoint", help='Checkpoint file', action='store', default='checkpoint.pth')
    parser.add_argument("--topk", help='Top values', action='store', default=2, type=int)
    parser.add_argument("--category_name", help='Name of category with json', action='store', default='cat_to_name.json', type=str)
    parser.add_argument("--gpu", help='GPU(cuda)', action='store')
    
    args = parser.parse_args()
    topk_probs, topk_classes, topk_names = predict(args.image_path, model_checkpoint=args.checkpoint, category_names = args.category_name, topk=args.topk, device=args.gpu)
    '''
    print('Class names:   ', topk_names)
    print('Classes:       ', topk_classes)
    print('Probabilities: ', topk_probs)
    '''
    zipped_result = zip(topk_classes, topk_names, topk_probs)
    result_list = list(zipped_result)
    for classes, names, probs in result_list:
        print(f"Name of Flower is: {names}... Probability of that flower is: {probs}")


