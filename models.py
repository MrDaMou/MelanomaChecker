# The class containing the model
import torch
import torch.nn as nn
from PIL import Image
import torchvision
from torchvision import transforms

class HAM10000_Densenet:
    def __init__(self):
        self.model = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 7)

        self.model.load_state_dict(torch.load("HAM10000-Densenet.pth", map_location=torch.device('cpu')))

        self.model.eval()
    
    def infer(self, image_path, logger):
        input_image = Image.open(image_path)

        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.76303625, 0.5456404, 0.5700425], std=[0.140928, 0.15261285, 0.1699707]),
        ])
        input_tensor = preprocess(input_image)

        # create a mini-batch as expected by the model
        #input_batch = torch.stack(input_tensor)
        input_batch = input_tensor.unsqueeze(0) 

        # move the input and model to GPU for speed if available
        #if torch.cuda.is_available():
        #    input_batch = input_batch.to('cuda')
        #    self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        output = torch.nn.functional.softmax(output[0], dim=0)

        return ("Melanoma", output[6])


