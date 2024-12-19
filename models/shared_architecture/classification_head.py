import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, model_dimension, num_labels):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Linear(model_dimension, num_labels)

    def forward(self, x):
        return self.classifier(x)

