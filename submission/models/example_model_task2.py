import torch
import torch.nn as nn

class SimpleModel2(nn.Module):
    def __init__(self):
        super(SimpleModel2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.padding = torch.nn.AdaptiveAvgPool2d(7)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 3)  

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.padding(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel2()
torch.save(model.state_dict(), 'models/model_task2.pth')





# Save the model

