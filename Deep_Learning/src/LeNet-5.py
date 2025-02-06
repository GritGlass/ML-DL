import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.sub2=nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.sub4=nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv5=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=4)
        self.fc6=nn.Linear(120,84)
        self.fc7=nn.Linear(84,10)
        self.tanh=nn.Tanh()

    def forward(self,x):
        x=self.conv1(x)
        x=self.tanh(x)
        x=self.sub2(x)


        x=self.conv3(x)
        x=self.tanh(x)
        x=self.sub4(x)


        x=self.conv5(x)
        x=self.tanh(x)

        x = torch.flatten(x, start_dim=1)

        x=self.fc6(x)
        x=self.tanh(x)
        x=self.fc7(x)

        return x

if __name__=='__main__':
    transform=transforms.ToTensor()
    train=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    test=datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    train_dataloader=DataLoader(train,
               batch_size=4,
               shuffle=True,)
    test_dataloader=DataLoader(test,
               batch_size=4,
               shuffle=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=LeNet5().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    loss=torch.nn.CrossEntropyLoss()

    loss_score=0
    best_loss = 100
    for epoch in range(200):
        for i,(data,label) in enumerate(train_dataloader):
            data=data.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            output=model(data)
            loss_val=loss(output,label)
            loss_val.backward()
            optimizer.step()

            if i%1000==0:
                print('Epoch:',epoch,'Step:',i,'Loss:',loss_val.item())

        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # 평가에서는 그래디언트 계산을 하지 않음
            for data, labels in test_dataloader:
                data=data.to(device)
                labels=labels.to(device)
                outputs = model(data)
                loss_test = loss(outputs, labels)
                eval_loss += loss_test.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy:',correct/len(test))
            test_loss=eval_loss/len(test_dataloader)
            print('Test Loss:',test_loss)
            if best_loss>test_loss:
              best_loss=test_loss
              torch.save(model.state_dict(), './model_best.pth')
