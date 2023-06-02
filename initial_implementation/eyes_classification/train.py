import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import time
from datetime import timedelta
from model import Net
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./logs')      

data_dir = './eyes'

transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

total_size = len(data)
training_size = int(0.6 * len(data))
validation_size = int(0.2 * len(data))
test_size = total_size - training_size - validation_size

training_dataset, validation_dataset, test_dataset = random_split(
    data, 
    [training_size, validation_size, test_size]
)

train_batch_size = 32
test_batch_size = 64
train_dataloader = DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=test_batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# 64*64
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

model = Net().to(device)
summary(model, (3, 256, 256))

# Hyperparameters
epochs = 20
learning_rate = 1e-3
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_function, optimizer, epoch):
    model.train()      # set the model in training mode
    avg_train_loss, correct = 0, 0
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        predictions = model(X)      # forward propagation
        loss = loss_function(predictions, y)        # loss
        avg_train_loss += loss.item()
        optimizer.zero_grad()   # zero the parameter gradients
        loss.backward()         # backpropagation
        optimizer.step()        
        _, predicted = torch.max(predictions.data, 1)  # the class with the highest energy is what we choose as prediction
        correct += (predicted == y).sum().item()
        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_train_loss /= len(dataloader)
    train_accuracy = 100*correct/len(dataloader.dataset)
    statistics('training', train_accuracy, avg_train_loss, epoch)
    return

def evaluate_validation(dataloader, model, loss_function, epoch):
    model.eval()        # set to evaluation model
    avg_validation_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            predictions = model(images)
            avg_validation_loss += loss_function(predictions, labels).item()       # loss
            _, predicted = torch.max(predictions.data, 1)   # the class with the highest energy is what we choose as prediction
            correct += (predicted == labels).sum().item()
    avg_validation_loss /= len(dataloader)
    validation_accuracy = 100*correct/len(dataloader.dataset)
    statistics('validation', validation_accuracy, avg_validation_loss, epoch)
    return

def statistics(dataset, accuracy, loss, epoch):
    writer.add_scalar('Loss/' + dataset, loss, epoch)
    writer.add_scalar('Accuracy/' + dataset, accuracy, epoch)
    print("{},\tLoss: {:.{}f}\t| Accuracy: {:.{}f}".format(dataset.title(), loss, 3, accuracy, 3))
    return

def optimize(epochs, train_dataloader, validation_dataloader, model, loss_function, optimizer):
    start_time = time.time()
    for i in range(epochs):
        print(f"\nEpoch {i+1}\n----------------------------------------------")
        train(train_dataloader, model, loss_function, optimizer, i)
        evaluate_validation(validation_dataloader, model, loss_function, i)
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    return

# training
optimize(epochs, train_dataloader, validation_dataloader, model, loss_function, optimizer)   

print('Finished Training')
torch.save(model.state_dict(), "eye_model.pth")
writer.close()

