import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.0f}%)\n')
    return test_loss, accuracy


epochs = 10
train_losses, validation_losses, train_accuracies, validation_accuracies = [], [], [], []

for epoch in range(1, epochs + 1):
    train(epoch)
    train_loss, train_accuracy = test(train_loader)
    validation_loss, validation_accuracy = test(validation_loader)
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)


torch.save(model.state_dict(), 'mnist_ann-model.pth')


from torchvision import transforms
from PIL import Image


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load('mnist_ann-model.pth'))
model.eval()


def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Convertir en niveaux de gris si nécessaire
        transforms.Resize((28, 28)), # Redimensionner l'image en 28x28
        transforms.ToTensor(), # Convertir l'image en tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Normaliser comme pour les données d'entraînement
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0) # Ajouter une dimension de batch
    return image

# Demander à l'utilisateur de saisir le chemin de l'image
image_path = input("Entrez le chemin de votre image: ")

try:
    # Préparation de l'image
    image = transform_image(image_path).to(device)

    # Faire une prédiction
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True).item()

    # Afficher l'image et la prédiction
    plt.imshow(Image.open(image_path), cmap='gray')
    plt.title(f'Prédiction: {pred}')
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Une erreur est survenue: {e}. Assurez-vous que le chemin de l'image est correct.")
