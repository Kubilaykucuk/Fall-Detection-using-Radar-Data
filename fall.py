import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns

label_mapping = {
    "Sitting": 0,
    "Sit-to-stand": 1,
    "Walking": 2,
    "Falling": 3
}

plots_dir = './plots'
os.makedirs(plots_dir, exist_ok=True)

# def csv_to_series(file_path, num_frames=24, width=128):
#     # Read CSV file
#     data = pd.read_csv(file_path, header=None).values
    
#     # Normalize data
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     normalized_data = scaler.fit_transform(data)
    
#     # Flatten the data
#     flattened_data = normalized_data.flatten()
    
#     # Perform gradient-based equalization (interpolation)
#     series = np.interp(
#         np.linspace(0, len(flattened_data) - 1, num_frames * width),
#         np.arange(len(flattened_data)),
#         flattened_data
#     )
    
#     # Reshape to num_frames x width
#     reshaped_series = series.reshape(num_frames, width)
    
#     return reshaped_series

# def save_as_png(data, save_path):
#     plt.imsave(save_path, data, cmap='gray')

# # Directory paths
train_dir = './fall_detection/data/train'
test_dir = './fall_detection/data/test'
# csv_dir = './fall_detection/RadarSeries_1'

# # Create directories for train and test data
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # List of CSV files
# csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
# train_files, test_files = train_test_split(csv_files, test_size=0.2, random_state=42)

# # Process each CSV file
# for file_name in train_files:
#     file_path = os.path.join(csv_dir, file_name)
#     series = csv_to_series(file_path)
#     save_as_png(series, os.path.join(train_dir, file_name.replace('.csv', '.png')))

# for file_name in test_files:
#     file_path = os.path.join(csv_dir, file_name)
#     series = csv_to_series(file_path)
#     save_as_png(series, os.path.join(test_dir, file_name.replace('.csv', '.png')))


class FallDetectionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (batch_size, 32, 12, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 6, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, 3, 16)
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),  # Output should be in the range [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Take the output of the last time step
        return output

# Parameters
batch_size = 4
learning_rate = 0.001
num_epochs = 150
num_classes = 4  # Sitting, Sit-to-stand, Walking, Falling
hidden_size = 128  # Size of LSTM hidden layer
input_size = 128  # This should match the feature size after encoding (128 for this example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((24, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1
])

# Dataset and DataLoader for autoencoder
train_dataset = FallDetectionDataset('./fall_detection/data/train', transform=transform)
test_dataset = FallDetectionDataset('./fall_detection/data/test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the ConvAutoencoder
autoencoder = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

autoencoder_losses = []
lstm_losses = []

# Training loop for the autoencoder
for epoch in range(num_epochs):
    autoencoder.train()
    running_loss = 0.0

    for images in train_loader:
        images = images.to(device)

        # Forward pass
        outputs = autoencoder(images)
        loss = criterion(outputs, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    autoencoder_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Save the trained autoencoder
torch.save(autoencoder, './models/conv_autoencoder.pth')

plt.figure(figsize=(10, 5))
plt.plot(autoencoder_losses, label='Autoencoder Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plots_dir, 'autoencoder_loss.png'))
plt.close()

# Feature Extraction
def extract_features(dataloader, model):
    model.eval()
    features = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            encoded = model.encoder(images)  # Get the features from the encoder
            features.append(encoded.view(encoded.size(0), -1))  # Flatten the features
    return torch.cat(features).cpu().numpy()

# Extract features
train_features = extract_features(train_loader, autoencoder)
test_features = extract_features(test_loader, autoencoder)


# Reshape features for LSTM input
train_features = train_features.reshape(-1, 1, input_size)  # (num_samples, seq_length=1, input_size)
test_features = test_features.reshape(-1, 1, input_size)

# Prepare LSTM dataset
class LSTMFeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.labels = np.random.randint(0, num_classes, size=(features.shape[0],))  # Dummy labels for unsupervised

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, torch.tensor(label, dtype=torch.long)  # Return features and dummy labels

# Create LSTM dataset and DataLoader
lstm_dataset = LSTMFeatureDataset(train_features)
lstm_loader = DataLoader(lstm_dataset, batch_size=batch_size, shuffle=True)
lstm_test_dataset = LSTMFeatureDataset(test_features)
lstm_test_loader = DataLoader(lstm_test_dataset, batch_size=batch_size, shuffle=False)

# Train the LSTM classifier
lstm_model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
criterion_lstm = nn.CrossEntropyLoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Training loop for the LSTM classifier
for epoch in range(num_epochs):
    lstm_model.train()
    running_loss = 0.0

    for features, labels in lstm_loader:
        features, labels = features.to(device), labels.to(device).long()  # Ensure labels are long tensors

        # Forward pass
        outputs = lstm_model(features)
        loss = criterion_lstm(outputs, labels)

        # Backward pass and optimization
        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()

        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(lstm_loader.dataset)
    lstm_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], LSTM Loss: {epoch_loss:.4f}')

# Save the trained LSTM model
torch.save(lstm_model, './models/lstm_classifier.pth')

plt.figure(figsize=(10, 5))
plt.plot(lstm_losses, label='LSTM Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LSTM Classifier Training Loss')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plots_dir, 'lstm_loss.png'))
plt.close()

def predict_on_test_images(test_image_dir):
    test_dataset = FallDetectionDataset(test_image_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    autoencoder_results = []
    lstm_model_results = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Get autoencoder predictions
            autoencoder_outputs = autoencoder(images)
            autoencoder_results.extend(autoencoder_outputs.cpu().numpy())  # Save autoencoder predictions

            # Extract features from autoencoder for LSTM input
            encoded_features = autoencoder.encoder(images).view(images.size(0), -1)  # Flatten the features
            encoded_features = encoded_features.reshape(-1, 1, input_size)  # Reshape for LSTM

            # Get LSTM predictions
            lstm_outputs = lstm_model(encoded_features.to(device))
            _, lstm_preds = torch.max(lstm_outputs, 1)
            lstm_model_results.extend(lstm_preds.cpu().numpy())  # Save LSTM predictions

    return np.array(autoencoder_results), np.array(lstm_model_results)

# Predict on test images
autoencoder_results, lstm_model_results = predict_on_test_images('./fall_detection/data/test')

lstm_model_results = lstm_model_results.reshape(20, 48)

from collections import Counter

# Initialize a list to store the final labels
lstm_final_labels = []

# Iterate over each segment of predictions
for segment in lstm_model_results:
    # Count occurrences of each label in the segment
    label_counts = Counter(segment)
    
    # Find the most common label
    most_common_label, _ = label_counts.most_common(1)[0]
    
    # Append the most common label to the final labels list
    lstm_final_labels.append(most_common_label)

lstm_final_labels = np.array(lstm_final_labels)

test_images_path = './fall_detection/data/test'  # Adjust this path if necessary
test_images = []
image_files = os.listdir(test_images_path)[:4]

for img_file in image_files:
    img_path = os.path.join(test_images_path, img_file)
    img = Image.open(img_path).convert('L')  # Open image and convert to grayscale
    img = img.resize((128, 24))  # Resize to 128x24
    img = np.array(img) / 255.0  # Convert to numpy array and normalize to [0, 1]
    test_images.append(img)

# Convert to numpy array
test_images = np.array(test_images)

# Plot the images and their predicted labels
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Adjust the figure size as needed
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(test_images[i], cmap='gray')  # Use gray colormap for grayscale images
    ax.set_title(f'Predicted Label: {lstm_final_labels[i]}')
    ax.axis('off')  # Hide axis

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'results.png'))
plt.show()