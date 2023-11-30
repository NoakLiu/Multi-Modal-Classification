# main.py
import torch
from torchvision import transforms
from data_loader import TextImageDataLoader, TextImageDataset
from tokenizer import Tokenizer
from model import Resnext50, TextEmbedding, BiGRUAttention, ConcatenateEmbedModel
from train import train_epoch, validation, test_pred

# Set file paths and constants
FILENAME_TRAIN = "image2text_train.csv"
FILENAME_TEST = "image2text_test.csv"
PICTURE_PATH = "data/"
BATCH_SIZE = 64
VAL_RATIO = 0.2
N_EPOCHS = 15
LOG_INTERVAL = 50
LEARNING_RATE = 3e-4

def main():
    # Load and preprocess the data
    data_loader = TextImageDataLoader(FILENAME_TRAIN, FILENAME_TEST, PICTURE_PATH)
    df_train, df_test = data_loader.load_data()

    # Tokenization and preprocessing
    tokenizer = Tokenizer()
    x_train, y_train, x_test, _ = tokenizer.tokenize_and_preprocess(df_train, df_test)

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = TextImageDataset(x_train['ImageID'], x_train['Caption_encoded'], y_train, train_transform, PICTURE_PATH)
    test_dataset = TextImageDataset(x_test['ImageID'], x_test['Caption_encoded'], None, test_transform, PICTURE_PATH)

    # Create data loaders
    train_loader, val_loader = data_loader.train_val_loader(train_dataset, VAL_RATIO, BATCH_SIZE)
    test_loader = data_loader.test_loader(test_dataset, BATCH_SIZE)

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = len(set(y_train))
    img_model = Resnext50(num_class).to(device)
    n_hidden = 128
    text_model = BiGRUAttention(emb_table, n_hidden, n_emb=256).to(device)
    n_emb = 256
    model = ConcatenateEmbedModel(img_model, text_model, n_emb, num_class).to(device)

    # Training and evaluation
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    best_f1_score = -float("inf")

    for epoch in range(N_EPOCHS):
        train_epoch(LOG_INTERVAL, model, train_loader, optimizer, epoch, criterion)
        f1_score = validation(model, device, val_loader, criterion, threshold=0.5)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            torch.save(model, "best_model.pt")

    # Prediction on test set
    best_model = torch.load("best_model.pt")
    submission_df = test_pred(best_model, test_loader, threshold=0.5)
    submission_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
