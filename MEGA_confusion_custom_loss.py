import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, MegaForSequenceClassification, get_scheduler
import torchinfo
from tqdm.auto import tqdm
import evaluate
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim import AdamW
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tokenize_func(examples):
    return tokenizer(
        examples["tweet"], padding="max_length", truncation=True
        )

def train_epoch(model, train_dataloader, optimizer, criterion, lam, lr_scheduler):
    progress_bar = tqdm(range(len(train_dataloader)))
    metric = evaluate.load("accuracy")
    model.train()
    model.to(device)
    #criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        input = batch["input_ids"]
        mask = batch["attention_mask"]
        y = batch["class"]

        outputs = model(input, attention_mask=mask)
        with torch.no_grad():
            _, preds = torch.max(outputs.logits, 1)
        #loss = criterion(outputs.logits, y)
        loss = criterion(outputs.logits, y, lam)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        metric.add_batch(predictions=preds, references=y)

        epoch_loss += loss.item()
        progress_bar.update(1)
    lr_scheduler.step()
    epoch_loss /= len(train_dataloader)
    accuracy = metric.compute()['accuracy']

    return epoch_loss, accuracy

def eval(model, test_dataloader, criterion, lam, seed, epoch):
    progress_bar = tqdm(range(len(test_dataloader)))
    metric = evaluate.load("accuracy")
    model.eval()
    confusion_metric = evaluate.load("confusion_matrix")

    #criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = 0
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        input = batch["input_ids"]
        mask = batch["attention_mask"]
        y = batch["class"]
        
        with torch.no_grad():
            outputs = model(input, attention_mask=mask)

        logits = outputs.logits
        #loss = criterion(outputs.logits, y)
        loss = criterion(outputs.logits, y, lam)

        predictions = torch.argmax(logits, dim=-1)

        metric.add_batch(predictions=predictions, references=y)
        confusion_metric.add_batch(predictions=predictions, references=y)


        epoch_loss += loss.item()
        progress_bar.update(1)

    epoch_loss /= len(test_dataloader)
    accuracy = metric.compute()['accuracy']
    confusion_matrix = confusion_metric.compute()["confusion_matrix"]


    confusion_matrix = np.array(confusion_matrix)
    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_normalized, display_labels=('hate','offensive','neither'))
    disp.plot(cmap=plt.cm.binary)

    plt.savefig(f"matrix_{seed}_{lam}_{epoch}.png", dpi=300, bbox_inches='tight')
    
    return epoch_loss, accuracy

def train(model,
          train_dataloader,
          test_dataloader,
          criterion,
          optimizer,
          lr_scheduler,
          num_epochs=2,
          patience=4,
          seed = 1):

    counter = 0
    best_val_acc = 0
    best_epoch = 0
    best_model = None

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        if epoch < 3:
            model.mega.requires_grad_(False)
        else:
            model.mega.requires_grad_(True)
        train_loss, train_acc = train_epoch(
            model, train_dataloader, optimizer, criterion, lam, lr_scheduler)
        val_loss, val_acc = eval(model, test_dataloader, criterion, lam, seed, epoch)

        print(f"Epoch {epoch+1} accuracy: train={train_acc:.3f}, test={val_acc:.3f}")

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # early stopping
        if val_acc <= best_val_acc + 1e-4:
            counter += 1
            if counter >= patience:
                break
        else:
            best_val_acc = val_acc
            counter = 0
            best_epoch = epoch
            best_model = model.state_dict()

    return train_accuracies, val_accuracies, train_losses, val_losses, best_epoch, best_model

def custom_loss(input, target, lam):
    with torch.no_grad():
        _, preds = torch.max(input, 1)
    
        # Mask for the conditions
        mask_pred2 = (preds == 2) & ((target == 0) | (target == 1)) # most critic, we predict nothing and there is hate/offensive speech
        #mask_pred1 = (preds == 1) & (target == 2) # we predict offensiv but there is hate
        #mask_else = ~(mask_pred2 | mask_pred1) # other cases
        mask_else = ~mask_pred2
        
        is_empty_pred2 = torch.all(~mask_pred2).to(device)
        #is_empty_pred1 = torch.all(~mask_pred1).to(device)
        is_empty_else = torch.all(~mask_else).to(device)

    loss = torch.tensor(0.0).to(device)

    if not is_empty_pred2:
        loss_pred2 = lam * torch.sum(torch.nn.CrossEntropyLoss()(input[mask_pred2], target[mask_pred2])).to(device)
        loss += loss_pred2

    #if not is_empty_pred1:
    #    loss_pred1 = lam/2 * torch.sum(torch.nn.CrossEntropyLoss()(input[mask_pred1], target[mask_pred1])).to(device)
    #    loss += loss_pred1

    if not is_empty_else:
        loss_else = torch.sum(torch.nn.CrossEntropyLoss()(input[mask_else], target[mask_else])).to(device)
        loss += loss_else

    return loss



best_train_accs = {}
best_train_losses = {}
best_val_accs = {}
best_val_losses = {}

best_epochs = {}

all_train_accs = {}
all_train_losses = {}
all_val_accs = {}
all_val_losses = {}

N_EPOCHS = 20
tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
criterion = custom_loss

lr = 1e-4
lambdas = [1.5, 2, 5, 10]

N_SEEDS = 5
for lam in lambdas:
    best_train_accs[f"lam_{lam}"] = []
    best_val_accs[f"lam_{lam}"] = []
    best_train_losses[f"lam_{lam}"] = []
    best_val_losses[f"lam_{lam}"] = []
    best_epochs[f"lam_{lam}"] = []

    all_train_accs[f"lam_{lam}"] = []
    all_train_losses[f"lam_{lam}"] = []
    all_val_accs[f"lam_{lam}"] = []
    all_val_losses[f"lam_{lam}"] = []
    for seed in tqdm(range(N_SEEDS)):
        model = MegaForSequenceClassification.from_pretrained("mnaylor/mega-base-wikitext")
        model.classifier.out_proj = torch.nn.Linear(in_features=128, out_features=3, bias=True)
        model.to(device)
        data_check_test = load_dataset(
            "csv", data_files="data/tweets.csv"
        )
        data_check_test = data_check_test['train'].train_test_split(test_size=0.25)
        tokenized_data = data_check_test.map(tokenize_func, batched=True)
        tokenized_data = tokenized_data.remove_columns([
        "idx", "count", "tweet", "hate_speech", "offensive_language", "neither"])
        tokenized_data.set_format("torch")

        train_dataloader = DataLoader(tokenized_data['train'], batch_size=8)
        test_dataloader = DataLoader(tokenized_data['test'], batch_size=8)


        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_scheduler(name="polynomial", optimizer=optimizer, num_warmup_steps=1, num_training_steps=N_EPOCHS*len(train_dataloader))
        train_acc, val_acc, train_losses, val_losses, best_epoch, best_model = train(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=N_EPOCHS, seed = seed)

        torch.save(best_model, f"model_{seed}_{lam}.pth")

        best_train_accs[f"lam_{lam}"].append(train_acc[best_epoch])
        best_val_accs[f"lam_{lam}"].append(val_acc[best_epoch])
        best_train_losses[f"lam_{lam}"].append(train_losses[best_epoch])
        best_val_losses[f"lam_{lam}"].append(val_losses[best_epoch])
        best_epochs[f"lam_{lam}"].append(best_epoch)

        all_train_accs[f"lam_{lam}"].append(train_acc)
        all_train_losses[f"lam_{lam}"].append(train_losses)
        all_val_accs[f"lam_{lam}"].append(val_acc)
        all_val_losses[f"lam_{lam}"].append(val_losses)


save_all = (best_train_accs, best_val_accs, best_train_losses, best_val_losses, best_epochs, all_train_accs, all_train_losses, all_val_accs, all_val_losses)
with open(f"confusion.obj", 'wb') as f:
    pickle.dump(save_all, f)
