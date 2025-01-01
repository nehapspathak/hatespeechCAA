# %%
# !pip install sentence_transformers

# %%
# !pip install --upgrade scikit-learn

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AdamW, get_linear_schedule_with_warmup
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('always')

# %%
base_path="./data/input/"
model_dir='mbert'
# %%
# # curating dataset
# caa_anti = pd.read_excel(f'{base_path}caa_anti.xlsx', sheet_name="NK_CAA_Anti")
# caa_pro = pd.read_excel(f'{base_path}caa_pro.xlsx', sheet_name="NK")
# # df.head()

# %%
# caa_anti['dataset'] = 0
# caa_pro['dataset'] = 0

# %%
# df = pd.concat([caa_pro, caa_anti])
# df.head()

# %%
# train, test, train_y, test_y = train_test_split(df, df.Stance, test_size=0.3, random_state=42)
# train, val, train_y, val_y = train_test_split(train, train_y, random_state=42, test_size=0.2)

# %%
# train.to_csv(f'{base_path}train.csv', index = False)
# val.to_csv(f'{base_path}val.csv', index=False)
# test.to_csv(f'{base_path}test.csv', index=False)

# %%
#!/usr/bin/env python
# coding: utf-8

def clean_text(row):
    text = []
    [text.extend(i.strip().split('।')) for i in row]
    text = [i.strip() for i in text]
    text = list(filter(None, text))
    return text

def clean_dataset():
    train = pd.read_csv(f'{base_path}train.csv')
    val = pd.read_csv(f'{base_path}val.csv')
    test = pd.read_csv(f'{base_path}test.csv')

    train.text = train.text.map(lambda x : clean_text(x))
    val.text = val.text.map(lambda x : clean_text(x))
    test.text = test.text.map(lambda x : clean_text(x))
    return train, val, test


# %%

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.stance = self.df.Stance.map({'Anti': 0, 'Pro': 1})
        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = {}
        embeddings = self.sentence_model.encode(
            self.df.text.iloc[idx]
        )

        sample['embeddings'] = torch.from_numpy(embeddings)
        sample['stance'] = torch.Tensor([self.stance.iloc[idx]])
        sample['dataset'] = torch.Tensor([self.df.dataset.iloc[idx]])
        sample['hate'] = torch.Tensor([self.df.Hate.iloc[idx]])

        return sample 


# %%

def custom_collate(batch):

    dataset, stance, embs, hate = [], [], [], []
    for item in batch:
        dataset.append(item['dataset'])
        stance.append(item['stance'])
        hate.append(item['hate'])
        embs.append(item['embeddings'])

    dataset = pad_sequence(dataset, batch_first=True)
    embs = pad_sequence(embs, batch_first=True)
    stance = pad_sequence(stance, batch_first=True)
    hate = pad_sequence(hate, batch_first=True)
    return embs, dataset.long(), stance.long(), hate.long()



# %%

class MultiTaskModel(nn.Module):
    def __init__(
        self,
        nhead=1,
        nlayers=1,
        # use_cls=True,
        d_model=768
    ):
        super(MultiTaskModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead,
                batch_first=True
            ), 
            nlayers, 
            norm=None
        )
        self.dataset_classifier = nn.Linear(d_model, 2)
        self.stance_classifier = nn.Linear(d_model, 2)
        self.hate_classifier = nn.Linear(d_model, 2)
        
        ## Use [cls] token or pooling output for bail prediction
        # self.use_cls = use_cls
        self.d_model = d_model

  
    def forward(self, x):
        ## x: (batch_size, padded_length, 768)
        batch_size = x.size()[0]
    
        x = self.encoder_layer(x)
        x = torch.sum(x, dim=1) 

        stance_logits = self.stance_classifier(x)  ## (batch_size, 2) 
        dataset_logits = self.dataset_classifier(x) ## (batch_size, 2) 
        hate_logits = self.hate_classifier(x) ## (batch_size, 2) 
        
        return dataset_logits, stance_logits, hate_logits


# %%

def train_func_epoch(epoch, model, dataloader, device, optimizer, scheduler, gradient_accumulation_steps=1):

    # Put the model into the training mode
    model.train()

    total_loss = 0

    with tqdm(dataloader, unit="batch", total=len(dataloader)) as single_epoch:

        for step, batch in enumerate(single_epoch):

            single_epoch.set_description(f"Training- Epoch {epoch}")

            embeddings, dataset_label, stance_label, hate_label = batch 

            ## Load the inputs to the device
            embeddings = embeddings.to(device)
            dataset_label = dataset_label.to(device)
            stance_label = stance_label.to(device)
            hate_label = hate_label.to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. 
            dataset_logits, stance_logits, hate_logits = model(embeddings) ## mask

            # # Flaten the logits and label-  logits: (b_s, number_of_sentences, 2) - > (b_s * number_of_sentences, 2)
            # saliency_logits = saliency_logits.contiguous().view(-1, saliency_logits.size(-1))
            # saliency_label = saliency_label.contiguous().view(-1)

            ## Calculate the final loss
            loss_stance = F.cross_entropy(stance_logits, stance_label.squeeze(1))
            loss_dataset = F.cross_entropy(dataset_logits, dataset_label.squeeze(1))
            loss_hate = F.cross_entropy(hate_logits, hate_label.squeeze(1))

            loss = loss_stance + loss_dataset + loss_hate

            total_loss += loss.item()
            
            loss.backward()

            if step % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                ## Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                ## torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

            single_epoch.set_postfix(train_loss=total_loss/(step+1))

    return total_loss / len(dataloader)



# %%

# In[5]:

def eval_func_epoch(model, dataloader, device, epoch):

    # Put the model into the training mode
    model.eval()

    total_loss = 0

    targets_stance = []
    targets_dataset = []
    targets_hate = []

    predictions_dataset = []
    predictions_stance = []
    predictions_hate = []

    with tqdm(dataloader, unit="batch", total=len(dataloader)) as single_epoch:

        for step, batch in enumerate(single_epoch):

            single_epoch.set_description(f"Evaluating- Epoch {epoch}")

            embeddings, dataset_label, stance_label, hate_label = batch 

            ## Load the inputs to the device
            embeddings = embeddings.to(device)
            dataset_label = dataset_label.to(device)
            stance_label = stance_label.to(device)
            hate_label = hate_label.to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. 
            with torch.no_grad():
                dataset_logits, stance_logits, hate_logits = model(embeddings) ## mask

            ## Flaten the logits and label-  logits: (b_s, number_of_sentences, 2) - > (b_s * number_of_sentences, 2)
            # saliency_logits = saliency_logits.contiguous().view(-1, saliency_logits.size(-1))
            # stance_label = stance_label.contiguous().view(-1)

            ## Calculate the final loss
            loss_dataset = F.cross_entropy(dataset_logits, dataset_label.squeeze(1))
            loss_stance = F.cross_entropy(stance_logits, stance_label.squeeze(1))
            loss_hate = F.cross_entropy(hate_logits, hate_label.squeeze(1))

            loss = loss_stance + loss_dataset + loss_hate
            total_loss += loss.item()
            
            single_epoch.set_postfix(train_loss=total_loss/(step+1))

            # Finding predictions 
            pred_dataset = torch.argmax(dataset_logits, dim=1).flatten().cpu().numpy()
            pred_stance = torch.argmax(stance_logits, dim=1).flatten().cpu().numpy()
            pred_hate = torch.argmax(hate_logits, dim=1).flatten().cpu().numpy()

            predictions_dataset.append(pred_dataset)
            predictions_stance.append(pred_stance)
            predictions_hate.append(pred_hate)
            
            targets_dataset.append(dataset_label.squeeze(1).cpu().numpy())
            targets_stance.append(stance_label.squeeze(1).cpu().numpy())
            targets_hate.append(hate_label.squeeze(1).cpu().numpy())

            #if step == 10:
               #break

    targets_dataset = np.concatenate(targets_dataset, axis=0)
    targets_stance = np.concatenate(targets_stance, axis=0)
    targets_hate = np.concatenate(targets_hate, axis=0)
    
    predictions_dataset = np.concatenate(predictions_dataset, axis=0)
    predictions_stance = np.concatenate(predictions_stance, axis=0)
    predictions_hate = np.concatenate(predictions_hate, axis=0)
    
    epoch_validation_loss = total_loss/len(dataloader)

    report_dataset = classification_report(targets_dataset, predictions_dataset, output_dict=True, labels=[0,1])
    report_stance = classification_report(targets_stance, predictions_stance, output_dict=True, labels=[0,1])
    report_hate = classification_report(targets_hate, predictions_hate, output_dict=True, labels=[0,1])

    cm_d = confusion_matrix(targets_dataset, predictions_dataset, labels=[0,1])
    cm_s = confusion_matrix(targets_stance, predictions_stance, labels=[0,1])
    cm_h = confusion_matrix(targets_hate, predictions_hate, labels=[0,1])
    
    
    tn_d, fp_d, fn_d, tp_d = cm_d.ravel()
    tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
    tn_h, fp_h, fn_h, tp_h = cm_h.ravel()
    
    if epoch == "TESTING":
        disp = ConfusionMatrixDisplay(cm_d, display_labels=[0,1])
        disp.plot()
        plt.savefig(f"./data/output/{model_dir}/dataset.png", dpi=300)
        disp = ConfusionMatrixDisplay(cm_s, display_labels=[0,1])
        disp.plot()
        plt.savefig(f"./data/output/{model_dir}/stance.png", dpi=300)
        disp = ConfusionMatrixDisplay(cm_h, display_labels=[0,1])
        disp.plot()
        plt.savefig(f"./data/output/{model_dir}/hate.png", dpi=300)
    
    return epoch_validation_loss, report_dataset, tn_d, fp_d, fn_d, tp_d, report_stance, tn_s, fp_s, fn_s, tp_s, report_hate, tn_h, fp_h, fn_h, tp_h




# %%
# In[6]:


batch_size = 16
gradient_accumulation_steps = 1
epochs = 15
num_warmup_steps = 0
save_model = True 
model_path = f"./data/output/{model_dir}/model_epochs_{epochs}.pt" ## Change after every experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Reading CSV files.")
train, val, test = clean_dataset()

train_dataset = Dataset(train)
val_dataset = Dataset(val)
print("Trainng and Validation datasets ready.")

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate
)

cls_bail_embed = torch.ones(1, 1, 768).to(device) ## embedding size = 768
d_model=768 ## embedding size = 768



# %%
model = MultiTaskModel(d_model=768)

model.to(device)
print("Model ready.")
## Load weights
# loaded_state_dict = torch.load(model_path,  map_location=device)
# model.load_state_dict(loaded_state_dict)

optimizer = AdamW(model.parameters(), lr=5*1e-5)

num_train_steps = math.ceil(len(train_dataloader)/gradient_accumulation_steps)  * epochs 

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
)

best_loss = np.inf
best_epoch = 0


# %%

for epoch in range(epochs):
    print(f"\n---------------------- Epoch: {epoch+1} ---------------------------------- \n")
    ## Training Loop
    train_loss = train_func_epoch(epoch+1, model, train_dataloader, device, optimizer, scheduler)

    ## Validation loop
    (
        val_loss, report_d, tn_d, fp_d, fn_d, tp_d, 
        report_s, tn_s, fp_s, fn_s, tp_s,
        report_h, tn_h, fp_h, fn_h, tp_h
    ) = eval_func_epoch(model, val_dataloader, device, epoch+1)

    print(f"\nEpoch: {epoch+1} | Training loss: {train_loss} | Validation Loss: {val_loss}")
    print()
    print(report_d)
    print()
    print(f"Dataset ==> TP: {tp_d} | FP: {fp_d} | TN: {tn_d}, FN: {fn_d} ")
    print()
    print(report_s)
    print()
    print(f"Stance ==> TP: {tp_s} | FP: {fp_s} | TN: {tn_s}, FN: {fn_s} ")
    print()
    print(report_h)
    print()
    print(f"Hate ==> TP: {tp_h} | FP: {fp_h} | TN: {tn_h}, FN: {fn_h} ")
    print(f"\n----------------------------------------------------------------------------")

    ## Save the model 
    if (val_loss < best_loss) and (save_model == True):
        torch.save(model.state_dict(), model_path)
        best_loss = val_loss
        best_epoch = epoch+1


# %%
# best_epoch = 8

# %%
# model = MultiTaskModel(d_model=768)

# model.to(device)


# %%

# # In[5]:

# # def eval_func_epoch(model, dataloader, device, epoch):

# epoch="TESTING"
# dataloader = test_dataloader
# # Put the model into the training mode
# model.eval()

# total_loss = 0

# targets_stance = []
# targets_dataset = []
# targets_hate = []

# predictions_dataset = []
# predictions_stance = []
# predictions_hate = []

# with tqdm(dataloader, unit="batch", total=len(dataloader)) as single_epoch:

#     for step, batch in enumerate(single_epoch):

#         single_epoch.set_description(f"Evaluating- Epoch {epoch}")

#         embeddings, dataset_label, stance_label, hate_label = batch 

#         ## Load the inputs to the device
#         embeddings = embeddings.to(device)
#         dataset_label = dataset_label.to(device)
#         stance_label = stance_label.to(device)
#         hate_label = hate_label.to(device)

#         # Zero out any previously calculated gradients
#         model.zero_grad()

#         # Perform a forward pass. 
#         with torch.no_grad():
#             dataset_logits, stance_logits, hate_logits = model(embeddings) ## mask

#         ## Flaten the logits and label-  logits: (b_s, number_of_sentences, 2) - > (b_s * number_of_sentences, 2)
#         # saliency_logits = saliency_logits.contiguous().view(-1, saliency_logits.size(-1))
#         # stance_label = stance_label.contiguous().view(-1)

#         ## Calculate the final loss
#         loss_dataset = F.cross_entropy(dataset_logits, dataset_label.squeeze(1))
#         loss_stance = F.cross_entropy(stance_logits, stance_label.squeeze(1))
#         loss_hate = F.cross_entropy(hate_logits, hate_label.squeeze(1))

#         loss = loss_stance + loss_dataset + loss_hate
#         total_loss += loss.item()

#         single_epoch.set_postfix(train_loss=total_loss/(step+1))

#         # Finding predictions 
#         pred_dataset = torch.argmax(dataset_logits, dim=1).flatten().cpu().numpy()
#         pred_stance = torch.argmax(stance_logits, dim=1).flatten().cpu().numpy()
#         pred_hate = torch.argmax(hate_logits, dim=1).flatten().cpu().numpy()

#         predictions_dataset.append(pred_dataset)
#         predictions_stance.append(pred_stance)
#         predictions_hate.append(pred_hate)

#         targets_dataset.append(dataset_label.squeeze(1).cpu().numpy())
#         targets_stance.append(stance_label.squeeze(1).cpu().numpy())
#         targets_hate.append(hate_label.squeeze(1).cpu().numpy())

#         #if step == 10:
#            #break

# targets_dataset = np.concatenate(targets_dataset, axis=0)
# targets_stance = np.concatenate(targets_stance, axis=0)
# targets_hate = np.concatenate(targets_hate, axis=0)

# predictions_dataset = np.concatenate(predictions_dataset, axis=0)
# predictions_stance = np.concatenate(predictions_stance, axis=0)
# predictions_hate = np.concatenate(predictions_hate, axis=0)

# epoch_validation_loss = total_loss/len(dataloader)

# report_dataset = classification_report(targets_dataset, predictions_dataset, output_dict=True, labels=[0,1])
# report_stance = classification_report(targets_stance, predictions_stance, output_dict=True, labels=[0,1])
# report_hate = classification_report(targets_hate, predictions_hate, output_dict=True, labels=[0,1])

# cm_d = confusion_matrix(targets_dataset, predictions_dataset, labels=[0,1])
# cm_s = confusion_matrix(targets_stance, predictions_stance, labels=[0,1])
# cm_h = confusion_matrix(targets_hate, predictions_hate, labels=[0,1])


# tn_d, fp_d, fn_d, tp_d = cm_d.ravel()
# tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
# tn_h, fp_h, fn_h, tp_h = cm_h.ravel()

# # if epoch == "TESTING":
# #     disp = ConfusionMatrixDisplay(cm_d, display_labels=[0,1])
# #     disp.figure_.savefig("./dataset.png", dpi=300)
# #     disp = ConfusionMatrixDisplay(cm_s, display_labels=[0,1])
# #     disp.figure_.savefig("./stance.png", dpi=300)
# #     disp = ConfusionMatrixDisplay(cm_h, display_labels=[0,1])
# #     disp.figure_.savefig("./hate.png", dpi=300)

# return epoch_validation_loss, report_dataset, tn_d, fp_d, fn_d, tp_d, report_stance, tn_s, fp_s, fn_s, tp_s, report_hate, tn_h, fp_h, fn_h, tp_h




# %%

# disp = ConfusionMatrixDisplay(cm_d, display_labels=[0,1])
# disp.plot()
# plt.savefig("./dataset.png", dpi=300)
# disp = ConfusionMatrixDisplay(cm_s, display_labels=[0,1])
# disp.plto()
# plt.savefig("./stance.png", dpi=300)
# disp = ConfusionMatrixDisplay(cm_h, display_labels=[0,1])
# disp.plot()
# plt.savefig("./hate.png", dpi=300)


# %%
## After training get the score on test set 
test_dataset = Dataset(test)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate
)

## Load weights
loaded_state_dict = torch.load(model_path,  map_location=device)
model.load_state_dict(loaded_state_dict)

print(f"\n---------------------- Testing best model (at epoch: {best_epoch} )---------------------------------- \n")
# test_loss,report, tn, fp, fn, tp = eval_func_epoch(model, test_dataloader, device, "TESTING")
(
    test_loss, report_d, tn_d, fp_d, fn_d, tp_d, 
    report_s, tn_s, fp_s, fn_s, tp_s,
    report_h, tn_h, fp_h, fn_h, tp_h
) = eval_func_epoch(model, test_dataloader, device, "TESTING")

print(f"\nTest loss: {test_loss}")
print()
print(report_d)
print()
print(f"Dataset ==> TP: {tp_d} | FP: {fp_d} | TN: {tn_d}, FN: {fn_d} ")
print()
print(report_s)
print()
print(f"Stance ==> TP: {tp_s} | FP: {fp_s} | TN: {tn_s}, FN: {fn_s} ")
print()
print(report_h)
print()
print(f"Hate ==> TP: {tp_h} | FP: {fp_h} | TN: {tn_h}, FN: {fn_h} ")
print(f"\n----------------------------------------------------------------------------") 



# %%

# In[7]:
import json
with open(f"./data/output/{model_dir}/dataset.json","w") as f:
    json.dump(report_d ,f,indent=4) 

with open(f"./data/output/{model_dir}/stance.json","w") as f:
    json.dump(report_s ,f,indent=4) 

with open(f"./data/output/{model_dir}/hate.json","w") as f:
    json.dump(report_h ,f,indent=4) 

# %%



