import os
import torch
import neptune
import torch.nn.functional as F
import numpy as np

from transformers import BertForSequenceClassification

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    return total_norm

def train_model(model: BertForSequenceClassification,
                optimizer,
                train_dl,
                val_dl,
                num_epoch,
                criterion,
                preload=False,
                weights_name=None,
                neptune_run_id=None
    ):
    
    if preload and weights_name is not None:
        model.load_state_dict(torch.load(weights_name))
    
    run = neptune.init_run(
    project="bng215/Model-Collapse",
    with_id=neptune_run_id,
    api_token=os.environ.get('NEPTUNE_API_TOKEN'),
    )
    
    for epoch in range(num_epoch):
        model.train()
        with tqdm(train_dl, unit='batch') as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                tokens, labels = batch
                
                tokens = tokens.cuda().squeeze(1)
                labels = labels.cuda().to(torch.long)
                
                attention_mask = (tokens != 0).to(torch.uint8).squeeze(1)
                
                logits = model(tokens, attn_mask=attention_mask)
                
                loss = criterion(logits, labels)
                
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                
                run["Loss (Train)"].append(loss.item())
                run["Gradient norm"].append(get_gradient_norm(model))
        
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_acc = []
            test_f1 = []
            
            for batch in val_dl:
                tokens, labels = batch
                    
                tokens = tokens.cuda().squeeze(1)
                labels = labels.cuda().to(torch.long)
                
                attention_mask = (tokens != 0).to(torch.uint8).squeeze(1)
                    
                logits = model(tokens, attn_mask=attention_mask)
                
                loss = criterion(logits, labels)
                
                pred = F.softmax(logits, 1).argmax(1).cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                
                test_loss.append(loss.item())
                test_acc.append(accuracy_score(labels, pred))
                test_f1.append(f1_score(labels, pred, average='macro'))
                
            run["Loss (Test)"].append(np.mean(test_loss))
            run["Accuracy"].append(np.mean(test_acc))
            run["F1-Score"].append(np.mean(test_f1))
            
        torch.save(model.state_dict(), f"weights/bert_ep_{epoch}.pth")
        
    run.stop()          
            
            