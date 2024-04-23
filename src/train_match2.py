import torch
from tqdm.auto import tqdm
from torch import autocast
import wandb
import datetime
from src.utils import masked_bce


def match2_vector_eval_step(model, batch, TF_inds, loss):
    inputs, labels = batch['input_ids'].to("cuda"), batch['labels'].to("cuda")
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = loss(outputs, labels) #Masks loss over values where final label dimension is 1

            prediction_seq_locs = torch.where(labels[:,:,TF_inds[0]:TF_inds[1]+1] == 1) #get sequence locations where the match2 label is non-zero
            m2_labels = torch.argmax(labels[prediction_seq_locs[0],prediction_seq_locs[1],TF_inds[0]:TF_inds[1]+1], dim=-1)
            m2_preds = torch.argmax(outputs[prediction_seq_locs[0],prediction_seq_locs[1],TF_inds[0]:TF_inds[1]+1], dim=-1)
            accuracy = torch.mean((m2_labels == m2_preds).float())

    return {'log_loss':loss,'accuracy':accuracy}

def match2_vector_eval_loop(model, eval_loader, TF_inds, loss):
    model.eval()
    total_log_loss, total_accuracy = 0, 0
    total_steps = 0
    for batch in tqdm(eval_loader, desc="Evaluating"):
        batch_eval_dict = match2_vector_eval_step(model, batch, TF_inds, loss)
        total_log_loss += batch_eval_dict['log_loss']
        total_accuracy += batch_eval_dict['accuracy']
        total_steps += 1
    avg_log_loss = total_log_loss / total_steps
    avg_accuracy = total_accuracy / total_steps
    return {'log_loss':avg_log_loss, 'accuracy':avg_accuracy}

def vector_train_loop(tf_inds, weight_labels, epochs, mpt, accumulation_factor, optim, lr_decay_on, no_wdb, checkpoint, max_grad_norm, base_path, initial_word, model, 
                      epoch_steps, eval_steps, checkpoint_steps, train_data_loader, eval_data_loader, optimizer, decay_scheduler, scaler,):
    loss_func = lambda outputs, labels: masked_bce(outputs, labels, reweight_final=weight_labels, tf_coords=tf_inds)
    for e in range(epochs):
        print(f'###### NEW EPOCH {e}')
        for b,batch in enumerate(tqdm(train_data_loader, desc=f"Training")):                
            model.train()
            inputs = batch['input_ids'].to("cuda")
            labels = batch['labels'].to("cuda")
            if mpt:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    if b==0: print(f"memory used after FP GB {torch.cuda.max_memory_allocated('cuda') / (1e9):.3f}")
                    loss = loss_func(outputs, labels,) #Masks loss over values where final label dimension is 1
                scaler.scale(loss).backward()
                if b==0: print(f"memory used after BWP GB {torch.cuda.max_memory_allocated('cuda') / (1e9):.3f}")
                if (b+1) % accumulation_factor == 0:
                    if optim=="adam":
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if lr_decay_on: decay_scheduler.step()
            else:
                outputs = model(inputs_embeds=inputs, output_hidden_states=True)
                loss = loss_func(outputs, labels)
                loss.backward()
                if (b+1) % accumulation_factor == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    if lr_decay_on: decay_scheduler.step()
            optimizer.zero_grad()

        
            if (b+1) % eval_steps ==0:
                eval_metrics = match2_vector_eval_loop(model, eval_data_loader, tf_inds, loss_func)

                print(f"Train Loss is {loss:.3f}")
                for k,v in eval_metrics.items():
                    print('eval metrics')
                    print(f"{k} is {v:.3f}")
                if not no_wdb:
                    eval_metrics['step'] = b+e*len(train_data_loader)
                    eval_metrics['train_loss'] = loss
                    wandb.log(eval_metrics)
                print(f"LR is {optimizer.param_groups[0]['lr']}")

        if checkpoint: #end of epoch
            today = datetime.datetime.now().strftime("%Y-%m-%d-%H")
            model.save_pretrained(base_path+f"output_dir/{today}-{initial_word}-checkpoint-epoch-{e}-endofepoch")
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model.save_pretrained(base_path+f"output_dir/{today}-{initial_word}-checkpoint-final")


