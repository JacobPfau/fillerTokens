import torch
from tqdm.auto import tqdm
from torch import autocast
import wandb
import datetime
import numpy as np

from src.utils import reshape_crossent


def vector_train_loop(
    # Training control parameters
    epochs, mpt, accumulation_factor, early_stop, no_wdb, checkpoint,
    
    # Model and data-related parameters
    model, train_data_loader, eval_data_loader, tf_inds, batch_to_type, tf_label_inds,
    
    # Optimization parameters
    optim, lr_decay_on, max_grad_norm, optimizer, decay_scheduler, scaler,
    
    # Miscellaneous and path parameters
    base_path, run_name, eval_steps
):
    stop = False
    tf = not batch_to_type is None
    for e in range(epochs):
        print(f'###### NEW EPOCH {e}')
        for b,batch in enumerate(tqdm(train_data_loader, desc=f"Training")):                
            model.train()
            inputs = batch['input_ids'].to("cuda")
            labels = batch['labels'].to("cuda")
            if mpt:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)['logits']
                    if b==0: print(f"memory used after FP GB {torch.cuda.max_memory_allocated('cuda') / (1e9):.3f}")
                    loss = reshape_crossent(outputs, labels,) #Masks loss over values where label is -1
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
                with autocast(device_type='cuda', dtype=torch.float):
                    outputs = model(inputs,)['logits']
                    loss = reshape_crossent(outputs, labels)
                    loss.backward()
                    if (b+1) % accumulation_factor == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        if lr_decay_on: decay_scheduler.step()
            optimizer.zero_grad()

        
            if (b+1) % eval_steps ==0:
                eval_metrics = match3_vector_eval_loop(model, eval_data_loader, tf_inds, reshape_crossent, tf, batch_to_type=batch_to_type, cot_tf_bounds=tf_label_inds)

                print(f"Train Loss is {loss:.3f}")
                for k,v in eval_metrics.items():
                    if not v is None:
                        print('eval metrics')
                        print(f"{k} is {v:.3f}")
                if not no_wdb:
                    eval_metrics['step'] = b+e*len(train_data_loader)
                    eval_metrics['train_loss'] = loss
                    wandb.log(eval_metrics)
                else:
                    print(eval_metrics)
                print(f"LR is {optimizer.param_groups[0]['lr']}")
                if early_stop and eval_metrics['accuracy'] > 0.995:
                    stop = True
                    print('Early stopped due to high accuracy')
                    break
        if stop:
            break
        if checkpoint: #end of epoch
            today = datetime.datetime.now().strftime("%Y-%m-%d-%H")
            model.save_pretrained(base_path+f"output_dir/{today}-{run_name}-checkpoint-epoch-{e}-endofepoch")
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model.save_pretrained(base_path+f"output_dir/{today}-{run_name}-checkpoint-final")


def match3_vector_eval_loop(model, eval_loader, TF_inds, loss, tf=True, batch_to_type=None, cot_tf_bounds=None):

    model.eval()
    total_log_loss, total_accuracy, total_final_loss  = 0, 0, 0
    if batch_to_type:
        total_cot_pos_acc, positional_3sum_acc, cot_sums_acc = 0, 0, 0
    total_steps = 0
    ACC_AGGREGATE = {
        'dot': [],
        'cot': [],
    }

    for batch in tqdm(eval_loader, desc="Evaluating"):
        batch_eval_dict = match3_vector_eval_step(model, batch, TF_inds, loss, tf, batch_to_type, cot_tf_bounds)
        total_log_loss += batch_eval_dict.pop('log_loss')
        total_accuracy += batch_eval_dict.pop('accuracy')
        total_final_loss += batch_eval_dict.pop('final_loss')
        if batch_to_type:
            total_cot_pos_acc += batch_eval_dict.pop('cot_pos_acc')
            positional_3sum_acc += batch_eval_dict.pop('positional_3sum_acc')
            cot_sums_acc += batch_eval_dict.pop('cot_sums_acc')
            for key, value in batch_eval_dict.items(): #remaining are ACC_BREAKDOWN batch_to_type determined subset mask accuracies
                if not value is None: ACC_AGGREGATE[key].append(value)
        total_steps += 1

    avg_log_loss = total_log_loss / total_steps
    avg_accuracy = total_accuracy / total_steps
    avg_final_loss = total_final_loss / total_steps
    if batch_to_type:
        avg_cot_pos_acc = total_cot_pos_acc / total_steps
        avg_positional_3sum_acc = positional_3sum_acc / total_steps
        avg_cot_sums_acc = cot_sums_acc / total_steps

    ACC_BREAKDOWN = {}
    if batch_to_type:
        for key, value in ACC_AGGREGATE.items():
            ACC_BREAKDOWN[f'{key}_acc'] = np.mean(value) if not value is [] else None

    if batch_to_type:
        return {'log_loss': avg_log_loss, 'accuracy': avg_accuracy, 'final_loss':avg_final_loss, 'cot_pos_acc': avg_cot_pos_acc, 'positional_3sum_acc':avg_positional_3sum_acc, 'cot_sums_acc':avg_cot_sums_acc, **ACC_BREAKDOWN}
    else:
        return {'log_loss': avg_log_loss, 'accuracy': avg_accuracy, 'final_loss':avg_final_loss, **ACC_BREAKDOWN}


def find_consecutive_dots_mask(labels, word_index_map):
    # Create a mask of where dots are in the labels
    dot_mask = labels == word_index_map['.']

    # Shift the mask to the left, so we can compare each element with its successor
    shifted_dot_mask = torch.roll(dot_mask, shifts=-1, dims=1)

    # We need to ensure that we don't consider the shifted in value at the end of the tensor
    # For simplicity, we can just set the last column to False since it cannot be the start of a consecutive sequence
    shifted_dot_mask[:, -1] = False

    # Find places where both a dot and its next element are dots
    consecutive_dots_mask = dot_mask & shifted_dot_mask

    # Now, we want to check if there's at least one instance of consecutive dots in each sequence
    # Summing along the sequence length (dim=1) gives us the number of consecutive dot pairs
    # We check if this sum is greater than 0, indicating at least one pair of consecutive dots exists
    has_consecutive_dots = consecutive_dots_mask.sum(dim=1) > 0

    return has_consecutive_dots

def dot_tf_batch_to_type(labels, word_index_map, cot_t_bnd):
    '''
    Assume:
    Multiple T/Fs -> CoT
    '.' -> filler
    else -> direct

    Generically used as a lambda labels: dot_tf_batch_to_type(labels, test_set.word_index_map)

    labels: tensor of shape (batch_size, max_len)
    match3dataset: Match3VectorDataset word_index_map
    '''
    dot_mask = find_consecutive_dots_mask(labels, word_index_map)
    cot_mask = ~dot_mask

    return {
        'dot': dot_mask,
        'cot': cot_mask
    }


def match3_vector_eval_step(model, batch, TF_inds, loss, tf=True, batch_to_type=None, cot_tf_bounds=None):
    '''
    TF_inds: a 2-tuple of the indices of the True and False labels in the word index map
    batch_to_type: a function that takes a tensor of labels and returns a tensor of the same dim with the type of each label, 
                   0 for filler, 1 for direct, 2 for CoT 
    '''
    inputs, labels = batch['input_ids'].to("cuda"), batch['labels'].to("cuda")
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)['logits']
            all_loss = loss(outputs, labels)

            ACC_BREAKDOWN = {
                'dot': 0,
                'cot': 0
            }
            if tf:
                ###### Calculate the accuracy over all positional tokens in CoT
                cot_pos_locs = labels>=cot_tf_bounds[0]
                if cot_pos_locs.any():
                    cot_pos_locs = cot_pos_locs & (labels<cot_tf_bounds[1])
                    cot_pos_indices = cot_pos_locs.nonzero(as_tuple=True)
                    cot_pos_labels = labels[cot_pos_locs]
                    cot_pos_preds = torch.argmax(outputs[cot_pos_indices[0], cot_pos_indices[1],:], dim=-1)
                    cot_pos_mat = (cot_pos_labels == cot_pos_preds).float()
                    cot_pos_acc = torch.mean(cot_pos_mat).item()

                    ###### Now calculate the accuracy over Match-2 True results only by flipping the majority parity of the sequence and only checking those positions
                    seq_len = labels.shape[1]
                    seq_dim_parity = 2*(torch.arange(seq_len) % 2)-1  # Create a tensor of 1s and -1s to represent even and odd positions
                    seq_dim_parity = seq_dim_parity.to(labels.device)  # Ensure parity tensor is on the same device as labels
                    # Step 2: Calculate the majority parity for each sequence and flip it
                    parity_cot_pos_locs = cot_pos_locs * seq_dim_parity.unsqueeze(0)
                    # Sum even (1) and odd (0) positions for each sequence to determine the majority
                    parity_counts = parity_cot_pos_locs.sum(dim=1, keepdim=True)
                    # Determine if even (count <0) or odd (count > 0) positions are the majority
                    majority_parity = (parity_counts > 0).long()
                    # Flip majority parity
                    flipped_majority_parity = 1 - majority_parity
                    assert (flipped_majority_parity == 0).all() | (flipped_majority_parity == 1).all(), "Flipped majority parity must be uniformly 0 or 1"
                    # Use broadcasted multiplication to mask out positions not matching the flipped majority parity
                    seq_dim_parity_expanded = seq_dim_parity.unsqueeze(0).expand_as(cot_pos_locs)  # Expand seq_dim_parity to match cot_pos_locs shape
                    flipped_parity_selection = cot_pos_locs & (seq_dim_parity_expanded == flipped_majority_parity.expand_as(seq_dim_parity_expanded))
                    # Step 3: Select labels and predictions based on flipped parity
                    flipped_parity_indices = flipped_parity_selection.nonzero(as_tuple=True)
                    flipped_parity_labels = labels[flipped_parity_selection]
                    flipped_parity_preds = torch.argmax(outputs[flipped_parity_indices[0], flipped_parity_indices[1], :], dim=-1)
                    flipped_parity_mat = (flipped_parity_labels == flipped_parity_preds).float()
                    parity_acc = flipped_parity_mat.mean().item()

                    ###### Calculate CoT digit summations acc
                    cot_sums_locs = labels>=cot_tf_bounds[1]
                    cot_sums_indices = cot_sums_locs.nonzero(as_tuple=True)
                    cot_sums_labels = labels[cot_sums_locs]
                    cot_sums_preds = torch.argmax(outputs[cot_sums_indices[0], cot_sums_indices[1],:], dim=-1)
                    cot_sums_mat = (cot_sums_labels == cot_sums_preds).float()
                    cot_sums_acc = torch.mean(cot_sums_mat).item()
                else:
                    cot_pos_acc = None
                    parity_acc = None
                    cot_sums_acc = None
            else:
                cot_pos_acc = None
                parity_acc = None
                cot_sums_acc = None

            ###### Calculate the accuracy over final tokens pre-EOS
            seq_locs = torch.where(labels==0, 1, 0)
            seq_locs = seq_locs[:,1:]
            seq_locs = torch.cat([seq_locs, torch.zeros_like(seq_locs[:,0:1])], dim=1).bool()
            acc_mat = (labels == torch.argmax(outputs, dim=-1)).float()
            final_acc_mat = acc_mat[seq_locs]

            #check
            final_labels = labels[seq_locs]
            assert ((final_labels == TF_inds[0]) | (final_labels == TF_inds[1])).all(), "Final token must be either True or False"


            if batch_to_type is not None:
                masks_dict = batch_to_type(labels)
                for type_str, mask in masks_dict.items():
                    masked_vals = acc_mat[mask.squeeze()]
                    if masked_vals.numel() > 0:
                        acc = torch.mean(masked_vals).item()
                    else:
                        acc = None
                    ACC_BREAKDOWN[f'{type_str}'] = acc

            final_labels = torch.where(seq_locs==1, labels, torch.tensor(-100))
            final_loss = loss(outputs, final_labels)
            accuracy = torch.mean(final_acc_mat).item()            

    return {'log_loss': all_loss.item(), 'accuracy': accuracy, 'final_loss': final_loss.item(), 'cot_pos_acc':cot_pos_acc, 'positional_3sum_acc':parity_acc, 'cot_sums_acc':cot_sums_acc, **ACC_BREAKDOWN}