from torch.nn.functional import softmax
import csv
import textwrap
import torch
import math
import glob
import os
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler

def train_steps(epochs, train_batch_size, num_evals, accumulation_factor, checkpoint, train_set):
    epoch_steps = len(train_set) // train_batch_size
    tot_opt_steps = epochs*epoch_steps
    tot_fp = accumulation_factor*tot_opt_steps
    if num_evals is None:num_evals = max(25, epochs) #default to 25 evals for short runs
    eval_steps = accumulation_factor*epoch_steps//(num_evals//epochs) 
    checkpoint_steps = accumulation_factor*epoch_steps//checkpoint if checkpoint else None
    return epoch_steps,tot_opt_steps,tot_fp,eval_steps,checkpoint_steps


def get_optimizer(optim, lr_decay_on, weight_decay, mpt, learning_rate, adam_beta1, adam_beta2, model, tot_opt_steps):
    if optim=="adam":
        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer")
    if lr_decay_on: 
        def decay_rate(step):
            warm = tot_opt_steps//20
            if step<=warm:
                return step/warm
            else:
                return 1-step/tot_opt_steps
        decay_scheduler = LambdaLR(optimizer, decay_rate)

    if mpt: scaler = GradScaler()
    else: scaler = None
    return optimizer,decay_scheduler,scaler

CROSS_ENTROPY = nn.CrossEntropyLoss()
def reshape_crossent(logits, labels):
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    return CROSS_ENTROPY(logits, labels)

BCE = BCEWithLogitsLoss(reduction='none')
def masked_bce(logits, labels, reweight_final=0, tf_coords=(None,None)):
    if reweight_final: assert tf_coords[0] is not None
    losses = BCE(logits, labels)
    mask = labels[:,:,-1]==1
    masked_tot = torch.sum(mask)*labels.shape[2]
    mask = mask.unsqueeze(-1).expand_as(losses)
    losses = losses.masked_fill(mask,0.)
    if reweight_final: 
        losses[:,-2,tf_coords[0]:tf_coords[1]]*=reweight_final
    num_indices = torch.prod(torch.tensor(losses.shape))
    return torch.sum(losses)/(num_indices-masked_tot)


class MultiLabelCausalTransformer(nn.Module):
    # Takes as forward inputs a tensor of shape (batch_size, seq_len, input_dim)
    def __init__(self, base_model, input_dim, output_dim):
        super(MultiLabelCausalTransformer, self).__init__()
        self.base_model = base_model
        model_dim = self.base_model.config.hidden_size # Assuming base model's hidden size is model_dim. 
        self.input_layer = nn.Linear(input_dim, model_dim)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, inputs):
        inputs_embeds = self.input_layer(inputs)
        outputs = self.base_model(inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1]
        output = self.output_layer(last_hidden_state)
        return output

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, 'model_weights.pt'))

    @classmethod
    def from_pretrained(cls, base_model, save_directory, input_dim, output_dim):
        model = cls(base_model=base_model, input_dim=input_dim, output_dim=output_dim)
        model.load_state_dict(torch.load(os.path.join(save_directory, 'model_weights.pt')))
        return model
    
    
class InputEmbedCausalTransformer(nn.Module):
    # Takes as forward inputs a tensor of shape (batch_size, seq_len, input_dim)
    def __init__(self, base_model, input_dim,):
        super(InputEmbedCausalTransformer, self).__init__()
        self.base_model = base_model
        model_dim = self.base_model.config.hidden_size # Assuming base model's hidden size is model_dim. 
        self.input_layer = nn.Linear(input_dim, model_dim)

    def forward(self, inputs):
        inputs_embeds = self.input_layer(inputs)
        return self.base_model(inputs_embeds=inputs_embeds,)

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, 'model_weights.pt'))

    @classmethod
    def from_pretrained(cls, base_model, save_directory, input_dim=None):
        state_dict = torch.load(os.path.join(save_directory, 'model_weights.pt'))
        prev_input_dim = state_dict['input_layer.weight'].shape[1]
        model = cls(base_model=base_model, input_dim=prev_input_dim)
        model.load_state_dict(state_dict)
        if input_dim is not None:
            model.input_layer = nn.Linear(input_dim, base_model.config.hidden_size)
        return model

 
def format_data_file_name(name, base_path):
    if '/' in name:
        return name
    else:
        return base_path+'data/'+name


def dump_dataset_to_csv(dataset, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for line in dataset:
            csvwriter.writerow([line])

def get_files_from_name(data_name, base_path):
    list_of_files = glob.glob(base_path+'data/*') 
    # Sort files by modification time
    list_of_files.sort(key=os.path.getmtime)
        
    train_data = next((file for file in reversed(list_of_files) if ("train" in file and data_name in file)), None)
    test_data = next((file for file in reversed(list_of_files) if ("test" in file and data_name in file)), None)
    data_config = next((file for file in reversed(list_of_files) if ("args" in file and data_name in file)), None)
    return train_data, test_data, data_config


def pprint_model_predictions(labels, logits, width=120, n_chars=3):
    '''
    Pretty prints model predictions alongside true labels and input sequences.

    Parameters:
    - inputs: Tensor of input sequences.
    - labels: Tensor of true labels for the sequences.
    - logits: Tensor of model's logits for each sequence element.
    - width: Width for the printed output to wrap around.
    - n_chars: Number of characters to display per sequence element.
    '''
    probabilities = softmax(logits, dim=-1)
    max_prob_indices = probabilities.argmax(dim=-1)
    token_probs = probabilities.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).numpy()

    # Convert tensors to displayable format
    labels_str = ['|'.join([str(int(id)).ljust(n_chars) for id in sequence]) for sequence in labels]
    preds_str = ['|'.join([str(int(id)).ljust(n_chars) for id in sequence]) for sequence in max_prob_indices]
    probs_str = ['|'.join([str(int(prob * 100)).ljust(n_chars) for prob in sequence_probs]) for sequence_probs in token_probs]

    # Use textwrap to wrap the strings
    for label_line, pred_line, prob_line in zip(labels_str, preds_str, probs_str):
        wrapped_labels = textwrap.wrap(label_line, width=width)
        wrapped_preds = textwrap.wrap(pred_line, width=width)
        wrapped_probs = textwrap.wrap(prob_line, width=width)

        for line in range(len(wrapped_labels)):
            print('Label :', wrapped_labels[line])
            print('Preds :', wrapped_preds[line])
            print('Probs :', wrapped_probs[line])
            print()  # Add an empty line between entries


def pprint_logits(inputs, logits, tokenizer, width=120, n_chars=3):
    '''
    Pretty print logits for a single input

    '''
    char_space = ' '*n_chars
    probabilities = softmax(logits, dim=-1)
    # Get the indices of the tokens with maximum probability
    max_prob_indices = probabilities.argmax(dim=-1)
    next_input_ids = inputs[:,1:]
    token_probs = probabilities.gather(dim=-1, index=next_input_ids.unsqueeze(-1)).squeeze(-1).cpu().numpy()[0]

    input_tokens = [tokenizer.decode(id) for id in inputs[0]]
    input_tokens = [token[:n_chars]+char_space[:max(0,n_chars-len(token))] for token in input_tokens]
    tokens_str = '|'.join(input_tokens)

    token_probs = [f"{int(prob*(100))}" for prob in token_probs]
    token_probs = [char_space]+[prob+char_space[:max(0,n_chars-len(prob))] for prob in token_probs]
    probs_str = '|'.join(token_probs)

    # Decode the maximum probability token indices to get the corresponding tokens
    max_prob_tokens = [tokenizer.decode(id) for id in max_prob_indices[0]]
    max_prob_tokens = [char_space]+[token[:n_chars]+char_space[:max(0,n_chars-len(token))] for token in max_prob_tokens]
    max_prob_tokens_str = '|'.join(max_prob_tokens)

    # Use textwrap to wrap the strings
    wrapped_tokens = textwrap.wrap(tokens_str, width=width)
    wrapped_probs = textwrap.wrap(probs_str, width=width)
    max_prob_tokens_str = textwrap.wrap(max_prob_tokens_str, width=width)

    print('Note all tokens are truncated/padded to n_chars characters')
    # Print the tokens and probabilities
    for l, line in enumerate(wrapped_tokens):
        print('Input: ', line)
        print('Probs: ', wrapped_probs[l])
        print('Top1 : ', max_prob_tokens_str[l])
        print()  # Add an empty line between pairs of lines


def initialize_pythia(module, dim, n_layers):
    children = list(module.named_modules())
    for name, mod in children[1:]:
        if hasattr(mod, 'weight'):
            if name in ['query_key_value', 'dense', 'dense_h_to_4h', 'embed_in', 'embed_out']:
                torch.nn.init.normal_(mod.weight, mean=0.0, std=math.sqrt(2 / (5 * dim)))
            elif name in ['dense_4h_to_h',]:
                torch.nn.init.normal_(mod.weight, mean=0.0, std=2 / n_layers / math.sqrt(dim))
        else:
            initialize_pythia(mod, dim, n_layers)


def copy_model(source_model, target_model, grads_off=False, num_layers=None):
    '''
    Copy weights from source_model to target_model (skip final layer). Models must have the same architecture, and dimensions. Target model may have more layers.
    '''
    target_children = list(target_model.named_modules())
    for ind, source_mod_pair in enumerate(source_model.named_modules()):
        name, source_mod = source_mod_pair
        target_mod = target_children[ind][1]
        if 'final' in name:
            return
        if num_layers and 'layers.'+str(num_layers) in name:
            return
        if hasattr(source_mod, 'weight'):
            if hasattr(target_mod, 'weight'):
                target_mod.weight.data.copy_(source_mod.weight.data)
                if grads_off:
                    target_mod.weight.requires_grad = False
            else:
                print('no weight')
                print(target_children[ind][0], source_mod_pair[0])
        if hasattr(source_mod, 'bias') and source_mod.bias is not None:
            target_mod.bias.data.copy_(source_mod.bias.data)
            if grads_off:
                    target_mod.bias.requires_grad = False

def freeze_model(model, num_layers):
    '''
    Freeze all layers except layers past num_layers
    '''
    found = False
    model.requires_grad = False
    for _, source_mod_pair in enumerate(model.named_modules()):
        name, source_mod = source_mod_pair
        if not found and 'layers.'+str(num_layers) in name:
            found = True
        elif not found or 'base_model' not in name: 
            continue
        print('Setting grads on for: ', name)
        for param in source_mod.parameters():
            param.requires_grad = True

def remove_dots(df, filler_percentage):
    
    def remove_filler(text, filler_percentage):
        filler = ' .'
        count = text.count(filler)
        remove_count = int(count * (1-filler_percentage))
        new_text = text
        for _ in range(remove_count):
            new_text = new_text.replace(filler, '', 1)  # Replace first occurrence
        return new_text
    
    df['text'] = df['text'].apply(lambda x: remove_filler(x, filler_percentage))
    return df