import argparse
import json
import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GPTNeoXForCausalLM
import wandb

from src.match2 import Match2Dataset, Match2VectorDataset
from src.train_match2 import vector_train_loop
from src.utils import get_files_from_name, format_data_file_name, initialize_pythia, MultiLabelCausalTransformer, get_optimizer, train_steps
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-de','--description', type=str, default='Train', 
                        help='description of the training run')
    
    parser.add_argument('-m', '--model_string', type=str, default='llama')
    parser.add_argument('-cc', '--custom_config', type=str, default=None, help='custom config file, overrides config. Assumed to be in base_path/misc/')
    parser.add_argument('-f', '--model_file', type=str, default=None)
    parser.add_argument('-pt', '--pre_train', action='store_true')
    parser.add_argument('-ct', '--continue_training', action='store_true')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-ld', '--lr_decay_on', action='store_false',)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-o', '--optim', type=str, default="adam")
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-mg','--max_grad_norm', type=float, default=1.0)
    parser.add_argument('-wl', '--weight_labels', type=float, default=0., help='Weight for final label dimension if using vectorized data and BCE loss, 0 for no reweighting')

    parser.add_argument('-ve', '--vectorized_data', action='store_false', help='Use vectorized data')
    parser.add_argument('-ma', '--mask', type=str, help='mask type, None for no masking, T for mask up through transform, Final for mask everything besides label and eos', default='T')
    parser.add_argument('-bs', '--train_batch_size', type=int, default=256)
    parser.add_argument('-af','--accumulation_factor', type=int, default=1, help='Number of batches to accumulate gradients over, 1 for no accumulation i.e. normal batch backprop')
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=256)
    parser.add_argument('-ne', '--num_evals', type=int, default=None)

    parser.add_argument('-mpt', '--mpt', action='store_false')

    parser.add_argument('-es','--eval_size', type=int, default=-1)
    parser.add_argument('-cpt', '--checkpoint', type=int, default=1, help='Total number of checkpoints, 1 for once per epoch, 0 for never, 2 for twice per epoch, etc.')

    parser.add_argument('-dn', '--data_name', type=str, default=None, help='data filename, used when tr/te/dc are None')
    parser.add_argument('-tr', '--train_data', type=str, default=None, help='train data filename, None for most recent')
    parser.add_argument('-te', '--test_data', type=str, default=None, help='test data filename, None for most recent')
    parser.add_argument('-dc', '--data_config', type=str, default=None, help='Filename for data config, None for most recent')
    parser.add_argument('-b', '--base_path', type=str, default='/scratch/jp6263/slackV2/')
    parser.add_argument('-hf', '--huggingface_path', type=str, default='/scratch/jp6263/hf/')

    parser.add_argument('-nhf', '--no_wdb', action='store_true')
    parser.add_argument('-wdb', '--wandb_project', type=str, default='slack-slack', help='wandb project name')

    args = parser.parse_arguments()
    
    initial_word = args.description.split(' ')[0]
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    hyperparameters_filename = f'{initial_word}_{current_datetime}_hyperparameters.json'
    with open(args.base_path+'output_dir/'+hyperparameters_filename, 'w') as f:
        json.dump(vars(args), f)

    # Get the most recent data file which includes args.data_name in name
    train, test, dargs = get_files_from_name(args.data_name, args.base_path)
    if args.train_data is None:
        args.train_data = train
    if args.test_data is None:
        args.test_data = test
    if args.data_config is None:
        args.data_config = dargs
    
    # Format the data file names to full path as needed
    args.train_data, args.test_data, args.data_config = [format_data_file_name(x, args.base_path) for x in [args.train_data, args.test_data, args.data_config]]

    with open(args.data_config, 'r') as jsonfile:
        data_args = json.load(jsonfile)
    
    if args.model_file is not None and args.continue_training:
        args.model_file = args.base_path+'output_dir/'+args.model_file

    if not args.no_wdb:
        wandb.init(project=args.wandb_project, name=f'{initial_word}_{current_datetime}')
        wandb.config.update(args) #Note this is updated below as well for dataset config
        wandb.config.update(data_args)

    if 'ythia' in args.model_string: full_name = "EleutherAI/"+args.model_string
    else: full_name = args.model_string
    
    if args.continue_training:
        model = GPTNeoXForCausalLM.from_pretrained(args.model_file,).to("cuda")
        config = model.config
    else:
        if args.custom_config is None:
            if args.pre_train:
                model = GPTNeoXForCausalLM.from_pretrained(
                    full_name,
                    revision="step143000", 
                    cache_dir=args.huggingface_path+'models/', 
                ).to("cuda")
            else:
                model = GPTNeoXForCausalLM.from_pretrained(
                    full_name,
                    revision="step0", 
                    cache_dir=args.huggingface_path+'models/', 
                ).to("cuda")
            config = model.config
        else:
            config = AutoConfig.from_pretrained(args.base_path+'misc/'+args.custom_config)
            if 'ythia' in args.model_string:
                model = GPTNeoXForCausalLM(config).to("cuda")
                initialize_pythia(model, config.hidden_size, config.num_hidden_layers) #This is necessary because HF doesn't initialize the weights properly
            else:
                model = AutoModelForCausalLM.from_config(config).to("cuda")
    if not args.no_wdb:
        wandb.config.update({'model_config':config})
    if 'ythia' in args.model_string: 
        tokenizer = AutoTokenizer.from_pretrained(
                full_name,
                revision="step143000",
                cache_dir=args.huggingface_path+'tokenizers/',
                )
    elif 'llama' in args.model_string:
        tokenizer = AutoTokenizer.from_pretrained(
            'JackFram/llama-68m',
            cache_dir=args.huggingface_path+'tokenizers/',
            )
    tokenizer.pad_token = tokenizer.eos_token
    
    train_df = pd.read_csv(args.train_data, header=None, names=["text"],)
    test_df = pd.read_csv(args.test_data, header=None, names=["text"],)

    token_dtype = torch.short
    if args.vectorized_data:
        train_set = Match2VectorDataset(train_df, data_args['length'], data_args['dimension'], data_args['mod'], data_args['max_transform_params'], 
                                        args.mask,)
        test_set = Match2VectorDataset(test_df, data_args['length'], data_args['dimension'], data_args['mod'], data_args['max_transform_params'], 
                                        args.mask,)
    else:
        train_set = Match2Dataset(train_df, tokenizer, mask=args.mask, dtype=token_dtype)
        test_set = Match2Dataset(test_df, tokenizer, mask=None, dtype=token_dtype,)

    if args.vectorized_data:
        model = MultiLabelCausalTransformer(model, train_set.input_dim, train_set.label_dim).cuda()
    

    epoch_steps, tot_opt_steps, tot_fp, eval_steps, checkpoint_steps = train_steps(args.epochs, args.train_batch_size, args.num_evals, args.accumulation_factor, 
                                                                                   args.checkpoint, train_set)

    print(f'Epoch steps {epoch_steps:.1e}', f'Total opt steps {tot_opt_steps:.1e}', f'Total FP steps {tot_fp:.1e}', 
          f'Eval steps {eval_steps:.1e}', f'Checkpoint steps {checkpoint_steps}')

    train_data_loader = DataLoader(train_set, batch_size=args.train_batch_size//args.accumulation_factor, shuffle=True, num_workers=8)# collate_fn=collate_fn)
    eval_data_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=8)#collate_fn=collate_fn)

    optimizer, decay_scheduler, scaler = get_optimizer(args.optim, args.lr_decay_on, args.weight_decay, args.mpt, args.learning_rate, ADAM_BETA1, ADAM_BETA2, model, tot_opt_steps)

    if args.vectorized_data:
        vector_train_loop(train_set.tf_dims, args.weight_labels, args.epochs, args.mpt, args.accumulation_factor, args.optim, args.lr_decay_on, args.no_wdb, args.checkpoint, args.max_grad_norm,
                           args.base_path, initial_word, model, epoch_steps, eval_steps, checkpoint_steps, train_data_loader, eval_data_loader, optimizer, decay_scheduler, scaler)
    else:
        raise NotImplementedError('Only vectorized data is supported at this time')



