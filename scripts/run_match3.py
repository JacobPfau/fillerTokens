import argparse
import wandb
import pandas as pd
import json
import datetime

from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, AutoConfig

from src.train_match3 import vector_train_loop, dot_tf_batch_to_type
from src.utils import get_files_from_name, format_data_file_name, initialize_pythia, InputEmbedCausalTransformer
from src.utils import train_steps, get_optimizer
from src import Match3VectorDataset

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-de','--description', type=str, default='Train', 
                        help='description of the training run')
    
    parser.add_argument('-m', '--model_string', type=str, default='llama')
    parser.add_argument('-cc', '--custom_config', type=str, default='llama_d384l4h6.json', help='custom config file, overrides config. Assumed to be in base_path/misc/')
    parser.add_argument('-f', '--model_file', type=str, default=None)
    parser.add_argument('-pt', '--pre_train', action='store_true')
    parser.add_argument('-ct', '--continue_training', action='store_true')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-be', '--betas', type=float, nargs=2, default=[0.9, 0.95])
    parser.add_argument('-ld', '--lr_decay_on', action='store_false',)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-o', '--optim', type=str, default="adam")
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-mg','--max_grad_norm', type=float, default=1.0)

    parser.add_argument('-ma', '--mask', type=str, help='mask type: P for mask up through encoding, Final for mask everything besides label and eos', default='P')
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
    parser.add_argument('-b', '--base_path', type=str, default='./')
    parser.add_argument('-hf', '--huggingface_path', type=str, default='./hf/')
    parser.add_argument('-st', '--early_stop', action='store_true', help='Early stop based on validation acc')

    parser.add_argument('--no_wdb', action='store_true', help='No wandb logging')
    parser.add_argument('-wdb', '--wandb_project', type=str, help='wandb project name')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_name = args.description.split(' ')[0]
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    hyperparameters_filename = f'{run_name}_{current_datetime}_hyperparameters.json'
    with open(args.base_path+'output_dir/'+hyperparameters_filename, 'w') as f:
        json.dump(vars(args), f)

    # Get the most recent data file which includes args.data_name in name
    train, test, dargs = get_files_from_name(args.data_name, args.base_path)
    if args.train_data is None:
        args.train_data = train
        print("Using train data", train)
    if args.test_data is None:
        args.test_data = test
        print("Using test data", test)
    if args.data_config is None:
        args.data_config = dargs
        print("Using args", dargs)
    
    # Format the data file names to full path as needed
    args.train_data, args.test_data, args.data_config = [format_data_file_name(x, args.base_path) for x in [args.train_data, args.test_data, args.data_config]]

    with open(args.data_config, 'r') as jsonfile:
        data_args = json.load(jsonfile)
    if 'no_label' in data_args.keys():#compatible with original data_args files
        no_label = data_args['no_label']
    else:
        no_label = True

    if args.model_file is not None and args.continue_training:
        args.model_file = args.base_path+'output_dir/'+args.model_file

    if not args.no_wdb:
        wandb.init(project=args.wandb_project, name=f'{run_name}_{current_datetime}')
        wandb.config.update(args) #Note this is updated below as well for dataset config
        wandb.config.update(data_args)

    if 'Pythia' in args.model_string or 'pythia' in args.model_string:
        full_name = "EleutherAI/"+args.model_string
    else:
        full_name = args.model_string
    
    if args.custom_config is None:
        if args.pre_train:
            if 'ythia' in full_name:
                model = GPTNeoXForCausalLM.from_pretrained(
                    full_name,
                    revision="step143000", 
                    cache_dir=args.huggingface_path+'models/', 
                ).to("cuda")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    full_name,
                    cache_dir=args.huggingface_path+'models/', 
                ).to("cuda")
        elif 'ythia' in full_name:
            model = GPTNeoXForCausalLM.from_pretrained(
                full_name,
                revision="step0", 
                cache_dir=args.huggingface_path+'models/', 
            ).to("cuda")
        else:
            raise NotImplementedError
        config = model.config
    else:
        config = AutoConfig.from_pretrained(args.base_path+'misc/'+args.custom_config)
        if 'ythia' in full_name:
            model = GPTNeoXForCausalLM(config).to("cuda")
            initialize_pythia(model, config.hidden_size, config.num_hidden_layers) #This is necessary because HF doesn't initialize the weights properly
        else:
            model = AutoModelForCausalLM.from_config(config).to("cuda")
    
    train_df = pd.read_csv(args.train_data, header=None, names=["text"],)
    test_df = pd.read_csv(args.test_data, header=None, names=["text"],)
    train_set = Match3VectorDataset(train_df, data_args['dimension'], data_args['length'], data_args['mod'], args.mask,)
    test_set = Match3VectorDataset(test_df, data_args['dimension'], data_args['length'], data_args['mod'], args.mask,)

    if args.continue_training:
        print('Note input dimension is updated, and so embedding layer is re-init with new input_dim')
        model = InputEmbedCausalTransformer.from_pretrained(model, args.model_file, train_set.input_dim,).cuda()
    else:
        model = InputEmbedCausalTransformer(model, train_set.input_dim,).cuda()
    if not args.no_wdb:
        wandb.config.update({'model_config':config})

    tf_label_inds = (len(train_set.word_index_map), len(train_set.word_index_map)+train_set.data_len*2)
    if 'batch_to_type' in data_args.keys():
        if data_args['batch_to_type'] == 'dot_tf_batch_to_type':
            batch_to_type = lambda labels: dot_tf_batch_to_type(labels, test_set.word_index_map, tf_label_inds)
        elif data_args['batch_to_type'] == None:
            print('Batch_to_type set to None')
            batch_to_type = None
        else:
            raise NotImplementedError
    print(f'batch_to_type is {batch_to_type}')

    
    epoch_steps, tot_opt_steps, tot_fp, eval_steps, checkpoint_steps = train_steps(args.epochs, args.train_batch_size, args.num_evals, args.accumulation_factor, 
                                                                                   args.checkpoint, train_set)

    print(f'Epoch steps {epoch_steps:.1e}', f'Total opt steps {tot_opt_steps:.1e}', f'Total FP steps {tot_fp:.1e}', 
          f'Eval steps {eval_steps:.1e}', f'Checkpoint steps {checkpoint_steps}')

    train_data_loader = DataLoader(train_set, batch_size=args.train_batch_size//args.accumulation_factor, shuffle=True, num_workers=8)# collate_fn=collate_fn)
    eval_data_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=8)#collate_fn=collate_fn)

    optimizer, decay_scheduler, scaler = get_optimizer(args.optim, args.lr_decay_on, args.weight_decay, args.mpt, args.learning_rate, args.betas[0], args.betas[1], model, tot_opt_steps)

    args_settings = {
    "epochs": args.epochs,
    "accumulation_factor": args.accumulation_factor,
    "optim": args.optim,
    "lr_decay_on": args.lr_decay_on,
    "no_wdb": args.no_wdb,
    "checkpoint": args.checkpoint,
    "max_grad_norm": args.max_grad_norm,
    "base_path": args.base_path,
    "mpt": args.mpt,
    "early_stop": args.early_stop,
    }

    other_settings = {
    "eval_steps": eval_steps,
    "tf_inds": train_set.tf_dims,
    "run_name": run_name,
    "model": model,
    "train_data_loader": train_data_loader,
    "eval_data_loader": eval_data_loader,
    "optimizer": optimizer,
    "decay_scheduler": decay_scheduler,
    "scaler": scaler,
    "batch_to_type": batch_to_type,
    "tf_label_inds": tf_label_inds,
    }

    vector_train_loop(**args_settings, **other_settings)
    print("Training complete. Exiting...")