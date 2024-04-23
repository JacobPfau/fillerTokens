import numpy as np
from torch.utils.data import Dataset
import torch
from utils import dump_dataset_to_csv
import os
import datetime
import json


class Match2():
    '''
    Class for generating Match2 instances.
    Transform_inputs are the decoded inputs, inputs are the encrypted inputs.
    '''

    def __init__(self, dimension, mod, length, transform=None, inverse_transform=None):
        self.dimension = dimension
        self.mod = mod
        self.length = length
        self.transform = transform
        self.inverse_transform = inverse_transform

        self.random = np.random.default_rng()
    
    def get_corrupted_instance(self, corruption_rate=4/3, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        _, transform_inputs, _ = self.get_true_instance()
        corruptions = self.random.geometric(1/corruption_rate)
        corruptions = np.minimum(corruptions, self.length)
        columns = self.random.integers(0, self.dimension, size=corruptions)
        transform_inputs[:corruptions,columns] = self.random.integers(0, self.mod, size=(corruptions,))
        rng.shuffle(transform_inputs)
        inputs = self.inverse_transform(transform_inputs)
        solution = self.solve(transform_inputs)
        return inputs, transform_inputs, solution

    def get_true_instance(self):
        # Note that this draws from a subset of the space of all possible solutions, since repeated matches are not allowed -- probs ok since neglected subset is smaller by O(mod^dim/len)
        transform_inputs = self.random.integers(0, self.mod, size=(self.length//2,self.dimension))
        inverses = np.mod(self.mod-transform_inputs, self.mod)
        transform_inputs = np.concatenate([transform_inputs, inverses], axis=0)
        if len(transform_inputs) < self.length:
            transform_inputs = np.concatenate([transform_inputs, transform_inputs[:1,:]], axis=0)
        self.random.shuffle(transform_inputs)
        inputs = self.inverse_transform(transform_inputs)
        return inputs, transform_inputs, [1 for _ in inputs]
    
    def solve(self, inputs):
        inverse = self.mod - inputs
        inverse = np.mod(inverse, self.mod)
        inputs_set = set([tuple(row) for row in inputs])
        inverse_present = [tuple(row) in inputs_set for row in inverse]
        return inverse_present

RNG = np.random.default_rng()
TRANSFORM_SEQS = RNG.integers(0, 10, size=(10,100)) #Note this should be fixed to allow for arbitrary data complexity
def lookup_transform(length, dim, transform_ind, mod):# offset, mod):
    transform_seqs = list(TRANSFORM_SEQS[transform_ind,:length*mod])
    def transform(inputs):
        inputs = inputs.copy()
        all_inds = inputs.shape[0]*inputs.shape[1]
        for i in range(all_inds):
            inputs[i//dim,i%dim] = transform_seqs[i]+inputs[i//dim,i%dim]
        inputs = np.mod(inputs, mod)
        return inputs
    
    def inverse_transform(transform_inputs):
        transform_inputs = transform_inputs.copy()
        all_inds = transform_inputs.shape[0]*transform_inputs.shape[1]
        for i in range(all_inds):
            transform_inputs[i//dim,i%dim] = -transform_seqs[i]+transform_inputs[i//dim,i%dim]
        transform_inputs = np.mod(transform_inputs, mod)
        return transform_inputs

    return transform, inverse_transform


def random_lookup_params(length, dimension, mod, rng=None):
    if rng is None:
        rng = np.random.default_rng() 
    transform_ind = rng.integers(0, 10) #TODO generalize
    # offset = rng.integers(0, length-1)
    return length, dimension, transform_ind, mod#offset, mod

def identity_transform(dummy, mod):
    '''
    '''
    def transform(inputs):
        return inputs
    
    def inverse_transform(transform_inputs):
        return transform_inputs
    return transform, inverse_transform


def cat_row(row):
    return ''.join([str(x) for x in row])


def b10_notransform_control_string(inputs, transform_inputs, solution, transform_params):
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' L' + str(sum(solution))
    return st

def b10_basic_string(inputs, transform_inputs, solution, transform_params, rng=None, mod=10):
    if rng is None:
        rng = np.random.default_rng()
    #CoT like cot string and includes the transform
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' T '
    st += ' '.join([str(x) for x in transform_params[:-1]]) #drop final param since that's mod which is fixed
    st += ' ' + ' '.join([cat_row(inputs[r])+' ' +cat_row(row) for r,row in enumerate(transform_inputs[:2])])
    st += ' ' + cat_row(np.mod(transform_inputs[0] + transform_inputs[1],mod))
    for r,row in enumerate(transform_inputs[2:-1]):
        ind = r+2
        if rng.binomial(1, 0.5) or not solution[ind]:
            st += ' ' + ' '.join([cat_row(inputs[ind])+' ' +cat_row(row)])
        else:
            st += ' ' + ' '.join([cat_row(inputs[ind])+' ' +cat_row(np.mod(mod-row, mod))])
    st += ' L' + str(sum(solution))
    # print(transform_inputs)
    # print(solution)
    return st

def b10_repeat_filler_string(inputs, transform_inputs, solution, transform_params):
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' T '
    st += ' '.join([str(x) for x in transform_params[:-1]]) #drop final param since that's mod which is fixed
    st += ' ' + ' '.join([cat_row(row)+' A' for r,row in enumerate(inputs[:2])])
    st += ' A' # For sum supervision
    for r,row in enumerate(inputs[2:-1]):
        ind = r+2
        st += ' ' + ' '.join([cat_row(row)+' A' ])
    st += ' L' + str(sum(solution))
    # print(transform_inputs)
    # print(solution)
    return st

def b10_no_filler_string(inputs, transform_inputs, solution, transform_params):
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' T '
    st += ' '.join([str(x) for x in transform_params[:-1]])
    st += ' L' + str(sum(solution))
    return st

STRING_FUNCTION_MAPPING = {
    'b10_basic': b10_basic_string,
    'b10_repeat': b10_repeat_filler_string,
    'b10_no_filler': b10_no_filler_string
    }



def generate_sample(length, dimension, mod, transform, type='True', corruption_rate=4/3, rng=None):
    if transform=='lookup':
        transform_params = random_lookup_params(length, dimension, mod, rng)
        transform, inverse_transform = lookup_transform(*transform_params)
        transform_params = transform_params[2:] #drop first two params since they're fixed
    elif transform=='identity':
        transform_params = (0, mod)
        transform, inverse_transform = identity_transform(*transform_params)
    else:
        raise ValueError('transform must be lookup or identity')
    m2 = Match2(dimension, mod, length, transform, inverse_transform)
    
    if type=='True':
        inputs, transform_inputs, solution = m2.get_true_instance()
    elif type=='Corrupted':
        inputs, transform_inputs, solution = m2.get_corrupted_instance(corruption_rate=corruption_rate, rng=rng)
    else:
        raise ValueError('type must be True or Corrupted')
    
    return inputs, transform_inputs, solution, transform_params


def GenerateMatch2Dataset(name, train_samples, test_samples,
                          dimension, mod, length, 
                          true_instance_rate=0.5, cot_rate=0.5, no_filler_rate=0, corruption_rate=4/3,
                          transform='lookup', 
                          filler_to_string=b10_repeat_filler_string, cot_to_string=b10_basic_string, no_filler_to_string=b10_no_filler_string,
                          data_path='./data/'):
    """
    Generate a dataset for the Match2 class.
    
    Args:
    - name (str): Name of the dataset
    - train_samples (int): Number of training samples to generate
    - test_samples (int): Number of test samples to generate
    - dimension, mod, length: Parameters for the Match2 class
    - true_instance_rate (float): Rate at which true instances should be used
    
    Returns:
    - None
    """
    if 'b10' in ''.join([filler_to_string.__name__, cot_to_string.__name__, no_filler_to_string.__name__]) and mod>10:
        raise ValueError('Base 10 string functions only work for mod<=10')
    randomizer = np.random.default_rng()
    corruption_vec = randomizer.binomial(1, true_instance_rate, size=train_samples+test_samples)
    corruption_vec = np.where(corruption_vec==1, 'True', 'Corrupted')
    assert cot_rate + no_filler_rate <= 1
    filler_vec = randomizer.choice([0, 1, 2], p=[cot_rate, 1-cot_rate-no_filler_rate, no_filler_rate], size=train_samples+test_samples) #0 is cot, 1 is filler, 2 is no filler

    train_dataset = [generate_sample(length, dimension, mod, transform, type=corruption_vec[i], corruption_rate=corruption_rate, rng=randomizer) for i in range(train_samples)]
    for i, sample in enumerate(train_dataset):
        if filler_vec[i] == 0:
            train_dataset[i] = cot_to_string(*sample, rng=randomizer, mod=mod) #CoT like cot string includes the transform
        elif filler_vec[i] == 1:
            train_dataset[i] = filler_to_string(*sample) #filler like string
        else:
            train_dataset[i] = no_filler_to_string(*sample) #No filler and no cot i.e. straight to answer

    test_dataset = [generate_sample(length, dimension, mod, transform, type=corruption_vec[i], corruption_rate=corruption_rate, rng=randomizer) for i in range(train_samples, train_samples+test_samples)]
    for i, sample in enumerate(test_dataset):
        j = i+train_samples
        if filler_vec[j] == 0:
            test_dataset[i] = cot_to_string(*sample, rng=randomizer, mod=mod)
        elif filler_vec[j] == 1:
            test_dataset[i] = filler_to_string(*sample)
        else:
            test_dataset[i] = no_filler_to_string(*sample)

    # Save hyperparameters
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    if transform=='lookup':
        max_params = (10,)#length) #TODO generalize
    elif transform=='identity':
        max_params = (1,)
    else:
        raise NotImplementedError('Only translate and identity transforms implemented so far')
    hyperparameters_filename = f"args_{name}_{today}.json"
    args = {
        "name": name,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "dimension": dimension,
        "mod": mod,
        "length": length,
        "true_instance_rate": true_instance_rate,
        "transform": transform,
        "max_transform_params": max_params,
        "filler": filler_to_string.__name__,
        "cot": cot_to_string.__name__,
        "no filler": no_filler_to_string.__name__
    }

    with open(os.path.join(data_path, hyperparameters_filename), 'w') as f:
        json.dump(args, f)
        
    train_desc = f"trainset_{today}.csv"
    dump_dataset_to_csv(train_dataset, os.path.join(data_path, name+'_'+train_desc))
    
    test_desc = f"testset_{today}.csv"
    dump_dataset_to_csv(test_dataset, os.path.join(data_path, name+'_'+test_desc))
    
    return


class Match2VectorDataset(Dataset):
    def __init__(self, dataframe, length, tuple_len, mod, transform_max, mask,):
        self.tuple_len = tuple_len
        self.mod = mod
        self.length = length #sequence length, used for task label encoding throughout
        self.transform_max = transform_max
        self.num_transform_params = len(transform_max)
        self.mask = mask
        self.label_dim = self.length + self.tuple_len * self.mod + 3 # 3 to include EOS, 'A' (i.e. filler token), and the masking flag dim
        self.input_dim = self.length + self.tuple_len * self.mod + 3 + sum(self.transform_max) + len(self.transform_max)
        self.tf_dims = (1, self.length)

        self.dataframe = dataframe
        self.max_len = dataframe['text'].str.split().apply(len).max()

        #Validate results:
        print('validate encodings')
        # print('encoded input 0', self.input_ids[0])
        print('raw input 0', dataframe.loc[0,'text'])
        # print('encoded label 0', self.labels[0])
        print('encoded sample 0', self.__getitem__(0))
        for i in range(3):
            print('raw input 0', dataframe.loc[i,'text'])
            in_dict = self.__getitem__(i)
            tensor_in, tensor_label = in_dict['input_ids'], in_dict['labels']
            nonzero_indices = torch.nonzero(tensor_in, as_tuple=True)
            nonzero_indices = [(a.item(), b.item()) for a,b in list(zip(nonzero_indices[0], nonzero_indices[1]))]
            print(f'non-zero inputs {i}', nonzero_indices)
            nonzero_indices = torch.nonzero(tensor_label, as_tuple=True)
            nonzero_indices = [(a.item(), b.item()) for a,b in list(zip(nonzero_indices[0], nonzero_indices[1]))]
            print(f'non-zero labels {i}', nonzero_indices)

    def _tensorize_str(self, str):
        '''
        Dimensions are as follows (all ranges are inclusive)
        0: EOS
        1-len+1: task label
        len+2-tuple_len*mod+len+1: tuple encodings
        '''
        tensor = torch.zeros(self.input_dim,)
        if self.tuple_len==1:
            raise NotImplementedError('Tuple length 1 not implemented')
        elif str=='A': #Assumed to be filler
            tensor[-2] = 1
        elif 'L' in str: #on 'L' we assume it's the label  
            tensor[1 + int(str[1:])] = 1 #drop the 'L'
        else:
            for c, char in enumerate(str):
                tensor[1 + self.length + 2 + c * self.mod + int(char)] = 1 #1 to skip EOS and self.length+2 for task label
        return tensor
    
    def _tensorize_transform(self, transform_list):
        tensor = torch.zeros(self.input_dim,)
        offset = 1 + self.length + 2 + self.tuple_len * self.mod
        for c, char in enumerate(transform_list):
            tensor[offset + int(char)] = 1
            offset += self.transform_max[c] + 1
        return tensor

    def tensorize_labels_worker(self, chunk):
        '''
        Dimensions are as follows (all ranges are inclusive):
        0: EOS
        1-len+1: task label
        len+2-tuple_len*mod+len+2: tuple encodings
        -2: 'A' i.e. filler
        -1: mask
        This drops the first ' ' separated substring, i.e. assuming no BOS, and appends an EOS label at the end.
        '''
        sequences = []
        sequences = torch.zeros(len(chunk), self.max_len, self.label_dim, dtype=torch.short)
        for t,text in enumerate(chunk['text']):
            t_found = 0
            text = text.strip()
            s = 0
            text = text.split(' ')[1:] #offset for LM prediction
            while s <  len(text):
                str = text[s]
                if str == 'T':
                    sequences[t, s:s + self.num_transform_params + 1 , -1] = 1 #Always mask transform params
                    s = s + self.num_transform_params #Note in this case s is incremented here and by 1 at the end of the loop
                    t_found = 1
                elif 'L' in str:
                    sequences[t, s, 1 + int(str[1:])] = 1 
                elif self.mask == 'Final' or (self.mask=='T' and t_found==0): # Final skips all labels below this point. 'T' conditionally skips.
                    sequences[t, s, -1] = 1
                elif 'A' in str:
                    sequences[t, s, -2] = 1
                else: #Not masked, so must be a tuple
                    for c, char in enumerate(str):
                        val = int(char)
                        sequences[t, s, c * self.mod + val + 1 + self.length+1] = 1 #2 to skip the EOS
                s += 1
            sequences[t, s, 0] = 1 #EOS never masked
            sequences[t, s + 1:, -1] = 1 #Post EOS always masked
        return sequences

    def tensorize_dataframe_worker(self, chunk):
        sequences = torch.zeros(len(chunk), self.max_len, self.input_dim, dtype=torch.float16)
        for t, text in enumerate(chunk['text']):
            text = text.strip()
            s = 0
            text = text.split(' ')
            while s <  len(text):
                str = text[s]
                if str == 'T':
                    tens = self._tensorize_transform(text[s+1:s+self.num_transform_params+1])
                    s = s + self.num_transform_params + 1
                    sequences[t, s-1, :] = tens #Write transform params at end of correpsonding string indices, at 'T...' remain 0, all have masked labels
                else:
                    sequences[t, s, :] = self._tensorize_str(str)
                    s += 1
        return sequences


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        single_row_df = self.dataframe.iloc[idx:idx+1]
        input_tensor = self.tensorize_dataframe_worker(single_row_df)
        label_tensor = self.tensorize_labels_worker(single_row_df)
        label_tensor.requires_grad_(False)
        return {"input_ids": input_tensor[0].type(torch.float16), "labels": label_tensor[0].type(torch.float)}
