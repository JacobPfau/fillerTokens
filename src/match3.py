import numpy as np
import torch
import json
import datetime
from src.utils import dump_dataset_to_csv
from torch.utils.data import Dataset
import os

class Match3():
    '''
    '''
    def __init__(self, dimension, mod, length, ):
        self.dimension = dimension
        self.mod = mod
        self.length = length
        self.random = np.random.default_rng()

    def get_instance(self):
        inputs = self.random.integers(0, self.mod, size=(self.length,self.dimension))
        solution = self.solve(inputs)
        return inputs, solution
    
    def get_corrupted_instance(self, corruption_rate=4/3, redo=True, dense=True, probabilistic=True, serial=False):
        inputs = self.random.integers(0, self.mod, size=(2,self.dimension))
        inverses = np.expand_dims(np.mod(self.mod-np.sum(inputs,axis=0), self.mod),0)
        inputs = np.concatenate([inputs, inverses], axis=0)
        corruptions = self.random.geometric(1/corruption_rate)
        corruptions = np.minimum(corruptions, 3)
        columns = self.random.integers(0, self.dimension, size=corruptions)
        inputs[:corruptions,columns] = self.random.integers(0, self.mod, size=(corruptions,))
        rest = self.random.integers(0, self.mod, size=(self.length-3,self.dimension))
        inputs = np.concatenate([inputs, rest], axis=0)
        self.random.shuffle(inputs)
        if serial: 
            solution = self.serial_solve(inputs)
        elif dense and probabilistic: 
            solution = self.probabilistic_dense_solve(inputs)
        elif dense:
            solution = self.dense_solve(inputs)
        else: solution = self.solve(inputs)
        if (dense and not serial and True in [s for _,s in solution]) or (not dense and solution) or (serial and solution[-1]=='True'):
            inputs, solution = self.get_corrupted_instance(corruption_rate, redo, dense, probabilistic=probabilistic, serial=serial)
        return inputs, solution

    def get_true_instance(self, dense=True, probabilistic=True, serial=False):
        # Uniform over correct inputs. p(a=i) is uniform over i and by symmetry so is any other tuple index.
        inputs = self.random.integers(0, self.mod, size=(2,self.dimension))
        inverses = np.expand_dims(np.mod(self.mod-np.sum(inputs,axis=0), self.mod),0)
        inputs = np.concatenate([inputs, inverses], axis=0)
        rest = self.random.integers(0, self.mod, size=(self.length-3,self.dimension))
        inputs = np.concatenate([inputs, rest], axis=0)
        self.random.shuffle(inputs)
        if serial: return inputs, self.serial_solve(inputs)
        elif not dense: return inputs, True
        elif probabilistic: return inputs, self.probabilistic_dense_solve(inputs)
        else: return inputs, self.dense_solve(inputs)
        
    def solve(self, inputs):
        for t,tup in enumerate(inputs):
            if t==self.length-1: return False
            sums = inputs[t+1:,:] + tup
            sums = np.mod(sums, self.mod)
            inverses = np.mod(self.mod-sums, self.mod)
            for i,inv in enumerate(inverses):
                for c, cand in enumerate(inputs[t+1+i+1:,:]):
                    if (cand==inv).all():
                        return True
    
    def probabilistic_dense_solve(self, inputs):
            # This version outputs a random match2 sum, or index of matched if it's matched
            labels = []
            for t,tup in enumerate(inputs):
                if t==self.length-1:
                    ind = self.random.choice(self.dimension)
                    labels.append((str(t),'F'+str(sums[i][ind])))
                    break

                #enumerate over pairs of tuples, combine tup with all inputs after it
                sums = inputs[t+1:,:] + tup
                sums = np.mod(sums, self.mod)
                inverses = np.mod(self.mod-sums, self.mod)
                
                #for each sum inverse, check if it exists in the remaining tuples (after both tup and other summand)
                for i,inv in enumerate(inverses):
                    found = -1
                    offset = t+1+i+1
                    for c, cand in enumerate(inputs[offset:,:]):
                        if (cand==inv).all():
                            found = offset + c
                            break

                    ind = self.random.choice(self.dimension)
                    sum_ind = sums[i][ind]
                    if self.random.binomial(1,0.5)==1: #Randomly include one position of the 2SUM summands
                        if found!=-1: labels.append((str(t),'-'+str(found))) #'-' signifies match, positional encoding of 3rd summand reported
                        else: labels.append((str(t),'F'+str(sum_ind))) #If no match found label includes one of the 2SUM digits
                    else:
                        if found!=-1: labels.append((str(i+t+1),'-'+str(found)))
                        else: labels.append((str(i+t+1),'F'+str(sum_ind)))
            return labels
    
    def serial_solve(self, inputs):
        labels = [] 
        first_digit_inputs = inputs[:,0]
        candidates = []
        three_sum = False
        for d,dig in enumerate(first_digit_inputs[:-2]):
            sums = first_digit_inputs[d+1:-1] + dig
            sums = np.mod(sums, self.mod)
            inverses = np.mod(self.mod-sums, self.mod)
            
            #for each sum inverse, check if it exists in the remaining digits
            for i,inv in enumerate(inverses):
                offset = d+1+i+1
                for c, cand in enumerate(first_digit_inputs[offset:]):
                    if cand==inv:
                        candidates.append([d, d+i+1, offset + c])
                        labels.extend([str(c)+'-' for c in candidates[-1]]) # Append positional index of matched summands (Solving this requires solving Match-3 in the 1D case)
                        cot_dim = self.random.choice(self.dimension)
                        labels.extend([str(inputs[c,cot_dim]) for c in candidates[-1]]) # Append a randomly chosen index from each summand (Solving this requires copying and projecting D-dimensional inputs to their coordinates)
                        intermediate_sum = [np.mod(inputs[d,j]+inputs[d+i+1,j]+inputs[offset+c,j],self.mod) for j in range(1,self.dimension)]
                        intermediate_sum = [str(c) for c in intermediate_sum]   
                        for intermediate in intermediate_sum:
                            labels.append(intermediate)
                            if intermediate!='0':
                                break
                        if all([sum=='0' for sum in intermediate_sum]): 
                            three_sum = True
                            break
                    if three_sum: break
                if three_sum: break
        labels.append(str(three_sum))
        return labels


def cat_row(row):
    return ''.join([str(x) for x in row])

def no_filler_parallel(inputs, solution):
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' P A ' + str(any(['-' in s[-1] for s in solution]))
    return st

def no_filler_serial(inputs, solution):
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' P A ' + solution[-1]
    return st

def dot_filler_serial(inputs, solution_list, num_filler,):
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' P '
    st += '. '*num_filler
    st += 'A ' + solution_list[-1]
    return st

def dot_filler_parallel(inputs, solution_list, num_filler,):
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' P '
    st += '. '*num_filler
    st += 'A ' + str(any(['-' in s[-1] for s in solution_list]))
    return st

def rand_cot(inputs, solution_list,): #rand_schedule_cot_string
    '''
    To string for CoT which randomly include one position of the 2SUM summands 
    '''
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' P '
    for ind1,subsoln in solution_list: 
        if subsoln[0]=='-':
            st += ind1+'-'+' '+str(subsoln[1:])+'-'+' '
        elif subsoln[0]=='F':
            st += ind1+'-'+' '+subsoln[1:]+' '
        else:
            print(subsoln[-1])
            raise ValueError('Unexpected subsoln')
    st += 'A ' + str(any(['-' in s[-1] for s in solution_list]))
    return st

def serial_cot(inputs, solution_list,):
    st = ' ' + ' '.join([cat_row(row) for row in inputs])
    st += ' P '
    for subsoln in solution_list[0:-1]: 
        st += subsoln+' '
    st += 'A ' + solution_list[-1]
    return st

STRING_FUNCTION_MAP = {
        'rand_cot': rand_cot,
        'serial': serial_cot,
    }

def GenerateMatch3Dataset(name, train_samples=int(1e4), test_samples=int(1e3),
                          dimension=3, mod=10, length=7, 
                          true_instance_rate=0.5, cot_rate=0.5, no_filler_rate=0, corruption_rate=4/3, 
                          filler_to_string=None, cot_to_string=rand_cot, no_filler_to_string=None,
                          data_path='/scratch/jp6263/slackV2/data/'):
    """
    Generate a dataset for the Match3 class.
    
    Args:
    - name (str): Name of the dataset
    - train_samples (int): Number of training samples to generate
    - test_samples (int): Number of test samples to generate
    - dimension, mod, length: Parameters for the Match3 class
    - true_instance_rate (float): Rate at which true instances should be used
    
    Returns:
    - None
    """
    if cot_to_string==rand_cot:
        filler_to_string = dot_filler_parallel
        no_filler_to_string = no_filler_parallel
    elif cot_to_string==serial_cot:
        filler_to_string = dot_filler_serial
        no_filler_to_string = no_filler_serial
    else:
        raise ValueError('Unexpected cot_to_string')
    randomizer = np.random.default_rng()
    corruption_vec = randomizer.binomial(1, true_instance_rate, size=train_samples+test_samples)
    assert cot_rate + no_filler_rate <= 1
    filler_rate = 1-cot_rate-no_filler_rate
    filler_vec = randomizer.choice([0, 1, 2], p=[cot_rate, filler_rate, no_filler_rate], size=train_samples+test_samples) #0 is cot, 1 is filler, 2 is no filler

    matcher = Match3(dimension, mod, length)
    filler_length = length**2

    dataset = []
    for i in range(train_samples+test_samples):
        if corruption_vec[i] == 1:
            sample = matcher.get_true_instance(dense=True, serial=(cot_to_string==serial_cot))
        else:
            sample = matcher.get_corrupted_instance(corruption_rate=corruption_rate, dense=True, serial=(cot_to_string==serial_cot))
        dataset.append(sample)
    
    train_dataset, test_dataset = [], []
    for i in range(train_samples):
        sample = dataset[i]
        if filler_vec[i] == 0:
            train_dataset.append(cot_to_string(*sample)) #CoT like cot string includes the transform
        elif filler_vec[i] == 1:
            train_dataset.append(filler_to_string(*sample, num_filler=filler_length)) #filler like string
        else:
            train_dataset.append(no_filler_to_string(*sample)) #No filler and no cot i.e. straight to answer
    for i in range(train_samples, train_samples+test_samples):
        sample = dataset[i]
        if filler_vec[i] == 0:
            test_dataset.append(cot_to_string(*sample))
        elif filler_vec[i] == 1:
            test_dataset.append(filler_to_string(*sample, num_filler=filler_length))
        else:
            test_dataset.append(no_filler_to_string(*sample))

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    if cot_to_string==rand_cot and no_filler_rate!=1 and filler_rate!=0:
        batch_to_type = 'dot_tf_batch_to_type'
    else:
        batch_to_type = None
    hyperparameters_filename = f"args_{name}_{today}.json"
    args = {
        "name": name,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "dimension": dimension,
        "mod": mod,
        "length": length,
        "true_instance_rate": true_instance_rate,
        "cot_rate": cot_rate,
        "no_filler_rate": no_filler_rate,
        "corruption_rate": corruption_rate,
        "filler": filler_to_string.__name__,
        "cot": cot_to_string.__name__,
        "no filler": no_filler_to_string.__name__,
        "batch_to_type": batch_to_type
    }

    with open(os.path.join(data_path, hyperparameters_filename), 'w') as f:
        json.dump(args, f)
        
    train_desc = f"trainset_{today}.csv"
    dump_dataset_to_csv(train_dataset, os.path.join(data_path, name+'_'+train_desc))
    
    test_desc = f"testset_{today}.csv"
    dump_dataset_to_csv(test_dataset, os.path.join(data_path, name+'_'+test_desc))
    
    return

class Match3VectorDataset(Dataset):
    def __init__(self, dataframe, tuple_len, data_len, mod, mask):
        self.tuple_len = tuple_len
        self.mod = mod
        self.mask = mask
        self.dataframe = dataframe
        assert self.mask in ['P', 'Final'], 'Mask must be P or Final'
        if self.mask=='P' and (~self.dataframe['text'].str.contains('P')).any():
            print('Compatibility with legacy data in which P was not included in no filler sequences')
            # Apply the transformation only if the condition is met
            self.dataframe['text'] = self.dataframe['text'].apply(lambda x: x if 'P' in x else x.replace(' A', ' P A'))

        self.data_len = data_len
        self.max_len = dataframe['text'].str.split().apply(len).max() + 1

        # First Pass: Create Word to Index Mapping
        self.word_index_map = self.create_word_index_map(dataframe)
        self.tf_dims = (self.word_index_map['True'], self.word_index_map['False'])

        # Set input dimension based on mapping and digit handling
        # This works whether or not we do tuple index encodings as filler coordination cots, if not using those the corresponding indices will just be unused
        self.input_dim = len(self.word_index_map) + self.data_len + self.tuple_len * self.mod + self.data_len*2 
        #Validate results:
        print('validate encodings')
        # print('encoded input 0', self.input_ids[0])
        print('raw input 0', dataframe.loc[0,'text'])
        # print('encoded label 0', self.labels[0])
        print('encoded sample 0', self.__getitem__(0))

    def create_word_index_map(self, dataframe):
        unique_words = set()
        for text in dataframe['text']:
            for word in text.split():
                if not word[0].isdigit() and word not in ['True', 'False']: #Includes filler tokens, final answer token, maybe end of input token too...
                    unique_words.add(word)

        word_index_map = {'True': 1, 'False': 2}

        sorted_unique_words = sorted(unique_words)
        offset = len(word_index_map) + 1  # Offset by the number of words already in the map
        for i, word in enumerate(sorted_unique_words):
            word_index_map[word] = i + offset

        return word_index_map

    def handle_digit_sequences(self, sequences, t, w, word):
        offset = len(self.word_index_map)
        if '-' in word: # Handle tuple index encodings
            indices = [int(d) for d in word.split('-') if d!='']
            for i,idx in enumerate(indices):
                sequences[t, w, offset + i*self.data_len + idx] = 1
        else:
            if w<self.data_len: # Adds bespoke positional value for initial input tuples but doesn't for later CoT digits.
                sequences[t, w, offset + w] = 1 #TODO  #Unique positional value appended fiddle with this REMOVE TO RELY ON POS EMBEDDING
            for c, char in enumerate(word):
                sequences[t, w, offset + self.data_len*2 + c * self.mod + int(char)] = 1  #Handles the digits in the tuples, and single digits in CoT
    
    def handle_digit_labels(self, sequences, t, w, word):
        offset = len(self.word_index_map)
        if '-' in word: # Handle tuple index encodings
            indices = [int(d) for d in word.split('-') if d!='']
            for i,idx in enumerate(indices):
                sequences[t, w] = offset + i*self.data_len + idx
        else:
            for c, char in enumerate(word):
                sequences[t, w] = offset + self.data_len*2 + c * self.mod + int(char) #Handles the digits in the tuples, and single digits in CoT
                
    def tensorize_inputs_worker(self, chunk):
        sequences = torch.zeros(len(chunk), self.max_len, self.input_dim, dtype=torch.float16)
        for t, text in enumerate(chunk['text']):
            text = text.strip().split(' ')
            for w, word in enumerate(text):
                if word[0].isdigit():
                    self.handle_digit_sequences(sequences, t, w, word)
                else:
                    index = self.word_index_map[word]
                    if index != -1:
                        sequences[t, w, index] = 1
            sequences[t, w + 1:, 0] = 1  # EOS
        return sequences
    
    def tensorize_labels_worker(self, chunk):
        '''
        Labels are determined based on the word index map. The mapping is as follows:
        -100: mask
        0: EOS
        Indices from word_index_map correspond to their respective labels.
        Additional handling for 'P' and 'Final' masking is done here.
        '''
        sequences = torch.zeros(len(chunk), self.max_len, dtype=torch.short)
        for t, text in enumerate(chunk['text']):
            text = text.strip().split(' ')
            marker_found = False
            for w, word in enumerate(text[1:]):
                #Masking conditions met
                if self.mask == 'Final' and not marker_found and word != 'A':
                    sequences[t, w] = -100
                    continue
                elif self.mask == 'Final' and word == 'A':
                    marker_found = True
                    sequences[t, w] = -100
                    continue
                elif self.mask == 'P' and not marker_found and word != 'P':
                    continue
                elif self.mask == 'P' and word == 'P':
                    marker_found = True
                    sequences[t, :w+1] = -100
                    continue
                #Masking conditions not met:
                if word in self.word_index_map:
                    index = self.word_index_map[word]
                    sequences[t, w] = index
                elif word[0].isdigit():
                    # Handle digit sequences as special cases
                    self.handle_digit_labels(sequences, t, w, word)
                else:
                    raise ValueError(f'Unexpected word {word}')                
            sequences[t, w + 1] = 0  # EOS
            sequences[t, w + 2:] = -100  # Post EOS always masked
        return sequences

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        single_row_df = self.dataframe.iloc[idx:idx+1]
        input_tensor = self.tensorize_inputs_worker(single_row_df)
        label_tensor = self.tensorize_labels_worker(single_row_df)
        label_tensor.requires_grad_(False)
        return {"input_ids": input_tensor[0].type(torch.float16), "labels": label_tensor[0].type(torch.int64)}