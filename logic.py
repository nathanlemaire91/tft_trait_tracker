import numpy as np
from const import *
import torch
import math
import itertools
import tracemalloc
tracemalloc.start()

def build_champ_matrix(min_cost):
    matrix = []
    champ_names = []
    for champ in all_champions:
        if champ["cost"] <= min_cost:
            champ_names.append(champ["name"])
            array = np.array([0] * len(min_trait_array))
            for trait in champ["traits"]:
                if trait in str_trait_array:
                    array[str_trait_array.index(trait)] = 1
            matrix.append(array)
    return np.vstack(tuple(matrix)), np.array(champ_names)

def compute_viable_compos(gpu, split_size, champ_matrix, *argv):    
    if gpu:
        current, _ = tracemalloc.get_traced_memory()
        print('Memory before : {}g'.format(current / 1e9))
        
        matrix_split = np.split(champ_matrix, indices_or_sections=split_size)
        
        current, _ = tracemalloc.get_traced_memory()
        print('Memory after : {}g'.format(current / 1e9))
        
        for sub_matrix in matrix_split:
            
            current, _ = tracemalloc.get_traced_memory()
            print('Memory iteration : {}g'.format(current / 1e9))
            
            compute_viable_compos_gpu(sub_matrix, *argv)
    else:
        compute_viable_compos_cpu(*argv)
        
def compute_viable_compos_gpu(index_matrix, champ_matrix, champ_names, number_of_minimum_traits, out_file):    
    index_matrix_gpu = torch.from_numpy(index_matrix).to(dtype=torch.float16, device='cuda')
    champ_matrix_gpu = torch.from_numpy(champ_matrix).to(dtype=torch.float16, device='cuda')
    min_trait_array_gpu = torch.from_numpy(min_trait_array).to(dtype=torch.float16, device='cuda')
    
    compo_traits = torch.matmul(index_matrix_gpu, champ_matrix_gpu)
    number_of_traits = torch.sum(torch.ge(compo_traits, min_trait_array_gpu), dim = 1)
    validated_traits_compos_gpu = torch.argwhere((number_of_traits >= number_of_minimum_traits))
    champ_index_validated_compo = torch.squeeze(index_matrix_gpu[validated_traits_compos_gpu]).to(device='cpu').numpy()
    if len(champ_index_validated_compo) > 0:
        out_file.write('\n'.join(np.apply_along_axis(lambda row: ' '.join(champ_names[row == 1]), 1, champ_index_validated_compo)))
        out_file.write('\n')
    

def compute_viable_compos_cpu(index_matrix, champ_matrix, champ_names, number_of_minimum_traits, out_file):
    compo_traits = np.matmul(index_matrix, champ_matrix)
    number_of_traits = np.sum(compo_traits >= min_trait_array, axis = 1)
    validated_traits_compos = number_of_traits >= number_of_minimum_traits
    champ_index_validated_compo = index_matrix[validated_traits_compos]
    if len(champ_index_validated_compo) > 0:
        out_file.write('\n'.join(np.apply_along_axis(lambda row: ' '.join(champ_names[row == 1]), 1, champ_index_validated_compo)))
        out_file.write('\n')
        
def generate_compos_combinaisons(remaining_length, remaining_number):
    global CACHE_DICT
    if((remaining_length, remaining_number) in CACHE_DICT):
        return CACHE_DICT[(remaining_length, remaining_number)]

    if remaining_number == 0:
        return np.zeros((1, remaining_length), dtype = np.int8)
    if remaining_number == remaining_length:
        return np.ones((1, remaining_length), dtype = np.int8)

    k_n_minus_1 = generate_compos_combinaisons(remaining_length-1, remaining_number)
    k_minus_1_n_minus_1 = generate_compos_combinaisons(remaining_length-1, remaining_number-1)

    k_n_matrix = np.insert(
        np.vstack([k_n_minus_1, k_minus_1_n_minus_1]), 
        0, 
        np.concatenate([
            np.zeros(k_n_minus_1.shape[0]), 
            np.ones(k_minus_1_n_minus_1.shape[0])
        ]), 
        axis = 1
    )
    
    CACHE_DICT[(remaining_length, remaining_number)] = k_n_matrix

    return k_n_matrix
    
def generate_compos_combinaisons_dynamic(max_length, max_compo_size):
    dynamic_column = []
                                                                                                
    for i in range(1, max_length+1):
        new_dynamic_column = [0] * i
        print('Binomial Matrix : rank {}, min k : {}'.format(i, max(1, max_compo_size+i-max_length)))
        for j in range(max(1, max_compo_size+i-max_length), min(i, max_compo_size)+1):
            if j == 1:
                new_dynamic_column[j-1] = np.eye(i, dtype = np.int8)
            elif j == i:
                new_dynamic_column[j-1] = np.ones((1, i), dtype=np.int8)
            else:
                k_n_minus_1 = dynamic_column[j-1]
                k_minus_1_n_minus_1 = dynamic_column[j-2]

                k_n_matrix = np.insert(
                    np.vstack([k_n_minus_1, k_minus_1_n_minus_1]), 
                    0, 
                    np.concatenate([
                        np.zeros(k_n_minus_1.shape[0]), 
                        np.ones(k_minus_1_n_minus_1.shape[0])
                    ]), 
                    axis = 1
                )

                new_dynamic_column[j-1] = k_n_matrix
            
        dynamic_column = new_dynamic_column
        
        print('RANK {} : {}'.format(i, tracemalloc.get_traced_memory()))
            
    return dynamic_column[j-1]


def generate_compos_combinaisons_dynamic_2(max_length, max_compo_size):
    previous_matrix = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    
    for i in range(2, max_length+1):   
        zero_in_i = np.zeros((1, i), dtype = np.int8)
        if i <= max_compo_size:
            i_in_i = np.ones((1, i), dtype = np.int8)
        
        local_i = min(i, max_compo_size)
        length_k_minus_1_in_n = itertools.chain.from_iterable((math.comb(i-1, k), math.comb(i-1, k-1)) for k in range(min(max_compo_size, i)+1, 1, -1))

    
