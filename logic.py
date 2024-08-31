import numpy as np
from const import *



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

def compute_viable_compos(index_matrix, champ_matrix, champ_names, numer_of_minimum_traits, out_file):
    compo_traits = np.matmul(index_matrix, champ_matrix)
    number_of_traits = np.sum(compo_traits >= min_trait_array, axis = 1)
    validated_traits_compos = number_of_traits >= numer_of_minimum_traits
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
