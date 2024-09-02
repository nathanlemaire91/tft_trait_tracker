import time
from const import *
from logic import *
from math import comb

COMPO_SIZE = 7
CHAMP_MAX_COST = 4
TRAIT_NUMBER = 7
begin = time.time()

with open('champs_list.txt', 'w') as f:

    champ_matrix, champ_names = build_champ_matrix(CHAMP_MAX_COST)
    print('{} champs, {} comps will be analysed'.format(len(champ_matrix), comb(len(champ_matrix), COMPO_SIZE)))
    
    binomial_matrix_begin = time.time()
    champion_index_matrix = generate_compos_combinaisons_dynamic(len(champ_names), COMPO_SIZE)
    print('Binomial matrix generated in {} seconds'.format(time.time() - binomial_matrix_begin))
    print('AFTER', tracemalloc.get_traced_memory())
    
    trait_computation = time.time()
    compute_viable_compos(True, 4, champion_index_matrix, champ_matrix, champ_names, TRAIT_NUMBER, f)
    print('{} compos analysed in {} seconds'.format(len(champion_index_matrix), time.time() - trait_computation))






