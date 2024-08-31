import time
from const import *
from logic import *


COMPO_SIZE = 6
CHAMP_MAX_COST = 4
TRAIT_NUMBER = 7
begin = time.time()

with open('champs_list.txt', 'w') as f:
    champ_matrix, champ_names = build_champ_matrix(CHAMP_MAX_COST)
    champion_index_matrix = generate_compos_combinaisons(len(champ_names), COMPO_SIZE)
    del CACHE_DICT
    compute_viable_compos(champion_index_matrix, champ_matrix, champ_names, TRAIT_NUMBER, f)
    print('{} compos analysed in {} seconds'.format(len(champion_index_matrix), time.time() - begin))






