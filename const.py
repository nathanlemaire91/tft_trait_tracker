import numpy as np

activation_thresholds = {
  "Arcana": 2,
  "Chrono": 2,
  "Dragon": 2,
  "Eldritch": 3,
  "Faerie": 2,
  "Frost": 3,
  "Honeymancy": 3,
  "Portal": 3,
  "Pyro": 2,
  "Sugarcraft": 2,
  "Witchcraft": 2,
  "Bastion": 2,
  "Blaster": 2,
  "Hunter": 2,
  "Incantor": 2,
  "Mage": 3,
  "Multistriker": 3,
  "Preserver": 2,
  "Scholar": 2,
  "Shapeshifter": 2,
  "Vanguard": 2,
  "Warrior": 2
}

min_trait_array = list(activation_thresholds.values())
str_trait_array = list(activation_thresholds.keys())

all_champions = [
  {"name": "Briar", "cost": 5, "traits": ["Eldritch", "Ravenous", "Shapeshifter"]},
  {"name": "Camille", "cost": 5, "traits": ["Chrono", "Multistriker"]},
  {"name": "Diana", "cost": 5, "traits": ["Frost", "Bastion"]},
  {"name": "Milio", "cost": 5, "traits": ["Faerie", "Scholar"]},
  {"name": "Morgana", "cost": 5, "traits": ["Witchcraft", "Preserver"]},
  {"name": "Norra & Yuumi", "cost": 5, "traits": ["Portal", "Mage"]},
  {"name": "Smolder", "cost": 5, "traits": ["Dragon", "Blaster"]},
  {"name": "Xerath", "cost": 5, "traits": ["Arcana", "Ascendant"]},
  {"name": "Fiora", "cost": 4, "traits": ["Witchcraft", "Warrior"]},
  {"name": "Gwen", "cost": 4, "traits": ["Sugarcraft", "Warrior"]},
  {"name": "Kalista", "cost": 4, "traits": ["Faerie", "Multistriker"]},
  {"name": "Karma", "cost": 4, "traits": ["Chrono", "Incantor"]},
  {"name": "Nami", "cost": 4, "traits": ["Eldritch", "Mage"]},
  {"name": "Nasus", "cost": 4, "traits": ["Pyro", "Shapeshifter"]},
  {"name": "Olaf", "cost": 4, "traits": ["Frost", "Hunter"]},
  {"name": "Rakan", "cost": 4, "traits": ["Faerie", "Preserver"]},
  {"name": "Ryze", "cost": 4, "traits": ["Portal", "Incantor"]},
  {"name": "Tahm Kench", "cost": 4, "traits": ["Arcana", "Vanguard"]},
  {"name": "Taric", "cost": 4, "traits": ["Portal", "Bastion"]},
  {"name": "Varus", "cost": 4, "traits": ["Pyro", "Blaster"]},
  {"name": "Bard", "cost": 3, "traits": ["Sugarcraft", "Preserver", "Scholar"]},
  {"name": "Ezreal", "cost": 3, "traits": ["Portal", "Blaster"]},
  {"name": "Hecarim", "cost": 3, "traits": ["Arcana", "Bastion", "Multistriker"]},
  {"name": "Hwei", "cost": 3, "traits": ["Frost", "Blaster"]},
  {"name": "Jinx", "cost": 3, "traits": ["Sugarcraft", "Hunter"]},
  {"name": "Katarina", "cost": 3, "traits": ["Faerie", "Warrior"]},
  {"name": "Mordekaiser", "cost": 3, "traits": ["Eldritch", "Vanguard"]},
  {"name": "Neeko", "cost": 3, "traits": ["Witchcraft", "Shapeshifter"]},
  {"name": "Shen", "cost": 3, "traits": ["Pyro", "Bastion"]},
  {"name": "Swain", "cost": 3, "traits": ["Frost", "Shapeshifter"]},
  {"name": "Veigar", "cost": 3, "traits": ["Honeymancy", "Mage"]},
  {"name": "Vex", "cost": 3, "traits": ["Chrono", "Mage"]},
  {"name": "Wukong", "cost": 3, "traits": ["Druid"]},
  {"name": "Ahri", "cost": 2, "traits": ["Arcana", "Scholar"]},
  {"name": "Akali", "cost": 2, "traits": ["Pyro", "Warrior", "Multistriker"]},
  {"name": "Cassiopeia", "cost": 2, "traits": ["Witchcraft", "Incantor"]},
  {"name": "Galio", "cost": 2, "traits": ["Portal", "Vanguard", "Mage"]},
  {"name": "Kassadin", "cost": 2, "traits": ["Portal", "Multistriker"]},
  {"name": "Kog'Maw", "cost": 2, "traits": ["Honeymancy", "Hunter"]},
  {"name": "Nilah", "cost": 2, "traits": ["Eldritch", "Warrior"]},
  {"name": "Nunu", "cost": 2, "traits": ["Honeymancy", "Bastion"]},
  {"name": "Rumble", "cost": 2, "traits": ["Sugarcraft", "Vanguard", "Blaster"]},
  {"name": "Shyvana", "cost": 2, "traits": ["Dragon", "Shapeshifter"]},
  {"name": "Syndra", "cost": 2, "traits": ["Eldritch", "Incantor"]},
  {"name": "Tristana", "cost": 2, "traits": ["Faerie", "Blaster"]},
  {"name": "Zilean", "cost": 2, "traits": ["Frost", "Chrono", "Preserver"]},
  {"name": "Ashe", "cost": 1, "traits": ["Eldritch", "Multistriker"]},
  {"name": "Blitzcrank", "cost": 1, "traits": ["Honeymancy", "Vanguard"]},
  {"name": "Elise", "cost": 1, "traits": ["Eldritch", "Shapeshifter"]},
  {"name": "Jax", "cost": 1, "traits": ["Chrono", "Multistriker"]},
  {"name": "Jayce", "cost": 1, "traits": ["Portal", "Shapeshifter"]},
  {"name": "Lillia", "cost": 1, "traits": ["Faerie", "Bastion"]},
  {"name": "Nomsy", "cost": 1, "traits": ["Dragon", "Hunter"]},
  {"name": "Poppy", "cost": 1, "traits": ["Witchcraft", "Bastion"]},
  {"name": "Seraphine", "cost": 1, "traits": ["Faerie", "Mage"]},
  {"name": "Soraka", "cost": 1, "traits": ["Sugarcraft", "Mage"]},
  {"name": "Twitch", "cost": 1, "traits": ["Frost", "Hunter"]},
  {"name": "Warwick", "cost": 1, "traits": ["Frost", "Vanguard"]},
  {"name": "Ziggs", "cost": 1, "traits": ["Honeymancy", "Incantor"]},
  {"name": "Zoe", "cost": 1, "traits": ["Portal", "Witchcraft", "Scholar"]}
]

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

CACHE_DICT = {}