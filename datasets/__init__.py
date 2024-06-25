from .stanford_cars import StanfordCars

dataset_list = {
    "stanford_cars": StanfordCars,
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)
