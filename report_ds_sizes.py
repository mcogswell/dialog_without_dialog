import options
from misc.dataloader import VQAPoolDataset


params = options.readCommandLine()
data_params = options.data_params(params)

splits = ['train', 'val1', 'val2']
for pt, ps in [('contrast', 2), ('random', 2), ('random', 4), ('random', 9)]:
    data_params['poolType'] = pt
    data_params['poolSize'] = ps
    dataset = VQAPoolDataset(data_params, splits)
    for split in splits:
        dataset.split = split
        print(f'{pt} {ps} {split}, {len(dataset)}')

