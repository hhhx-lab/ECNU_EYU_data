import json
from os import listdir
from os.path import join

root_data_dir = '../../DataSet/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
validation_L = []
for case in listdir(root_data_dir):
    entry_case = {
        'seg': f'{case}/{case}-seg.nii.gz',
        't1c': f'{case}/{case}-t1c.nii.gz',
        't1n': f'{case}/{case}-t1n.nii.gz',
        't2f': f'{case}/{case}-t2f.nii.gz',
        't2w': f'{case}/{case}-t2w.nii.gz'
    }
    validation_L.append(entry_case)

with open('./BraTS2023-Missing_modal_training_data_split.json', 'w') as json_file:
    json.dump({'training': validation_L}, json_file, indent=4)
print('JSON created successfully.')
