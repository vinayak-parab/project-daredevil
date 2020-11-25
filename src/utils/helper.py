import json
import numpy as np

def samples_by_language(path_to_language_dict, path_to_sample_list,lang='all'):

    samples = list()

    with open(path_to_language_dict) as file:
        language_annotations = json.load(file)
    
    sample_list = np.load(path_to_sample_list)
    
    for s in sample_list:
        if lang == 'all':
            samples.append(s)
        else:
            if lang == language_annotations[s]:
                samples.append(s)
                
    return samples