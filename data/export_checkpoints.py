import glob
import shutil
import os
import tqdm

filenames = glob.glob('PLMs/*/*')

for filename in tqdm.tqdm(filenames):
    if 'mlm/' in filename and not 'checkpoint' in filename:
        new_filename = filename.replace('PLMs', 'FINAL_MODELS')
        new_dir = '/'.join(new_filename.split('/')[:-1])
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        shutil.copy(filename, new_filename)
