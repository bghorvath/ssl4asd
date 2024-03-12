import numpy as np
import os
import soundfile as sf
import librosa
from tqdm import tqdm
from model import adjust_size


########################################################################################################################
# Load data and compute embeddings
########################################################################################################################
def load_data():
    target_sr = 16000

    # load train data
    print('Loading train data')
    categories = os.listdir("./dev_data")+os.listdir("./eval_data")
    categories_dev = os.listdir("./dev_data")
    categories_eval = os.listdir("./eval_data")

    if os.path.isfile(str(target_sr) + '_train_raw.npy'):
        train_raw = np.load(str(target_sr) + '_train_raw.npy')
        train_ids = np.load('train_ids.npy')
        train_files = np.load('train_files.npy')
        train_atts = np.load('train_atts.npy')
        train_domains = np.load('train_domains.npy')
    else:
        train_raw = []
        train_ids = []
        train_files = []
        train_atts = []
        train_domains = []
        dicts = ['./dev_data/', './eval_data/']
        eps = 1e-12
        for dict in dicts:
            for label, category in enumerate(os.listdir(dict)):
                print(category)
                for count, file in tqdm(enumerate(os.listdir(dict + category + "/train")),
                                        total=len(os.listdir(dict + category + "/train"))):
                    if file.endswith('.wav'):
                        file_path = dict + category + "/train/" + file
                        wav, fs = sf.read(file_path)
                        raw = librosa.core.to_mono(wav.transpose()).transpose()
                        raw = adjust_size(raw, 288000)
                        train_raw.append(raw)
                        train_ids.append(category + '_' + file.split('_')[1])
                        train_files.append(file_path)
                        train_domains.append(file.split('_')[2])
                        train_atts.append('_'.join(file.split('.wav')[0].split('_')[6:]))
        # reshape arrays and store
        train_ids = np.array(train_ids)
        train_files = np.array(train_files)
        train_raw = np.expand_dims(np.array(train_raw, dtype=np.float32), axis=-1)
        train_atts = np.array(train_atts)
        train_domains = np.array(train_domains)
        np.save('train_ids.npy', train_ids)
        np.save('train_files.npy', train_files)
        np.save('train_atts.npy', train_atts)
        np.save('train_domains.npy', train_domains)
        np.save(str(target_sr) + '_train_raw.npy', train_raw)

    # load evaluation data
    print('Loading evaluation data')
    if os.path.isfile(str(target_sr) + '_eval_raw.npy'):
        eval_raw = np.load(str(target_sr) + '_eval_raw.npy')
        eval_ids = np.load('eval_ids.npy')
        eval_normal = np.load('eval_normal.npy')
        eval_files = np.load('eval_files.npy')
        eval_atts = np.load('eval_atts.npy')
        eval_domains = np.load('eval_domains.npy')
    else:
        eval_raw = []
        eval_ids = []
        eval_normal = []
        eval_files = []
        eval_atts = []
        eval_domains = []
        eps = 1e-12
        for label, category in enumerate(os.listdir("./dev_data/")):
            print(category)
            for count, file in tqdm(enumerate(os.listdir("./dev_data/" + category + "/test")),
                                    total=len(os.listdir("./dev_data/" + category + "/test"))):
                if file.endswith('.wav'):
                    file_path = "./dev_data/" + category + "/test/" + file
                    wav, fs = sf.read(file_path)
                    raw = librosa.core.to_mono(wav.transpose()).transpose()
                    raw = adjust_size(raw, 288000) #288000 or 192000
                    eval_raw.append(raw)
                    eval_ids.append(category + '_' + file.split('_')[1])
                    eval_normal.append(file.split('_test_')[1].split('_')[0] == 'normal')
                    eval_files.append(file_path)
                    eval_domains.append(file.split('_')[2])
                    eval_atts.append('_'.join(file.split('.wav')[0].split('_')[6:]))
        # reshape arrays and store
        eval_ids = np.array(eval_ids)
        eval_normal = np.array(eval_normal)
        eval_files = np.array(eval_files)
        eval_atts = np.array(eval_atts)
        eval_domains = np.array(eval_domains)
        eval_raw = np.expand_dims(np.array(eval_raw, dtype=np.float32), axis=-1)
        np.save('eval_ids.npy', eval_ids)
        np.save('eval_normal.npy', eval_normal)
        np.save('eval_files.npy', eval_files)
        np.save('eval_atts.npy', eval_atts)
        np.save('eval_domains.npy', eval_domains)
        np.save(str(target_sr) + '_eval_raw.npy', eval_raw)

    # load test data
    print('Loading test data')
    if os.path.isfile(str(target_sr) + '_test_raw.npy'):
        test_raw = np.load(str(target_sr) + '_test_raw.npy')
        test_ids = np.load('test_ids.npy')
        test_files = np.load('test_files.npy')
    else:
        test_raw = []
        test_ids = []
        test_files = []
        eps = 1e-12
        for label, category in enumerate(os.listdir("./eval_data/")):
            print(category)
            for count, file in tqdm(enumerate(os.listdir("./eval_data/" + category + "/test")),
                                    total=len(os.listdir("./eval_data/" + category + "/test"))):
                if file.endswith('.wav'):
                    file_path = "./eval_data/" + category + "/test/" + file
                    wav, fs = sf.read(file_path)
                    raw = librosa.core.to_mono(wav.transpose()).transpose()
                    raw = adjust_size(raw, 288000) #288000 or 192000
                    test_raw.append(raw)
                    test_ids.append(category + '_' + file.split('_')[1])
                    test_files.append(file_path)
        # reshape arrays and store
        test_ids = np.array(test_ids)
        test_files = np.array(test_files)
        test_raw = np.expand_dims(np.array(test_raw, dtype=np.float32), axis=-1)
        np.save('test_ids.npy', test_ids)
        np.save('test_files.npy', test_files)
        np.save(str(target_sr) + '_test_raw.npy', test_raw)

    return target_sr, \
    train_files, eval_files, test_files, \
    train_ids, eval_ids, test_ids, \
    train_atts, eval_atts, \
    train_raw, eval_raw, test_raw, \
    eval_normal, eval_domains, \
    categories_dev, categories_eval
