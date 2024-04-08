from argparse import ArgumentParser
import pandas as pd
import numpy as np
import keras
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from mixup_layer import MixupLayer
from openl3_idea_aug_layer_classwise import AugLayer
from subcluster_adacos import SCAdaCos
from scipy.stats import hmean
from sklearn.cluster import KMeans
from statex_aug_layer_classwise import StatExLayer
from mdam import MultiDimensionalAttention, FrequencyAttention, TemporalAttention, ChannelAttention
from model import SqueezeAndExcitationBlock, MagnitudeSpectrogram, mixupLoss, length_norm, model_emb_cnn
from data import load_data


parser = ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--aeons", type=int, default=1)
parser.add_argument("--ensemble_size", type=int, default=5)
parser.add_argument("--mdam", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--statex", action="store_true")
parser.add_argument("--featex", action="store_true")
args = parser.parse_args()

target_sr, \
    train_files, eval_files, test_files, \
    train_ids, eval_ids, test_ids, \
    train_atts, eval_atts, \
    train_raw, eval_raw, test_raw, \
    eval_normal, eval_domains, \
    categories_dev, categories_eval = load_data()

# encode ids as labels
le_4train = LabelEncoder()

source_train = np.array([file.split('_')[3] == 'source' for file in train_files.tolist()])
source_eval = np.array([file.split('_')[3] == 'source' for file in eval_files.tolist()])
train_ids_4train = np.array(['###'.join([train_ids[k], train_atts[k], str(source_train[k])]) for k in np.arange(train_ids.shape[0])])
eval_ids_4train = np.array(['###'.join([eval_ids[k], eval_atts[k], str(source_eval[k])]) for k in np.arange(eval_ids.shape[0])])
le_4train.fit(np.concatenate([train_ids_4train, eval_ids_4train], axis=0))
num_classes_4train = len(np.unique(np.concatenate([train_ids_4train, eval_ids_4train], axis=0)))
train_labels_4train = le_4train.transform(train_ids_4train)
eval_labels_4train = le_4train.transform(eval_ids_4train)

le = LabelEncoder()
train_labels = le.fit_transform(train_ids)
eval_labels = le.transform(eval_ids)
test_labels = le.transform(test_ids)
num_classes = len(np.unique(train_labels))

# distinguish between normal and anomalous samples on development set
unknown_raw = eval_raw[~eval_normal]
unknown_labels = eval_labels[~eval_normal]
unknown_labels_4train = eval_labels_4train[~eval_normal]
unknown_files = eval_files[~eval_normal]
unknown_ids = eval_ids[~eval_normal]
unknown_domains = eval_domains[~eval_normal]
source_unknown = source_eval[~eval_normal]
eval_raw = eval_raw[eval_normal]
eval_labels = eval_labels[eval_normal]
eval_labels_4train = eval_labels_4train[eval_normal]
eval_files = eval_files[eval_normal]
eval_ids = eval_ids[eval_normal]
eval_domains = eval_domains[eval_normal]
source_eval = source_eval[eval_normal]

# training parameters
batch_size = args.batch_size
batch_size_test = args.batch_size
epochs = args.epochs
aeons = args.aeons
alpha = 1
n_subclusters = 16
ensemble_size = args.ensemble_size

final_results_dev = np.zeros((ensemble_size, 6))
final_results_eval = np.zeros((ensemble_size, 6))

pred_eval = np.zeros((eval_raw.shape[0], np.unique(train_labels).shape[0]))
pred_unknown = np.zeros((unknown_raw.shape[0], np.unique(train_labels).shape[0]))
pred_test = np.zeros((test_raw.shape[0], np.unique(train_labels).shape[0]))
pred_train = np.zeros((train_labels.shape[0], np.unique(train_labels).shape[0]))

for k_ensemble in np.arange(ensemble_size):
    # prepare scores and domain info
    y_train_cat = keras.utils.to_categorical(train_labels, num_classes=num_classes)
    y_eval_cat = keras.utils.to_categorical(eval_labels, num_classes=num_classes)
    y_unknown_cat = keras.utils.to_categorical(unknown_labels, num_classes=num_classes)

    y_train_cat_4train = keras.utils.to_categorical(train_labels_4train, num_classes=num_classes_4train)
    y_eval_cat_4train = keras.utils.to_categorical(eval_labels_4train, num_classes=num_classes_4train)
    y_unknown_cat_4train = keras.utils.to_categorical(unknown_labels_4train, num_classes=num_classes_4train)

    # compile model
    data_input, label_input, loss_output, loss_output_ssl, loss_output_ssl2 = model_emb_cnn(num_classes=num_classes_4train,
                                                             raw_dim=eval_raw.shape[1], n_subclusters=n_subclusters, use_bias=False,
                                                             mdam=args.mdam, mixup=args.mixup, statex=args.statex, featex=args.featex
                                                             )
    model = tf.keras.Model(inputs=[data_input, label_input], outputs=[loss_output, loss_output_ssl, loss_output_ssl2])
    model.compile(loss=[mixupLoss, mixupLoss, mixupLoss], optimizer=tf.keras.optimizers.Adam() ,loss_weights=[1,0,1])
    print(model.summary())
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1)

    for k in np.arange(aeons):
        print('ensemble iteration: ' + str(k_ensemble+1))
        print('aeon: ' + str(k+1))
        # fit model
        weight_path = 'wts_' + str(k+1) + 'k_' + str(target_sr) + '_' + str(k_ensemble+1) + '_ssl_statex+featex_single.h5'
        if not os.path.isfile(weight_path):
            model.fit(
                [train_raw, y_train_cat_4train], [y_train_cat_4train,y_train_cat_4train,y_train_cat_4train], verbose=1,
                batch_size=batch_size, epochs=epochs,
                validation_data=([eval_raw, y_eval_cat_4train], [y_eval_cat_4train,y_eval_cat_4train,y_eval_cat_4train]),
                callbacks=[tensorboard_callback]
                )
            model.save(weight_path)
        else:
            model = tf.keras.models.load_model(weight_path,
                                               custom_objects={'MixupLayer': MixupLayer, 'mixupLoss': mixupLoss,
                                                               'SCAdaCos': SCAdaCos,
                                                               'MagnitudeSpectrogram': MagnitudeSpectrogram, 'AugLayer': AugLayer, 'StatExLayer': StatExLayer, 'SqueezeAndExcitationBlock': SqueezeAndExcitationBlock,
                                                                'MultiDimensionalAttention': MultiDimensionalAttention,
                                                                'ChannelAttention': ChannelAttention,
                                                                'TemporalAttention': TemporalAttention,
                                                                'FrequencyAttention': FrequencyAttention})

        # extract embeddings
        emb_model = tf.keras.Model(model.input, model.layers[-8].output)
        eval_embs = emb_model.predict([eval_raw, np.zeros((eval_raw.shape[0], num_classes_4train))], batch_size=batch_size)
        train_embs = emb_model.predict([train_raw, np.zeros((train_raw.shape[0], num_classes_4train))], batch_size=batch_size)
        unknown_embs = emb_model.predict([unknown_raw, np.zeros((unknown_raw.shape[0], num_classes_4train))], batch_size=batch_size)
        test_embs = emb_model.predict([test_raw, np.zeros((test_raw.shape[0], num_classes_4train))], batch_size=batch_size)

        # length normalization
        x_train_ln = length_norm(train_embs)
        x_eval_ln = length_norm(eval_embs)
        x_test_ln = length_norm(test_embs)
        x_unknown_ln = length_norm(unknown_embs)

        for j, lab in tqdm(enumerate(np.unique(train_labels))):
            cat = le.inverse_transform([lab])[0]
            kmeans = KMeans(n_clusters=n_subclusters, random_state=0).fit(x_train_ln[source_train*(train_labels == lab)])
            means_source_ln = kmeans.cluster_centers_
            means_target_ln = x_train_ln[~source_train * (train_labels == lab)]

            # compute cosine distances
            eval_cos = np.min(1-np.dot(x_eval_ln[eval_labels == lab], means_target_ln.transpose()),axis=-1, keepdims=True)
            eval_cos = np.minimum(eval_cos,np.min(1-np.dot(x_eval_ln[eval_labels == lab], means_source_ln.transpose()), axis=-1, keepdims=True))
            unknown_cos = np.min(1-np.dot(x_unknown_ln[unknown_labels == lab], means_target_ln.transpose()), axis=-1, keepdims=True)
            unknown_cos = np.minimum(unknown_cos,np.min(1-np.dot(x_unknown_ln[unknown_labels == lab], means_source_ln.transpose()), axis=-1, keepdims=True))
            test_cos = np.min(1-np.dot(x_test_ln[test_labels==lab], means_target_ln.transpose()), axis=-1, keepdims=True)
            test_cos = np.minimum(test_cos, np.min(1-np.dot(x_test_ln[test_labels==lab], means_source_ln.transpose()), axis=-1, keepdims=True))
            train_cos = np.min(1-np.dot(x_train_ln[train_labels==lab], means_target_ln.transpose()), axis=-1, keepdims=True)
            train_cos = np.minimum(train_cos, np.min(1-np.dot(x_train_ln[train_labels==lab], means_source_ln.transpose()), axis=-1, keepdims=True))

            if np.sum(eval_labels == lab) > 0:
                pred_eval[eval_labels == lab, j] += np.min(eval_cos, axis=-1)
                pred_unknown[unknown_labels == lab, j] += np.min(unknown_cos, axis=-1)
            if np.sum(test_labels == lab) > 0:
                pred_test[test_labels == lab, j] += np.min(test_cos, axis=-1)
            pred_train[train_labels == lab, j] += np.min(train_cos, axis=-1)
        print('#######################################################################################################')
        print('DEVELOPMENT SET')
        print('#######################################################################################################')
        aucs = []
        p_aucs = []
        aucs_source = []
        p_aucs_source = []
        aucs_target = []
        p_aucs_target = []
        for j, cat in enumerate(np.unique(eval_ids)):
            y_pred = np.concatenate([pred_eval[eval_labels == le.transform([cat]), le.transform([cat])],
                                     pred_unknown[unknown_labels == le.transform([cat]), le.transform([cat])]],
                                     axis=0)
            y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))),
                                     np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
            p_aucs.append(p_auc)
            print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))

            source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])],
                                         source_unknown[unknown_labels == le.transform([cat])]], axis=0)
            auc = roc_auc_score(y_true[source_all], y_pred[source_all])
            p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
            aucs_source.append(auc)
            p_aucs_source.append(p_auc)
            print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
            auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
            p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
            aucs_target.append(auc)
            p_aucs_target.append(p_auc)
            print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
        print('####################')
        aucs = np.array(aucs)
        p_aucs = np.array(p_aucs)
        for cat in categories_dev:
            mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
            print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
            mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
            print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
        print('####################')
        for cat in categories_dev:
            mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
            mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
            print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
        print('####################')
        mean_auc_source = hmean(aucs_source)
        print('mean AUC for source domain: ' + str(mean_auc_source * 100))
        mean_p_auc_source = hmean(p_aucs_source)
        print('mean pAUC for source domain: ' + str(mean_p_auc_source * 100))
        mean_auc_target = hmean(aucs_target)
        print('mean AUC for target domain: ' + str(mean_auc_target * 100))
        mean_p_auc_target = hmean(p_aucs_target)
        print('mean pAUC for target domain: ' + str(mean_p_auc_target * 100))
        mean_auc = hmean(aucs)
        print('mean AUC: ' + str(mean_auc * 100))
        mean_p_auc = hmean(p_aucs)
        print('mean pAUC: ' + str(mean_p_auc * 100))
        final_results_dev[k_ensemble] = np.array([mean_auc_source, mean_p_auc_source, mean_auc_target, mean_p_auc_target, mean_auc, mean_p_auc])

        # print results for eval set
        print('#######################################################################################################')
        print('EVALUATION SET')
        print('#######################################################################################################')
        aucs = []
        p_aucs = []
        aucs_source = []
        p_aucs_source = []
        aucs_target = []
        p_aucs_target = []
        for j, cat in enumerate(np.unique(test_ids)):
            y_pred = pred_test[test_labels == le.transform([cat]), le.transform([cat])]
            y_true = np.array(pd.read_csv(
                './dcase2023_task2_evaluator-main/ground_truth_data/ground_truth_' + cat.split('_')[0] + '_section_' + cat.split('_')[1] + '_test.csv', header=None).iloc[:, 1] == 1)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
            p_aucs.append(p_auc)
            print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))
            source_all = np.array(pd.read_csv(
                './dcase2023_task2_evaluator-main/ground_truth_domain/ground_truth_' + cat.split('_')[0] + '_section_' + cat.split('_')[1] + '_test.csv', header=None).iloc[:, 1] == 0)
            auc = roc_auc_score(y_true[source_all], y_pred[source_all])
            p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
            aucs_source.append(auc)
            p_aucs_source.append(p_auc)
            print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
            auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
            p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
            aucs_target.append(auc)
            p_aucs_target.append(p_auc)
            print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
        print('####################')
        aucs = np.array(aucs)
        p_aucs = np.array(p_aucs)
        for cat in categories_eval:
            mean_auc = hmean(aucs[np.array([test_id.split('_')[0] for test_id in np.unique(test_ids)]) == cat])
            print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
            mean_p_auc = hmean(p_aucs[np.array([test_id.split('_')[0] for test_id in np.unique(test_ids)]) == cat])
            print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
        print('####################')
        for cat in categories_eval:
            mean_auc = hmean(aucs[np.array([test_id.split('_')[0] for test_id in np.unique(test_ids)]) == cat])
            mean_p_auc = hmean(p_aucs[np.array([test_id.split('_')[0] for test_id in np.unique(test_ids)]) == cat])
            print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
        print('####################')
        mean_auc_source = hmean(aucs_source)
        print('mean AUC for source domain: ' + str(mean_auc_source * 100))
        mean_p_auc_source = hmean(p_aucs_source)
        print('mean pAUC for source domain: ' + str(mean_p_auc_source * 100))
        mean_auc_target = hmean(aucs_target)
        print('mean AUC for target domain: ' + str(mean_auc_target * 100))
        mean_p_auc_target = hmean(p_aucs_target)
        print('mean pAUC for target domain: ' + str(mean_p_auc_target * 100))
        mean_auc = hmean(aucs)
        print('mean AUC: ' + str(mean_auc * 100))
        mean_p_auc = hmean(p_aucs)
        print('mean pAUC: ' + str(mean_p_auc * 100))
        final_results_eval[k_ensemble] = np.array([mean_auc_source, mean_p_auc_source, mean_auc_target, mean_p_auc_target, mean_auc, mean_p_auc])

# create challenge submission files
print('creating submission files')
sub_path = './teams/submission/team_fkie'
if not os.path.exists(sub_path):
    os.makedirs(sub_path)
for j, cat in enumerate(np.unique(test_ids)):
    # anomaly scores
    file_idx = test_labels == le.transform([cat])
    results_an = pd.DataFrame()
    results_an['output1'], results_an['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                    [str(s) for s in pred_test[file_idx, le.transform([cat])]]]
    results_an.to_csv(sub_path + '/anomaly_score_' + cat.split('_')[0] + '_section_' + cat.split('_')[-1] + '_test.csv',
                      encoding='utf-8', index=False, header=False)

    # decision results
    train_scores = pred_train[train_labels == le.transform([cat]), le.transform([cat])]
    threshold = np.percentile(train_scores, q=90)
    decisions = pred_test[file_idx, le.transform([cat])] > threshold
    results_dec = pd.DataFrame()
    results_dec['output1'], results_dec['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                      [str(int(s)) for s in decisions]]
    results_dec.to_csv(sub_path + '/decision_result_' + cat.split('_')[0] + '_section_' + cat.split('_')[-1] + '_test.csv',
                       encoding='utf-8', index=False, header=False)
print('####################')
print('####################')
print('####################')
print('final results for development set')
print(np.round(np.mean(final_results_dev*100, axis=0), 2))
print(np.round(np.std(final_results_dev*100, axis=0), 2))
print('final results for evaluation set')
print(np.round(np.mean(final_results_eval*100, axis=0), 2))
print(np.round(np.std(final_results_eval*100, axis=0), 2))

print('####################')
print('>>>> finished! <<<<<')
print('####################')
