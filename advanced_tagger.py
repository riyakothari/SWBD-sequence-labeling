import os
import sys
import hw2_corpus_tool as h
import pycrfsuite
import time

start_time = time.time()
speaker = ''
y_actual=[]
pred_actual = []

def extract_utterances(traindirectoryPath, testdirectoryPath):
    dialogs_train = h.get_data(traindirectoryPath)
    list_dialogs_train = list(dialogs_train)

    dialogs_test = h.get_data(testdirectoryPath)
    list_dialogs_test = list(dialogs_test)
    return list_dialogs_train, list_dialogs_test

def features(feature_list_of_dialogue, dialog_utterance, speaker_change, first_utterance):
    feature_of_utterance = []
    posTag_list = dialog_utterance.pos
    if posTag_list!=None:
        for postag in posTag_list:
            feature_of_utterance.append('Token_'+postag.token)
            feature_of_utterance.append('Pos_'+postag.pos)
            feature_of_utterance.append('NOT_EMPTY')
    else:
        feature_of_utterance.append('EMPTY')
    if speaker_change:
        feature_of_utterance.append('Speaker_change')
    else:
        feature_of_utterance.append('Not_Speaker_change')
    if first_utterance:
        feature_of_utterance.append('First_utterance')
    else:
        feature_of_utterance.append('Not_First_utterance')
    feature_list_of_dialogue.append(feature_of_utterance)

def get_features(list_dialogs):
    global speaker
    labels_list = []
    features_list = []
    speaker = list_dialogs[0][0].speaker
    for i in range(len(list_dialogs)):
        first_utterance  = True
        feature_list_of_dialog = []
        labels_list_of_dialog = []
        for j in range(len(list_dialogs[i])):
            if speaker == list_dialogs[i][j].speaker:
                speaker_change = False
            else:
                speaker = list_dialogs[i][j].speaker
                speaker_change = True
            labels_list_of_dialog.append(list_dialogs[i][j].act_tag)
            features(feature_list_of_dialog, list_dialogs[i][j], speaker_change, first_utterance)
            first_utterance = False
        features_list.append(feature_list_of_dialog)
        labels_list.append(labels_list_of_dialog)
    return features_list, labels_list


def get_advanced_features(dialog_features_list):
    new_features_of_dialog = []
    n = len(dialog_features_list)
    for i in range(0, n-1):
        new_features_of_utterance = []
        new_features_of_utterance.extend(dialog_features_list[i])
        for j in range(len(dialog_features_list[i+1])):
            if 'Next_'+dialog_features_list[i+1][j] not in new_features_of_utterance:
                new_features_of_utterance.append('Next_'+dialog_features_list[i+1][j])
        new_features_of_dialog.append(new_features_of_utterance)

    new_features_of_dialog.append(dialog_features_list[n-1])

    for i in range(1, n):
        new_features_of_utterance = []
        for j in range(len(dialog_features_list[i-1])): #each word of utterance
            if dialog_features_list[i-1][j]!='First_utterance' and 'Prev_'+dialog_features_list[i-1][j] not in new_features_of_utterance:
                new_features_of_utterance.append('Prev_'+dialog_features_list[i-1][j])
        new_features_of_dialog[i].extend(new_features_of_utterance)

    for i in range(2, n):
        new_features_of_utterance = []
        for j in range(len(dialog_features_list[i-2])): #each word of utterance
            if dialog_features_list[i-2][j]!='First_utterance' and 'Two_Prev_'+dialog_features_list[i-2][j] not in new_features_of_utterance:
                new_features_of_utterance.append('Two_Prev_'+dialog_features_list[i-2][j])
        new_features_of_dialog[i].extend(new_features_of_utterance)

    return new_features_of_dialog


def model_trainer(list_dialogs_train, list_dialogs_test):

    base_train, y_train = get_features(list_dialogs_train)

    base_test, y_test = get_features(list_dialogs_test)

    X_train = []
    for i in range(len(base_train)):
        X_train.append(get_advanced_features(base_train[i]))

    X_test = []
    for i in range(len(base_test)):
        X_test.append(get_advanced_features(base_test[i]))

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        'feature.possible_transitions': True
    })

    trainer.train('advanced_tagger.crfsuite')
    # print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])
    return X_test, y_test

def model_tagger(X_test):
    predictions = []
    tagger = pycrfsuite.Tagger()
    tagger.open('advanced_tagger.crfsuite')

    for test in X_test:
        prediction = tagger.tag(test)
        predictions.append(prediction)
    return predictions

def get_accuracy(y_test, pred):
    global y_actual, pred_actual
    if len(y_test)!=len(pred):
        return Exception
    true_label = 0
    false_label = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            y_actual.append(y_test[i][j])
            pred_actual.append(pred[i][j])
            if len(y_test[i])!=len(pred[i]):
                return Exception
            if pred[i][j]==y_test[i][j]:
                true_label += 1
            else:
                false_label += 1
    accuracy = true_label/(true_label+false_label)
    return accuracy

def finderrors(pred, y, d):
    for i in range(len(pred)):
        if y[i]!=pred[i]:
            if (y[i],pred[i]) not in d:
                d[(y[i],pred[i])]=0
            d[(y[i],pred[i])]+=1
    return d


if __name__ == '__main__':
    traindirectoryPath = sys.argv[1]
    testdirectoryPath = sys.argv[2]
    outputfile = sys.argv[3]
    list_dialogs_train, list_dialogs_test = extract_utterances(traindirectoryPath, testdirectoryPath)
    X_test, y_test = model_trainer(list_dialogs_train, list_dialogs_test)
    predictions = model_tagger(X_test)
    accuracy = get_accuracy(y_test, predictions)
    # print(accuracy)
    with open(outputfile, 'w+') as f:
        for i in range(len(predictions)):
            for j in predictions[i]:
                f.write(j+'\n')
            if i!=len(predictions)-1:
                f.write('\n')

    # print("Time taken: ",time.time()-start_time)
