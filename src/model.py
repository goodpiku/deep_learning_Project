from typing import List, Dict
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import torch
from itertools import chain


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.MSELoss()
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = None

    def load_model(self, path_to_saved_model: str) -> None:
        import sys
        try:
            self.load_state_dict(
                torch.load(path_to_saved_model, map_location=self.device))
            print(f"Loaded saved model from {path_to_saved_model}")
        except FileNotFoundError:
            print(
                f"\nERROR: Pre-trained model not found at {path_to_saved_model}")
            sys.exit(1)
        except Exception as e:
            print(
                f"\nERROR: Something did not work while loading pre-trained model. Giving up {e}")
            sys.exit(1)

    def run_epochs(self, training_data, validation_data, num_epochs):
        old_score = 0
        self.optimizer = torch.optim.Adam(self.parameters())
        dict_of_result = {}
        for epoch in range(num_epochs):
            print('-' * 60)
            print(f'Running epoch: {epoch + 1}')
            self.run_train(training_data)
            actual_intent_labels, predicted_intent_labels, actual_slot_labels, predicted_slot_labels = self.run_validation(
                validation_data)
            dict_of_intent_result = self.evaluation(actual_labels=actual_intent_labels,
                                                    predicted_labels=predicted_intent_labels)
            dict_of_slot_result = self.evaluation(actual_labels=list(chain.from_iterable(actual_slot_labels)),
                                                  predicted_labels=list(chain.from_iterable(predicted_slot_labels)))
            print(f"Macro F1 intent: {dict_of_intent_result['macro_f1']}, Slot:{dict_of_slot_result['macro_f1']}")
            # set this in run
            slot_macro_score = dict_of_slot_result['macro_f1']
            saved_model_path = f'../benchmark/results/parameters/parameters_{slot_macro_score}.pt'
            if slot_macro_score > old_score:
                torch.save(self.state_dict(), saved_model_path)
                old_score = slot_macro_score

        return dict_of_result

    def run_train(self, training_data):
        print('training started')
        self.train()  # train() is a func of nn.Module , as nn.Module is parent class of Module so we write self.train()
        list_of_loss = []
        for batch in training_data:
            self.optimizer.zero_grad()
            inputs, intent_labels, slot_one_hot_labels = batch['processed_text'], batch['intent_label'], batch[
                'slot_label']
            predicted_probabilities_1, predicted_probabilities_2 = self.__call__(batch)  # call the forward function

            # loss
            intent_labels = intent_labels.long().to(self.device)
            slot_one_hot_labels = slot_one_hot_labels.to(self.device)

            loss1 = self.loss1(predicted_probabilities_1, intent_labels)
            loss2 = self.loss2(predicted_probabilities_2, slot_one_hot_labels)
            # loss = sum(loss1, loss2)
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
        # mean_loss = np.mean(list_of_loss)
        mean_loss = .0
        print(f'mean loss: {mean_loss}')
        return mean_loss

    def run_validation(self, validation_data):  # do this
        print('validation started')
        self.eval()
        list_of_actual_intent_labels = []
        list_of_predicted_intent_labels = []
        list_of_actual_slot_labels = []
        list_of_predicted_slot_labels = []
        # call the forward function
        with torch.no_grad():  # no backpropagation
            for batch in validation_data:
                inputs, intent_labels, slot_one_hot_labels = batch['processed_text'], batch['intent_label'], batch[
                    'slot_label']
                predicted_probabilities_1, predicted_probabilities_2 = self.__call__(batch)

                # intent
                predicted_labels_intent_per_batch = torch.argmax(predicted_probabilities_1,
                                                                 dim=1)  # get predicted labels from predicted probabilities
                list_of_predicted_intent_labels.extend(predicted_labels_intent_per_batch.tolist())
                list_of_actual_intent_labels.extend(intent_labels.tolist())

                # slots
                predicted_slot_labels_per_batch = torch.argmax(predicted_probabilities_2,
                                                               dim=2)
                list_of_predicted_slot_labels.extend(predicted_slot_labels_per_batch.tolist())
                slot_labels = torch.argmax(slot_one_hot_labels, dim=2)
                list_of_actual_slot_labels.extend(slot_labels.tolist())
                """
                TODO: list_of_predicted_slot_labels
                """
        return list_of_actual_intent_labels, list_of_predicted_intent_labels, list_of_actual_slot_labels, list_of_predicted_slot_labels

    def run_test(self, test_data):  # do this
        print('test started')
        self.eval()
        list_of_text = []
        list_of_predicted_intent_labels = []
        list_of_predicted_slot_labels = []
        # call the forward function
        with torch.no_grad():  # no backpropagation
            for batch in test_data:
                predicted_probabilities_1, predicted_probabilities_2 = self.__call__(batch)
                # intent
                predicted_labels_per_batch = torch.argmax(predicted_probabilities_1,
                                                          dim=1)  # get predicted labels from predicted probabilities
                list_of_predicted_intent_labels.extend(predicted_labels_per_batch.tolist())

                # slots
                predicted_slot_labels_per_batch = torch.argmax(predicted_probabilities_2,
                                                               dim=2)
                list_of_predicted_slot_labels.extend(predicted_slot_labels_per_batch.tolist())

                list_of_text.extend(batch['text'])
        return list_of_text, list_of_predicted_intent_labels, list_of_predicted_slot_labels

    def evaluation(self, actual_labels, predicted_labels):  # test this
        micro_precision = precision_score(actual_labels, predicted_labels, average='micro', zero_division='warn')
        micro_recall = recall_score(actual_labels, predicted_labels, average='micro', zero_division='warn')
        micro_f1_score = f1_score(actual_labels, predicted_labels, average='micro', zero_division='warn')

        macro_precision = precision_score(actual_labels, predicted_labels, average='macro', zero_division='warn')
        macro_recall = recall_score(actual_labels, predicted_labels, average='macro', zero_division='warn')
        macro_f1_score = f1_score(actual_labels, predicted_labels, average='macro', zero_division='warn')
        dict_of_evaluation = {'micro_p': micro_precision, 'macro_p': macro_precision, 'micro_r': micro_recall,
                              'macro_r': macro_recall, 'micro_f1': micro_f1_score, 'macro_f1': macro_f1_score}

        return dict_of_evaluation

    # def evaluation(self, actual_labels: List, predicted_labels: List, list_of_labels: List) -> Dict:
    #     list_of_labels=
    #     dict_of_results = self.find_dict_of_result(actual_labels=actual_labels, predicted_labels=predicted_labels,
    #                                                list_of_labels=list_of_labels)
    #     macro_scores_dict = self.macro_scores(dict_of_results=dict_of_results)
    #     return macro_scores_dict
    #
    # def find_dict_of_result(self, actual_labels: List, predicted_labels: List, list_of_labels: List) -> Dict:
    #     # return accuracy, precision, recall, f1_score
    #     dict_of_results = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    #     for label in list_of_labels:
    #         tp = 0
    #         fp = 0
    #         fn = 0
    #         tn = 0
    #         list_of_label_indexes_in_predicted_labels = [ind for ind, val in enumerate(predicted_labels) if
    #                                                      val == label]
    #         list_of_label_indexes_in_actual_labels = [ind for ind, val in enumerate(actual_labels) if val == label]
    #         for index in list_of_label_indexes_in_predicted_labels:
    #             if index in list_of_label_indexes_in_actual_labels:
    #                 tp += 1
    #             else:
    #                 fp += 1
    #         for index in list_of_label_indexes_in_actual_labels:
    #             if index not in list_of_label_indexes_in_predicted_labels:
    #                 fn += 1
    #             else:
    #                 tn += 1
    #         dict_of_results['TP'].append(tp)
    #         dict_of_results['FP'].append(fp)
    #         dict_of_results['FN'].append(fn)
    #         dict_of_results['TN'].append(tn)
    #         return dict_of_results
    #
    # def macro_scores(self, dict_of_results: Dict) -> Dict:
    #     list_of_precision = []
    #     list_of_recall = []
    #     list_of_f1_score = []
    #     for ind in range(len(dict_of_results['TP'])):
    #         tp = dict_of_results['TP'][ind]
    #         fp = dict_of_results['FP'][ind]
    #         fn = dict_of_results['FN'][ind]
    #         if tp == 0:  # problem of zero nominator
    #             list_of_precision.append(0.0)
    #             list_of_recall.append(0.0)
    #             list_of_f1_score.append(0.0)
    #         else:
    #             precision = tp / (tp + fp)
    #             recall = tp / (tp + fn)
    #             list_of_precision.append(precision)
    #             list_of_recall.append(recall)
    #             list_of_f1_score.append((2 * precision * recall) / (precision + recall))
    #     macro_precision = sum(list_of_precision) / len(list_of_precision)
    #     macro_recall = sum(list_of_recall) / len(list_of_recall)
    #     macro_F1_score = sum(list_of_f1_score) / len(list_of_f1_score)
    #
    #     return {'macroPrec': macro_precision, 'macroRec': macro_recall, 'macroF1': macro_F1_score}
