import torch
from torch import nn, optim
from torch.autograd import Variable
import random
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

from dataset import MotionDataset
import dataset
from network_twostream import Network_Fusion
from network_onestream import Network_OneStream

EPOCHS = 80
BATCH_SIZE = 57
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-6
N_CLASSES = 4
BATCH_SIZE_TEST = 9
Ave_Acc = 0
Min_training_loss = 100
dataset.Model_Flag = "3DRJDP-CNN"  # select training model: 2s-CNN, 3DJP-CNN, 3DRJDP-CNN


# use_gpu = torch.cuda.is_available()

def main():
    Ave_Acc = 0
    dir = "data/"
    for fold in range(5):
        Min_training_loss = 100
        train_dir = list()
        print()
        print("===================Training and testing %d fold===================\n" % int(fold + 1))
        for train in range(5):
            if train == fold:
                train_dir.append(dir + str(train + 1) + str(train + 1) + "fold/")
                continue
            train_dir.append(dir + str(train + 1) + "fold/")
        test_dir = [dir + str(fold + 1) + "fold/"]

        # store training loss and testing loss
        training_list = list()
        testing_list = list()

        train_dataset = MotionDataset(data_dir=train_dir)
        test_dataset = MotionDataset(data_dir=test_dir)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE_TEST)

        if dataset.Model_Flag == "2s-CNN":
            net = Network_Fusion(3, 3, 64, 64, N_CLASSES)
        else:
            net = Network_OneStream(3, 64, N_CLASSES)
        # if(use_gpu):
        #     net = net.cuda()
        # print('parameters: ', sum(param.numel() for param in net.parameters()))

        ACE = nn.CrossEntropyLoss()
        opt = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        # opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in range(1, EPOCHS + 1):
            print('[Epoch %d]' % epoch)
            train_loss = 0
            train_correct, train_total = 0, 0
            if dataset.Model_Flag == "2s-CNN":
                for inputs, inputs2, labels in train_loader:
                    inputs, inputs2, labels = Variable(inputs), Variable(inputs2), Variable(labels)
                    opt.zero_grad()
                    preds = net(inputs, inputs2)
                    loss = ACE(preds, labels)
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
                    train_correct += (preds.argmax(dim=1) == labels).sum().item()
                    train_total += len(preds)

            else:
                for inputs, labels in train_loader:
                    inputs, labels = Variable(inputs), Variable(labels)
                    opt.zero_grad()
                    preds = net(inputs)
                    loss = ACE(preds, labels)
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
                    train_correct += (preds.argmax(dim=1) == labels).sum().item()
                    train_total += len(preds)

            train_temp_loss = train_loss / len(train_loader)
            print('train-acc : %.4f%% train-loss : %.5f' % (100 * train_correct / train_total, train_temp_loss))
            training_list.append(train_loss / len(train_loader))
            test_loss = 0
            test_correct, test_total = 0, 0
            classes = ('Health', 'Joint Problem', 'Muscle Weakness', 'Neurological Defect')
            class_correct = list(0. for i in range(N_CLASSES))
            class_total = list(0. for i in range(N_CLASSES))
            if dataset.Model_Flag == "2s-CNN":
                for inputs, inputs2, labels in test_loader:
                    with torch.no_grad():
                        inputs, inputs2, labels = Variable(inputs), Variable(inputs2), Variable(labels)
                        preds = net(inputs, inputs2)
                        print("confusion matrix:")
                        print(confusion_matrix(preds.argmax(dim=1), labels))

                        test_loss += ACE(preds, labels).item()

                        test_correct += (preds.argmax(dim=1) == labels).sum().item()
                        test_total += len(preds)

                        _, predicted = torch.max(preds, 1)
                        c = (predicted == labels).squeeze()
                        for i in range(BATCH_SIZE_TEST):
                            label = labels[i]
                            class_correct[label] += c[i].item()
                            class_total[label] += 1
            else:
                for inputs, labels in test_loader:
                    with torch.no_grad():
                        inputs, labels = Variable(inputs), Variable(labels)
                        preds = net(inputs)
                        print("confusion matrix:")
                        print(confusion_matrix(preds.argmax(dim=1), labels))

                        test_loss += ACE(preds, labels).item()

                        test_correct += (preds.argmax(dim=1) == labels).sum().item()
                        test_total += len(preds)

                        _, predicted = torch.max(preds, 1)
                        c = (predicted == labels).squeeze()
                        for i in range(BATCH_SIZE_TEST):
                            label = labels[i]
                            class_correct[label] += c[i].item()
                            class_total[label] += 1

            for i in range(N_CLASSES):
                print('%s : %.2f %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))

            acc = 100 * test_correct / test_total

            print('test-acc : %.2f %% test-loss : %.4f' % (acc, test_loss / len(test_loader)))
            testing_list.append(test_loss / len(test_loader))

            for i in range(len(training_list)):
                print('%.5f' % training_list[i], end=', ')
            print()
            for i in range(len(testing_list)):
                print('%.5f' % testing_list[i], end=', ')
            print()

            if train_temp_loss <= Min_training_loss:
                Min_training_loss = train_temp_loss
                Ave_ACC = acc

        Ave_Acc = Ave_Acc + Ave_ACC
        print('ave_acc : %.2f' % (Ave_Acc / (fold + 1)))


if __name__ == '__main__':
    main()
