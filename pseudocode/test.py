def perf_measure(test_data, pred_data):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(test_data)):
        if test_data[i] == 1.0 and pred_data[i] == 1.0:
            TP += 1
        if test_data[i] == 0.0 and pred_data[i] == 1.0:
            FP += 1
        if test_data[i] == 0.0 and pred_data[i] == 0.0:
            TN += 1
        if test_data[i] == 1.0 and pred_data[i] == 0.0:
            FN += 1
    print('accuracy is:', (TP+TN)/(TP+FP+FN+TN))
    print('precision is:',(TP/(TP+FP)))
    print('recall is:',TP/(TP+FN))
    print('F-measure:',2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN))))
