FUNCTION perf_measure(test_data, pred_data):
    TP, FP, TN, FN <- 0, 0, 0, 0
    for i in range(len(test_data)):
        IF test_data[i] = 1.0 AND pred_data[i] = 1.0:
            TP += 1
        ENDIF
        IF test_data[i] = 0.0 AND pred_data[i] = 1.0:
            FP += 1
        ENDIF
        IF test_data[i] = 0.0 AND pred_data[i] = 0.0:
            TN += 1
        ENDIF
        IF test_data[i] = 1.0 AND pred_data[i] = 0.0:
            FN += 1
        ENDIF
    ENDFOR
    OUTPUT 'accuracy is:', (TP+TN)/(TP+FP+FN+TN)
    OUTPUT 'precision is:',(TP/(TP+FP))
    OUTPUT 'recall is:',TP/(TP+FN)
    OUTPUT 'F-measure:',2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN)))
