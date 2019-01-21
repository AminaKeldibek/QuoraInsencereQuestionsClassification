# QuoraInsencereQuestionsClassification


Model 1: BasicRNNCell encoder of sequence and softmax decoder
| Train  | Dev  | Test |
| ------ | -----| -----|
| 0.88   | 0.51 | 0.51 |  -> uniform distribution of classes in train batches
| 0.75   | 0.6  | 0.6  |  -> P(C0) = 0.8, P(C1) = 0.2; best checkpoint: ckpt-8501
| 0.78   | 0.66 | 0.66 |  -> P(C0) = 0.8, P(C1) = 0.2, GRUCell; best checkpoint: ckpt-5001
| 0.79   | 0.66 | 0.66 |  -> P(C0) = 0.8, P(C1) = 0.2, lSTMCell; best checkpoint: ckpt-5001
| 0.69   | 0.67 | 0.67 |  -> P(C0) = 0.9, P(C1) = 0.1, GRUCell with Relu; best checkpoint: ckpt-8501
