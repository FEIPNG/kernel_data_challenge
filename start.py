from library_for_submissions import predict_mismatch_krr, predict_mismatch_svm, save_to_csv

prediction_0 = predict_mismatch_svm("./kernel_data_challenge/data/Xtr0.csv", "./kernel_data_challenge/data/Ytr0.csv", "./kernel_data_challenge/data/Xte0.csv", 8,1, 0.031048368555776834)
prediction_1 = predict_mismatch_svm("./kernel_data_challenge/data/Xtr1.csv", "./kernel_data_challenge/data/Ytr1.csv", "./kernel_data_challenge/data/Xte1.csv", 9, 1, 77.80986401052525)
prediction_2 = predict_mismatch_svm("./kernel_data_challenge/data/Xtr2.csv", "./kernel_data_challenge/data/Ytr2.csv", "./kernel_data_challenge/data/Xte2.csv", 9, 1, 0.009473888943332944)

save_to_csv(prediction_0, prediction_1, prediction_2, suffixe = "mismatch_svm")


prediction_0 = predict_mismatch_krr("./kernel_data_challenge/data/Xtr0.csv", "./kernel_data_challenge/data/Ytr0.csv", "./kernel_data_challenge/data/Xte0.csv", 8, 1, 612.1046949249738)
prediction_1 = predict_mismatch_krr("./kernel_data_challenge/data/Xtr1.csv", "./kernel_data_challenge/data/Ytr1.csv", "./kernel_data_challenge/data/Xte1.csv", 9, 1, 813.5712265256881)
prediction_2 = predict_mismatch_krr("./kernel_data_challenge/data/Xtr2.csv", "./kernel_data_challenge/data/Ytr2.csv", "./kernel_data_challenge/data/Xte2.csv", 8, 3, 732.6063350813271)

save_to_csv(prediction_0, prediction_1, prediction_2, suffixe = "mismatch_KRR")