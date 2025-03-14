start.py file reproduces the best two Kaggle submissions in terms public scores, i.e. SVM and KRR with mismatch kernel. 
The functions necessary to run this file are in library_for_submissions.py. The results are stored in the folder results under the name Yte_mismatch_{model}.csv

If you are interested in the spectrum kernel and/or the Optuna optimization, you can see the work in mismatch_kernel_notebook.ipynb
as well as in spectrum_kernel_notebook.ipynb. 

The substring kernel is implemented in the svm_subtring.py file.
The Weisfeiler-Lehman kernel is implemented in the wl_kernel_svm.py file. 