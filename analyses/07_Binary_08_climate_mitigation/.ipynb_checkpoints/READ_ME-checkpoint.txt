#READ ME

#Folder: 03_Binary-AllText-NewApproach

#Description: Binary classification carried out on "0_all-screen-results_screenExcl-codeIncl.txt", 
where non-random entries were used to train but not test the model during the model_selection.

#Subfolders:
	#0_model_selection_excl:
		#model_selection_excl.py script for fine tuning the hyperparameters
		#do_model_selection_excl.q batch script to run python script on supercomputer
		#model-selection-excl.57587862.out console output file of running model_selection_excl.py
		#model-selection-excl.57587862.err error file of running model_selection_excl.py
		#model_selection_{X}.csv, with X=0,1,2,3,4 is the results of the model_selection using 5 folds,
		it contains combinations of the hyperparameters with their performance scores.
 
	#1_predictions_excl:
		#binary_predictions_excl.py script for running the binary classification on unseen entries
		#do_binary_predictions_excl.q batch script to run python script on supercomputer
		#predictions-classifier-excl.57685074.out console output file of running binary_predictions_excl.py
		#predictions-classifier-excl.57685074.err error file of running binary_predictions_excl.py
		#y_preds_10fold_{X}.npz.npy, with X=0 to 9 results of binary classification with prediction values (0 to 1)
		#unseen_ids.npz.npy contains the list of unseen ids that correspond to the y_preds_10fold_{X}.npz.npy

	#2_cv (cross_validation):
		#cv_bert.py script for running the inner cross validation
		#do_cv_bert.q batch script to run python script on supercomputer
		#cv-bert.58130942.out console output file of running cv_bert.py
		#cv-bert.58130942.err error file of running cv_bert.py
		#cv_results_{X}_{Y}.csv results of cross validation bert
		#outer_cv_bert.py script for running outer cross validation
		#do_outer_cv_bert.q script to run python script on supercomputer
		#outer-bert.58294530.out console output file of running outer_cv_bert.py
		#outer-bert.58294530.err error file of running outer_cv_bert.py
		#y_preds_{X}.npz.npy, with X=0 to 4 results of cross validation binary classification with prediction values (0 to 1)
		
		#cv-svm:
			#cv_svm.py script for running the inner and outer cross validation with a support vector machine (SVM)
			#do_cv_svm.q batch script to run python script on supercomputer
			#cv-svm.58606591.out console output file of running cv_svm.py
			#cv-svm.58606591.err error file of running cv_svm.py
			#svm_inner_{X}.csv with X=0-4 output of running SVM model 
			#svm_outer_{Y}.csv with Y=0-4 output of running SVM model

#Other files:
	#0_unique_references.txt unseen references used for making predictions
	#0_all-screen-results_screenExcl-codeIncl.txt seen references used to train/test the model
	#1_document_relevance_v2.csv predicted relevance of unseen documents
	#3_compile_predictions.py script to read y_pred and compile predictions (two ways, using unseed_ids or re-constructing 
	df as in the predictions script, both should provide the same results). This does not require arge computer resources, i.e. you 
	can run it on your own machine.
	#3_predictions_unseen.png figure of predicted documents
	#4_cv_results compile cv results and plot CV figures as in supplementary material in Callaghan et al.

