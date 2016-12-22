import autoML as aML

# nose2 returns the number of tests executed.  That's the number of
# tests in this module.  To make this work out, create an array
# of command line parameters for each test.

##################### Simplest Classification Test  #############

def test_classification_0():

    # try classification on the simplest set of inputs.  Setting the -x
    # flag says we're doing a regression test.  We specifiy the time (-t)
    # to be long enough that all of the classification methods finish.
    # We set the number of iterations (-n) so that we don't have to
    # wait forever.
    plainClassificationTestArgs = ["-x", "True", "-s", "none", "-t", "14400", "-n", "10"]

    # Call the routines.  They return strings that we test to correctness
    (experts, confusionMatrix, classificationReport) = aML.test_autoML(commandLineArgs=plainClassificationTestArgs)

    expected_experts_0 = """
        Number of models: 11
        Models: ['GradientBoost: 0.803099', 'ExtraTrees: 0.799453', 'LDA: 0.799453', 'LogisticRegression: 0.798541', 'RandomForest: 0.795807', 'DecisionTree: 0.793072', 'SGD: 0.768459', 'QDA: 0.763902', 'GaussianNB: 0.734731', 'KNeighbors: 0.711030', 'AdaBoost: 0.608022']
        """
    expected_CM_0 = """[[650   0   0   0  11]
 [  0   0   0   1   2]
 [  2   0  13   6   1]
 [ 45   1   0  69  21]
 [122   0   1   3 149]]"""

    expected_classification_report_0 = """             precision    recall  f1-score   support

          0       0.79      0.98      0.88       661
          1       0.00      0.00      0.00         3
          2       0.93      0.59      0.72        22
          3       0.87      0.51      0.64       136
          4       0.81      0.54      0.65       275

avg / total       0.81      0.80      0.79      1097
"""
    print(experts)
    print(expected_experts_0)
    assert experts == expected_experts_0

    print(confusionMatrix)
    print(expected_CM_0)
    assert confusionMatrix == expected_CM_0

    print(classificationReport)
    print(expected_classification_report_0)
    assert classificationReport == expected_classification_report_0

#########################  End of Simplest Classification test  #################

# actually run the test.  This is how we get
# nose2 to actually record the number of tests and the duration of each one.
test_classification_0()

