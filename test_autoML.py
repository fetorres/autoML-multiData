import autoML as aML

# nose2 returns the number of tests executed.  That's the number of
# tests in this module.  To make this work out, create an array
# of command line parameters for each test.

# An odd feature of nose2 is that it discovers the definitions of
# routines whose names begin with "test" and runs them.  We don't have
# to run them.  The tests must be independent because nose2 will
# execute them in any order.

##################### Simplest Classification Test  #############

def test_classification_0():

    # try classification on the simplest set of inputs.  Setting the -x
    # flag says we're doing a regression test.  We specifiy the time (-t)
    # to be long enough that all of the classification methods finish.
    # We set the number of iterations (-n) so that we don't have to
    # wait forever.
    plainClassificationTestArgs = ["-x", "True", "--model_type", "classification", "-s", "none", "-t", "14400", "-n", "10"]

    # Call the routines.  They return strings that we test for correctness
    classificationResults =aML.test_autoML(commandLineArgs=plainClassificationTestArgs)
    experts = classificationResults['experts']
    confusionMatrix = classificationResults['confusionMatrix']
    classificationReport = classificationResults['classificationReport']

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

#########################  Simplest test of Regression  #########################

# It's unfortunate that "regression" has two very different meanings here.
# In software methology, a "regression test" just tries to guarantee that
# each method produces the same results as it did in the past.  It's used
# to see if changes had unexpected consequences.  In machine learning,
# "regression" is an operation that attempts to derive a curve from
# some data.  The curve can be used to predict the values associated with
# new data points.  This test is a regression test of regression methods.

def test_regression_0():

    # try regression on the simplest set of inputs.  Setting the -x
    # flag says we're doing a regression test.  We specifiy the time (-t)
    # to be long enough that all of the regression methods finish.
    # We set the number of iterations (-n) so that we don't have to
    # wait forever.
    plainRegressionTestArgs = ["-x", "True", "--model_type", "regression", "-s", "none", "-t", "14400", "-n", "10"]

    # Call the routines.  They return strings that we test for correctness
    regressionResults = aML.test_autoML(commandLineArgs=plainRegressionTestArgs)

    experts = regressionResults['experts']
    regressionReport = regressionResults['regressionReport']
    
    expected_experts_1 = """
        Number of models: 11
        Models: ['RandomForest: 0.454392', 'ExtraTrees: 0.442008', 'GradientBoost: 0.437175', 'BayesianRidge: 0.419504', 'Ridge: 0.419173', 'DecisionTree: 0.413304', 'AdaBoost: 0.397049', 'ElasticNet: 0.319148', 'KNeighbors: 0.285064', 'Lasso: -0.002674', 'SGD: -8999.723570']
        """
    expected_regression_report = """(5, 0.45269091107397952)"""

    print(experts)
    print(expected_experts_1)
    assert experts == expected_experts_1

    print(str(regressionReport))
    print(expected_regression_report)
    assert str(regressionReport) == expected_regression_report


