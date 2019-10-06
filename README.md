# P2 Specification

In this problem, you will implement and evaluate (i) naïve Bayes and (ii) logistic regression. Due dates: First commit 10/11: 30 points, Final commit 10/18: 30 points.
a. Naïve Bayes (20 points)
Implement the naïve Bayes algorithm discussed in class. Discretize continuous values. To do this, partition the range of the feature into k bins (value set through an option). Then replace the feature with a discrete feature that takes value x if the original feature’s value was in bin x. Use m-estimates to smooth your probability estimates. Use logs whenever possible to avoid multiplying too many probabilities together. Your main file should be called nbayes. It should take four options:
1. Option 1 will be the path to the data (see problem 1).
2. Option 2 is a 0/1 option. If 0, use cross validation. If 1, run the algorithm on the full sample.
3. Option 3 (at least 2) is the number of bins for any continuous feature.
4. Option 3 is the value of m for the m-estimate. If this value is negative, use Laplace smoothing. Note that m=0 is maximum likelihood estimation. The value of p in the m-estimate should be fixed to 1/v for a variable with v values.
When your code is run, it should first construct 5 folds using stratified cross validation if this option is provided. To ensure repeatability, set the random seed for the PRNG to 12345. Then it should produce naïve Bayes models on each fold (or the sample according to the option) and report as in part (c).
b. Logistic Regression (20 points)
Implement the logistic regression algorithm described in class. During learning, minimize the negative conditional log likelihood plus a constant (λ) times a penalty term, half of the 2-norm of the weights squared. You can use standard gradient descent for the minimization. Nominal attributes should be encoded as follows: map each value to a number 1…k if there are k values. The main file should be called logreg. It should take three options: the first two options above and a third which is a nonnegative real number that sets the value of the constant λ. The same notes about 5 fold stratified CV from above, etc. apply in this case.
c. Output format
When either algorithm is run on any problem, it must produce output in exactly the following format:
Accuracy: 0.xyz 0.abc
Precision: 0.xyz 0.abc
Recall: 0.xyz 0.abc
Area under ROC: 0.xyz
For all metrics expect Area under ROC, “0.xyz” is the average value of each quantity over five folds. “0.abc” is the standard deviation. For Area under ROC, use the “pooling” method. Here, after running the classifier on each fold, store all the test examples' classes and confidence values in an array. After all folds are done, use this global array to calculate the area. To calculate the area under ROC, first calculate
the TP and FP rates at each confidence level, using the numeric value output by the classifier as the confidence. Each pair of adjacent points is joined by a line, so the area under the curve is the sum over the areas of all trapezoids bounded by the FP rates of the adjacent points, the TP rates of the adjacent points, and the line joining the TP rates.
d. Writeup (20 points)
Prepare a writeup on your experiments. In your writeup, answer the following questions:
(a) What is the accuracy of naïve Bayes with Laplace smoothing and logistic regression (λ =1) on the different learning problems? For each problem, perform a t-test to determine if either is superior with 95% confidence.
(b) Examine the effect of the number of bins when discretizing continuous features in naïve Bayes. Do this by comparing accuracy across several different values of this parameter using volcanoes.
(c) Examine the effect of m in the naïve Bayes m-estimates. Do this by comparing accuracy across m=0, Laplace smoothing, m=10 and 100 on the given problems.
(d) Examine the effect of λ on logistic regression. Do this by comparing accuracy across λ=0, 1, and 10 for the given problems.
Write down any further insights or observations you made while implementing and running the algorithms, such as time and memory requirements, the complexity of the code, etc. Especially interesting insights may be awarded extra points.
Because this is a long assignment, you are expected to make intermediate weekly commits to the repository. Each week, we expect to see substantial progress made in the implementation. Create a separate subdirectory “P2” in your git repository, place your code in it and push to csevcs. Include a short README file containing (i) a summary of the work done for the week and (ii) your experience with the API and documentation, and anything you found confusing. Do NOT commit any other code
