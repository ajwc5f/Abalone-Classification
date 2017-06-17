Abalones, are sea snails (marine gastropod mollusks) that are native to many
areas. Tradiontionally, the age of abalone can be determine by cutting it open,
staining the inside, and counting its rings under a microscope. To determine age
of abalone, one must turn a live abalone upside down and wait for it to expose its
reproductive organ. These are very tedious and time consuming processes. This
project is a classification and prediction problem that aims to predict the age and
sex of abaline based on their physical characteristics. This was done using linear
regression and sex using logistic regression. For the linear regresssion, ordinary
least squares, ridge, bayesian ridge, and elastic net models are used. The results
show that linear regression is an effective way to predict age, whereas logistic
regression to predict sex is not.

The dataset was downloaded from the UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/).
The data file contains the following columns:

    Name		Data Type	Meas.	Description1
    ----		---------	-----	-----------
    Sex		nominal			M, F, and I (infant)
    Length		continuous	mm	Longest shell measurement
    Diameter	continuous	mm	perpendicular to length
    Height		continuous	mm	with meat in shell
    Whole weight	continuous	grams	whole abalone
    Shucked weight	continuous	grams	weight of meat
    Viscera weight	continuous	grams	gut weight (after bleeding)
    Shell weight	continuous	grams	after being dried
    Rings		integer			+1.5 gives the age in years
    
    
