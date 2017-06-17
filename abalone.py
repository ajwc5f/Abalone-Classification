import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

"""Predicts the abalone age with linear regression, or the abalone sex with logistic regression"""
class Abalone:

    """Extracts the data from file and split into training and testing subsets. Ordinate will br sex for logistic and rings for linear"""
    def readData(self, ordinate='sex'):
        data = pd.read_csv('abalone.data')
        columns = [column.lower() for column in data.columns.values]

        #Get the index of ordinate in the data
        index = [i for i, column in enumerate(columns) if column == ordinate.lower()]
        index = index[0]

        #For the sex column, M = 0, F = 1, and I = -1
        sex = data.iloc[:,0].as_matrix()
        m, f, i = sex == 'M', sex == 'F', sex == 'I'
        sex[m], sex[f], sex[i] = 0., 1., -1.
        sexVector = sex.reshape(len(sex), 1)

        #For the rings column, add 1.5 to get age
        age = (data.iloc[:,8] + 1.5).as_matrix()
        ageVector = age.reshape(len(age), 1)

        #Append the sex and age columns back into the data matrix
        dataFeatures = np.hstack([sexVector, data.iloc[:, 1:8].as_matrix(), ageVector])

        #Get ordinate from index variable
        print('Extracting {} for the y value.'.format(ordinate))
        y = dataFeatures[:,index]
        X = np.delete(dataFeatures, index, axis=1)

        #Shuffle the data
        X, y = shuffle(X, y)
        
        #Split the data into the training and testing subsets
        trainPercent = 0.8
        split = int(round(len(X) * trainPercent))
        xTrain, xTest = X[:split, :], X[split:, :]
        yTrain, yTest = y[:split], y[split:]

        return ({'xTrain': xTrain, 'xTest': xTest, 'yTrain': yTrain, 'yTest': yTest})

    """Defines the logistic function for use in logistic regression"""
    def logistic(self, z):
        return 1 / (1 + np.exp(-z))

    """Computes the classification error for a set of results and given class labels"""
    def classificationError(self, results, classLabels):
        n = results.size
        numErrors = 0.
        
        for i in xrange(n):
            if (results[i] >= 0.5 and classLabels[i]==0) or (results[i] < 0.5 and classLabels[i]==1):
                numErrors += 1

        return numErrors / n

    """Perform logistic regression to classify if abalone is adult male or female"""
    def doLogistic(self, doPCA=True):
        print('-----Starting Logistic Regression for sex prediction.-----\n')
        
        #Reading in the data to training and testing subsets
        dat = self.readData(ordinate='sex')
        xTrain, xTest = dat['xTrain'], dat['xTest']
        yTrain, yTest = dat['yTrain'], dat['yTest']

        print("Removing infants from the data.")
        #Remove infants from training subsets
        keep = np.where(yTest != -1)[0]
        yTrain = yTrain[keep].astype(int)
        xTrain = xTrain[keep,:]
        
        #Remove infants from testing subsets
        keep = np.where(yTrain != -1)[0]
        yTest = yTest[keep].astype(int)
        xTest = xTest[keep,:]

        #Use PCA to reduce dim to 3 so I can easily visualize
        if doPCA:
            print("Using PCA to reduce dimensionality to 3.")
            pca = PCA(n_components=3)
            xTrain = pca.fit_transform(xTrain)
            xTest = pca.fit_transform(xTest)

        print('Training the logistic regression model.')
        #Train the logistic regression model
        logReg = linear_model.LogisticRegression()
        logReg.fit(xTrain, yTrain)
        
        print('Performing predictions based on the trained model.')
        #Do classification using the trained model
        results = logReg.predict(xTest)

        error = self.classificationError(results, yTest)
        print('\nClassification Error: {:4.2f}%'.format(error*100))
        
        print('\n-----Logistic Regression for sex prediction completed.-----')

    """Preform Linear regression to predict the age of abalone"""
    """Models can be OLS = 'ols', Ridge = 'ridge', BayseianRidge = 'bayesian', or ElasticNet = 'elastic'"""
    def doLinear(self, model='elastic', doPCA=True):
        print('-----Starting Linear Regression for age prediction.-----\n')
         
        #Set the model based on model parameter
        model = model.lower()
        if 'ols' in model:
            linRegModel = linear_model.LinearRegression()
            print 'Using Ordinary Least Squares regression model.'
        elif 'ridge' in model:
            linRegModel = linear_model.Ridge(alpha=0.3)
            print 'Using Ridge regression model.'
        elif 'bayesian' in model:
            linRegModel = linear_model.BayesianRidge()
            print 'Using Bayesian Ridge regression model.'
        elif ('elastic' in model):
            linRegModel = linear_model.ElasticNet(alpha=0.3, l1_ratio=0.1)
            print 'Using Elastic Net regression model.'
        else:
            raise NameError("ERROR: Model {} is not supported".format(model))

        #Read in the data to the training and testing subsets
        dat = self.readData(ordinate='rings')
        xTrain, xTest = dat['xTrain'], dat['xTest']
        yTrain, yTest = dat['yTrain'], dat['yTest']
        
        #Use PCA to reduce dim to a number of components that keeps the explained variance ratio above 90%
        if doPCA:
            print('Using PCA to reduce the dimensionality of the data...')
            pca = PCA()
            initPCA = pca.fit(xTrain)
            
            #Find variance ratio and be sure it stays above 0.9
            varianceRatio = initPCA.explained_variance_ratio_
            vr, numComponents = 0, 0
            while vr <= 0.9:
                vr += varianceRatio[numComponents]
                numComponents += 1
                
            print('Reducing to dimension of {}.'.format(numComponents))
            print ('{} principle components gives {:2.1f}% contribution.'.format(numComponents, vr*100))
            
            # Use PCA with that many components to reduce dimensionality.
            pca = PCA(n_components=numComponents)
            xTrain = pca.fit_transform(xTrain) # xTrain reduced.
            xTest = pca.fit_transform(xTest)

        # Train the linear regression model.
        linRegModel.fit(xTrain, yTrain)
        results = linRegModel.predict(xTest)
        
        #Compute the errors
        diff = yTest - results
        error = np.dot(diff, diff) / len(diff)
        print('\nClassification error: {:4.2f}%'.format(error))

        print('\nPlotting the results.')
        #Plot the results
        plt.scatter(results, yTest,  color='black')
        plt.plot(results, results, color='red', linewidth=2)
        plt.ylabel('Actual Age')
        plt.xlabel('Predicted Age')

        plt.show()
        print('-----Linear Regression for age prediction completed.-----\n')