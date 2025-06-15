from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=.30,random_state=42)

xtest

#Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
rf = RandomForestClassifier(n_estimators=100)
rf.fit(xtrain,ytrain)
pred4 = rf.predict(xtest)
pred4

rf.score(xtest,ytest)
print("Accuracy:",metrics.accuracy_score(ytest, pred4))
cnf_matrix = metrics.confusion_matrix(ytest, pred4)
cnf_matrix

cnf_matrix = metrics.confusion_matrix(ytest, pred4)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("F1:",metrics.f1_score(ytest, pred4))
print("Accuracy:",metrics.accuracy_score(ytest, pred4))
print("Precision:",metrics.precision_score(ytest, pred4))
print("Recall:",metrics.recall_score(ytest, pred4))

# Add necessary library to tune hyperparameter using gridsearchcv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report


rf = RandomForestClassifier(random_state=42)

# Define hyperparameters grid
param_grid = {
    'n_estimators': [10, 20, 25, 50, 75, 100, 125, 150, 175, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=5, #Set cv to an integer
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)

# Fit GridSearchCV
grid_search.fit(xtrain, ytrain)

# Show best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Score (Cross-Validation Accuracy):", grid_search.best_score_)

# Evaluate on test data
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(xtest)
print("\nClassification Report:\n", classification_report(ytest, y_pred))

cnf_matrix = metrics.confusion_matrix(ytest, y_pred)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Greens" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# Add adaboost classifier 
from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier(n_estimators=50)
ab.fit(xtrain,ytrain)
pred8 = ab.predict(xtest)
ab.score(xtest,ytest)

cnf_matrix = metrics.confusion_matrix(ytest, pred8)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("F1:",metrics.f1_score(ytest, pred8))
print("Accuracy:",metrics.accuracy_score(ytest, pred8))
print("Precision:",metrics.precision_score(ytest, pred8))
print("Recall:",metrics.recall_score(ytest, pred8))



# add gridsearchcv to tune the model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report


base_estimator = DecisionTreeClassifier(random_state=42)

# Define AdaBoost model
adb = AdaBoostClassifier(estimator=base_estimator, random_state=42)

# Parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3]  # tuning the depth of the weak learner
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=adb,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1,
                           verbose=2)

# Fit the grid search to the data
grid_search.fit(xtrain, ytrain)

# Output the best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Test set evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(xtest)
print("\nClassification Report on Test Data:\n", classification_report(ytest, y_pred))


# Generate confusion matrix for gridsearchcv
cnf_matrix = metrics.confusion_matrix(ytest, y_pred)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Greens" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')