#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()

wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df["WineType"] = [wine.target_names[typ] for typ in wine.target]

wine_df.head(10)


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(wine.data, wine.target, train_size=0.8, random_state=123)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[8]:


from sklearn.ensemble import RandomForestClassifier 

rf_classif = RandomForestClassifier()

rf_classif.fit(X_train, Y_train)


# In[9]:


Y_test_preds = rf_classif.predict(X_test)
Y_test_preds[:5]


# In[10]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 

print("Test Accuracy : {:.2f}". format(accuracy_score(Y_test, Y_test_preds)))

print("\nConfusion Matrix :")
print(confusion_matrix(Y_test, Y_test_preds))

print("\nClassification Report :")
print(classification_report(Y_test, Y_test_preds))


# In[11]:


from joblib import dump, load 

dump(rf_classif, "RandomForest.model" )


# In[12]:


rf_classif_2 = load("RandomForest.model")

rf_classif_2


# In[13]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 

Y_test_preds = rf_classif_2.predict(X_test)

print("Test Accuracy : {:.2f}". format(accuracy_score(Y_test, Y_test_preds)))

print("\nConfusion Matrix :")
print(confusion_matrix(Y_test, Y_test_preds))

print("\nClassification Report :")
print(classification_report(Y_test, Y_test_preds))


# In[16]:


from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(X_train, mode="classification",
                                             class_names = wine.target_names,
                                             feature_names = wine.feature_names,
                                             )

explainer


# In[17]:


explanation = explainer.explain_instance(X_test[0], rf_classif.predict_proba,
                                        num_features=len(wine.feature_names),
                                        top_labels=3
                                        )
explanation.show_in_notebook()


# In[18]:


fig = explanation.as_pyplot_figure(label=2)


# In[ ]:




