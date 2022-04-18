import streamlit as st

from DSAI_DMV_Utility.DSAI_Utility import all_initialization,css_function
from DSAI_Text_Classification import ClassificationModels
from DSAI_DMV_Layout import ELP_Validation
import traceback
import sys

if __name__ == "__main__":
    
    st.set_page_config(page_title="DMV Vanity Plate Analyzer", layout="wide")
    try:
        css_function("DSAI_Model_Implementation_Sourcecode/DSAI_DMV_Utility/style.css")
        all_initialization()
        ELP_Validation()
        
    except BaseException as e:
        col1, col2, col3 = st.columns([1.5,9,1.5])
        with col2:
            st.write('')
            st.error('In Error block - '+str(e))
            traceback.print_exception(*sys.exc_info())
        
        
        
        
        
        
        
        
        
        
        
    
    # vAR_data = pd.read_csv('/home/jupyter/DSAI_DMV_Text_Analyzer/DSAI_Dataset/train.csv').head(300)
    # vAR_target_columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    # vAR_model_obj = ClassificationModels(vAR_data,vAR_target_columns)
    # vAR_model_obj.display_column()
    # vAR_test_data = None
    # vAR_corpus = vAR_model_obj.data_preprocessing(vAR_test_data)
    # print('Data Preprocessing Completed')
    
############## *****Execute Below Code For NaiveBayes Classification***** ##############

#     vAR_X,vAR_y = vAR_model_obj.bagofwords_vectorization(vAR_corpus,vAR_test_data)
#     print('Vectorization Completed Using Bag of Words')
#     vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test = vAR_model_obj.train_test_split(vAR_X,vAR_y)
#     print('Train & Test Data Splitted Successfully')

#     vAR_model = vAR_model_obj.train_model_naivebayes(vAR_X_train,vAR_y_train)
#     print('Naive Bayes Model Trained successfully')
#     vAR_prediction = vAR_model_obj.test_model(vAR_model,vAR_X_test)
#     print('Naive Bayes Model Tested successfully')
#     accuracy = vAR_model_obj.accuracy_score(vAR_prediction,vAR_y_test)
#     print('Naive Bayes Model Accuracy - ',accuracy)

############## ******************************************************** ###############

############## *****Execute Below Code For Random Forest Classification***** ##############
    
    # vAR_X,vAR_y = vAR_model_obj.bagofwords_vectorization(vAR_corpus,vAR_test_data)
    # print('Vectorization Completed Using Bag of Words')
    # vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test = vAR_model_obj.train_test_split(vAR_X,vAR_y)
    # print('Train & Test Data Splitted Successfully')
    # vAR_model = vAR_model_obj.train_model_random_forest(vAR_X_train,vAR_y_train)
    # print('Random Forest Model Trained successfully')
    # vAR_prediction = vAR_model_obj.test_model(vAR_model,vAR_X_test)
    # print('Random Forest Model Tested successfully')
    # accuracy = vAR_model_obj.accuracy_score(vAR_prediction,vAR_y_test)
    # print('Random Forest Model Accuracy - ',accuracy)

############## ******************************************************** ###############
    
############## *****Execute Below Code For LSTM RNN Deep Learning Model***** ############## 

#     vAR_X,vAR_y = vAR_model_obj.word_embedding_vectorization(vAR_corpus,vAR_test_data)
#     print('Vectorization Completed Using Word Embedding')
#     vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test = vAR_model_obj.train_test_split(vAR_X,vAR_y)
#     print('Train & Test Data Splitted Successfully')
    
#     vAR_model = vAR_model_obj.train_model_lstm(vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test)
#     print('LSTM Model Trained successfully')
#     vAR_prediction = vAR_model_obj.test_model(vAR_model,vAR_X_test)
#     print('LSTM Model Tested successfully')
    
############## ******************************************************** ###############


############## *****Execute Below Code When You want to test the model with custom text data***** ##############
    
    # vAR_test_data = pd.read_csv('/home/jupyter/DSAI_DMV_Text_Analyzer/DSAI_Dataset/test-compress-all-labels.csv')
    # vAR_X_test_data = vAR_test_data.drop(['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
    # print('Xtest length - ',len(vAR_test_data))
    # vAR_corpus = vAR_model_obj.data_preprocessing(vAR_test_data)
    # print('Data Preprocessing Completed')
    # vAR_X,vAR_y = vAR_model_obj.word_embedding_vectorization(vAR_corpus,vAR_test_data)
    # print('Vectorization Completed Using Word Embedding')
    # vAR_prediction = vAR_model_obj.test_model(vAR_model,vAR_X)
    # print('Naive Bayes Model Tested successfully')
    # print('ypred length - ',len(vAR_prediction))
    # vAR_X_test_data['toxic'] = vAR_prediction[:,0]
    # vAR_X_test_data['severe_toxic'] = vAR_prediction[:,1]
    # vAR_X_test_data['obscene'] = vAR_prediction[:,2]
    # vAR_X_test_data['threat'] = vAR_prediction[:,3]
    # vAR_X_test_data['insult'] = vAR_prediction[:,4]
    # vAR_X_test_data['identity_hate'] = vAR_prediction[:,5]
    # vAR_X_test_data.to_csv('/home/jupyter/DSAI_DMV_Text_Analyzer/DSAI_Model_Outcome/DSAI_Model_Outcome.csv')
    # print(vAR_X_test_data.tail(20))
    
############## ******************************************************** ###############
