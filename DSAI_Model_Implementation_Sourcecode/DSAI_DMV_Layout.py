import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from DSAI_DMV_Pattern_Denial import pattern_denial
from DSAI_Text_Classification import ClassificationModels
# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel

def ELP_Validation():
    col1, col2, col3, col4 = st.columns([1,4,4,1])
    vAR_model_result = None
    with col2:
        st.write('')
        st.write('')
        st.subheader('ELP Configuration')
    with col3:
        vAR_input_text = st.text_input('Enter input','').upper()
        
    
        
    if len(vAR_input_text)>0:
        vAR_pattern_result = pattern_denial(vAR_input_text)

        if  vAR_pattern_result:
            st.write('')
            col1, col2, col3 = st.columns([1,8,1])
            with col2:
                st.info('ELP Successfully processed for 1st Level')
            col1, col2, col3, col4 = st.columns([1,4,4,1])
            with col2:
                st.write('')
                st.write('')
                st.subheader('Select Model')
            with col3:
                vAR_model = st.selectbox('',('Select Model','BERT','LSTM-RNN'))
            if vAR_model == 'LSTM-RNN':
                vAR_model_result = lstm_model_result(vAR_input_text) 
            elif vAR_model == 'BERT':
                col1, col2, col3 = st.columns([1,8,1])
                with col2:
                    st.info('Development in-progress')
                # vAR_model_result = bert_model_result(vAR_input_text)
            elif vAR_model=='Select Model':
                st.write('')
                col1, col2, col3 = st.columns([1,8,1])
                with col2:
                    st.warning('Select model for 2nd level check')
            if vAR_model_result:
                col1, col2, col3 = st.columns([1,8,1])
                with col2:
                    st.info('ELP Successfully processed for 2nd Level')
                    st.success('ELP Configuration Approved Successfully')
            elif vAR_model_result is not None:
                col1, col2, col3 = st.columns([1,8,1])
                with col2:
                    st.error('ELP Configuration Failed to Meet the DMV Requirements at 2nd level')
        else:
            col1, col2, col3 = st.columns([1,8,1])
            with col2:
                st.error('ELP Configuration Failed to Meet the DMV Requirements at 1st level')
    else:
        col1, col2, col3 = st.columns([1,8,1])
        with col2:
            st.warning("Please enter input details")
    
    col1, col2, col3 = st.columns([1,8,1])
    with col2:
        vAR_recommend = st.button('ELP Configuration Recommendations')
        if vAR_recommend:
            st.info('Development in-progress')
    
    

    
def lstm_model_result(vAR_input_text):
    # Input Data Preprocessing
    vAR_data = pd.DataFrame()
    vAR_target_columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    vAR_model_obj = ClassificationModels(vAR_data,vAR_target_columns)
    vAR_test_data = pd.DataFrame([vAR_input_text],columns=['comment_text'])
    vAR_test_data['toxic'] = None
    vAR_test_data['severe_toxic'] = None
    vAR_test_data['obscene'] = None
    vAR_test_data['threat'] = None
    vAR_test_data['insult'] = None
    vAR_test_data['identity_hate'] = None
    print('Xtest length - ',len(vAR_test_data))
    vAR_corpus = vAR_model_obj.data_preprocessing(vAR_test_data)
    print('Data Preprocessing Completed')
    vAR_X,vAR_y = vAR_model_obj.word_embedding_vectorization(vAR_corpus,vAR_test_data)
    print('Vectorization Completed Using Word Embedding')
    
    vAR_load_model = tf.keras.models.load_model('/home/jupyter/DSAI_DMV_Text_Analyzer/DSAI_Model_Implementation_Sourcecode/LSTM_RNN_Model')

    vAR_model_result = vAR_load_model.predict(vAR_X)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = np.sum(vAR_model_result)
    col1, col2, col3 = st.columns([1,8,1])
    with col2:
        st.write(vAR_result_data)
        st.write('Sum of all Category values - '+str(vAR_target_sum))
    
    # Sum of predicted value with 20% as threshold
    return False  if vAR_target_sum>0.20 else True



def bert_model_result(vAR_input_text):
    
    vAR_test_sentence = vAR_input_text
    vAR_target_columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    
    # Name of the BERT model to use
    model_name = 'bert-base-uncased'

    # Max length of tokens
    max_length = 128

    # Load transformers config and set output_hidden_states to False
    config = BertConfig.from_pretrained(model_name)
    #config.output_hidden_states = False

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
    
    vAR_test_x = tokenizer(
    text=vAR_test_sentence,
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
    
    vAR_load_model = tf.keras.models.load_model('/home/jupyter/DSAI_DMV_Text_Analyzer/DSAI_Model_Implementation_Sourcecode/')

    vAR_model_result = vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = np.sum(vAR_model_result)
    col1, col2, col3 = st.columns([1,8,1])
    with col2:
        st.write(vAR_result_data)
        st.write('Sum of all Category values - '+str(vAR_target_sum))
    return False  if vAR_target_sum>0.20 else True