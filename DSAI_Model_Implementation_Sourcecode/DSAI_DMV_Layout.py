import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from DSAI_DMV_Pattern_Denial import pattern_denial
from DSAI_Text_Classification import ClassificationModels
# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel
from DSAI_DMV_Utility import SessionState

# ss = SessionState.get(vAR_choice_orders = None,vAR_choice_recommends=None,vAR_choice_search=None,vAR_model=None)

# import SessionState

session_state_order = SessionState.get(name="", vAR_choice_orders=False)
session_state_recommends = SessionState.get(name="", vAR_choice_recommends=False)
session_state_search = SessionState.get(name="", vAR_choice_search=False)


def ELP_Validation():
    try:
        vAR_model_result = None
        col5,col6,col7,col8,col9 = st.columns([1,3,3,3,1])
        col1, col2, col3, col4 = st.columns([1,4,4,1])

        with col6:
            vAR_choice_orders = st.button('Orders')
        with col7:
            vAR_choice_recommends = st.button('Recommendations')
        with col8:
            vAR_choice_search = st.button('Search')
        with col2:
            st.write('')
            st.write('')
            st.write('')
            st.subheader('ELP Configuration')
        with col3:
            st.write('')
            vAR_input_text = st.text_input('Enter input','').upper()

        if vAR_choice_orders:
            session_state_order.vAR_choice_orders = True
        if vAR_choice_recommends:
            session_state_recommends.vAR_choice_recommends = True
            session_state_order.vAR_choice_orders = False
        if vAR_choice_search:
            session_state_search.vAR_choice_search = True
            session_state_order.vAR_choice_orders = False
        if session_state_order.vAR_choice_orders:    
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
                        vAR_model_result = bert_model_result(vAR_input_text)
                    elif vAR_model=='Select Model':
                        st.write('')
                        col1, col2, col3 = st.columns([1,8,1])
                        with col2:
                            st.write('')
                            st.warning('Select model for 2nd level check')
                    if vAR_model_result:
                        col1, col2, col3 = st.columns([1,8,1])
                        with col2:
                            st.write('')
                            st.info('ELP Successfully processed for 2nd Level')
                            st.write('')
                            st.success('ELP Configuration Approved Successfully')
                    elif vAR_model_result is not None:
                        col1, col2, col3 = st.columns([1,8,1])
                        with col2:
                            st.write('')
                            st.error('ELP Configuration Failed to Meet the DMV Requirements at 2nd level')
                else:
                    col1, col2, col3 = st.columns([1,8,1])
                    with col2:
                        st.write('')
                        st.error('ELP Configuration Failed to Meet the DMV Requirements at 1st level')
            else:
                col1, col2, col3 = st.columns([1,8,1])
                with col2:
                    st.write('')
                    st.warning("Please enter input details")

        elif session_state_recommends.vAR_choice_recommends or session_state_search.vAR_choice_search:
            session_state_order.vAR_choice_orders = False
            col1, col2, col3 = st.columns([1,8,1])
            with col2:
                st.write('')
                st.write('')
                st.info("Search and Recommendations Functionality Development is in-progress")
    except AttributeError:
        session_state_recommends.vAR_choice_recommends = True

    
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
    
    vAR_load_model = tf.keras.models.load_model('DSAI_Model_Implementation_Sourcecode/LSTM_RNN_Model')

    vAR_model_result = vAR_load_model.predict(vAR_X)*100
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = np.sum(vAR_model_result)
    col1, col2, col3 = st.columns([3,4,3])
    with col2:
        vAR_result_data.index = pd.Index(['percentage'],name='category')
        st.write(vAR_result_data.transpose())
        st.write('Sum of all Category values - '+str(vAR_target_sum))
    
    # Sum of predicted value with 20% as threshold
    return False  if vAR_target_sum>20 else True



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
    
    vAR_load_model = tf.keras.models.load_model('DSAI_Model_Implementation_Sourcecode/BERT_MODEL')

    vAR_model_result = vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)*100
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = np.sum(vAR_model_result)
    col1, col2, col3 = st.columns([3,4,3])
    with col2:
        vAR_result_data.index = pd.Index(['percentage'],name='category')
        st.write(vAR_result_data.transpose())
        st.write('Sum of all Category values - '+str(vAR_target_sum))
    return False  if vAR_target_sum>20 else True