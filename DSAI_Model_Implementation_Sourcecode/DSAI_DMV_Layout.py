import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from DSAI_DMV_Pattern_Denial import pattern_denial
from DSAI_Text_Classification import ClassificationModels
# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel
from DSAI_DMV_Utility import SessionState
from bokeh.models.widgets import Div


# import SessionState

session_state_order = SessionState.get(name="", vAR_choice_orders=False)
session_state_recommends = SessionState.get(name="", vAR_choice_recommends=False)
session_state_search = SessionState.get(name="", vAR_choice_search=False)

# Validate ELP
def ELP_Validation():
    vAR_model_result = None
    vAR_input_list = []
    vAR_input_len_list = []
    try:
        col5,col6,col7,col8,col9 = st.columns([1.5,3,3,3,1.5])

        with col6:
            st.write('')
            vAR_choice_orders = st.button('Orders')
        with col7:
            st.write('')
            vAR_choice_recommends = st.button('Recommendations')
        with col8:
            st.write('')
            vAR_choice_search = st.button('Search')
        
        # Initialize Session state variable
        if "counter" not in st.session_state:
            st.session_state.counter = 0
        
        if vAR_choice_orders:
            session_state_order.vAR_choice_orders = True
            st.session_state.counter+=1
        if vAR_choice_recommends:
            session_state_recommends.vAR_choice_recommends = True
            st.session_state.counter+=1
            session_state_order.vAR_choice_orders = False
        if vAR_choice_search:
            session_state_search.vAR_choice_search = True
            st.session_state.counter+=1
            session_state_order.vAR_choice_orders = False
        
        if session_state_order.vAR_choice_orders and st.session_state.counter!=0:
            col1, col2, col3, col4,col_ = st.columns([1.5,4,1,4,1.5])
            st.session_state.counter+=1
            with col2:
                st.write('')
                st.write('')
                st.write('')
                st.subheader('No.Of Configuration')
                
            with col4:
                st.write('')
                vAR_number_of_config = st.number_input('',step=1,max_value=4,value=1)
                
            for vAR_idx in range(0,vAR_number_of_config):
                with col2:
                    st.write('')
                    st.write('')
                    st.subheader('ELP Configuration '+str(vAR_idx+1))
                with col4:
                    vAR_input_text = st.text_input('Enter input','',key=str(vAR_idx)).upper()
                    vAR_input_list.append(vAR_input_text)
            for vAR_value in vAR_input_list:
                length = len(vAR_value)
                vAR_input_len_list.append(length)
                
                
            if len(vAR_input_list)>0 and vAR_input_len_list.count(0)<1:
                vAR_model_result_list = process_result(vAR_input_list)
                print('Result list - ',vAR_model_result_list)
                print('input list - ',vAR_input_list)
                if vAR_model_result_list.count(False)>0:
                    col1, col2, col3 = st.columns([1.5,9,1.5])
                    with col2:
                        st.write('')
                        st.warning('Please try again with another configuration to proceed further')
                elif vAR_model_result_list.count(True)==len(vAR_input_list):
                    col1, col2, col3 = st.columns([1.5,9,1.5])
                    with col2:
                        st.write('')
                        vAR_payment = st.button("Initiate Payment to order license plate")
                    if vAR_payment:
                        col1, col2, col3 = st.columns([1.5,9,1.5])
                        with col2:
                            st.write('')
                            st.info('Development in-progress')
                
            else:
                col1, col2, col3 = st.columns([1.5,9,1.5])
                with col2:
                    st.write('')
                    st.warning("Please enter input details")



        elif (session_state_recommends.vAR_choice_recommends or session_state_search.vAR_choice_search) and st.session_state.counter!=0:
            session_state_order.vAR_choice_orders = False
            col1, col2, col3 = st.columns([1.3,7.6,1.3])
            with col2:
                st.write('')
                st.write('')
                st.info("Search and Recommendations Functionality Development is in-progress")
    except AttributeError:
        pass

        
@st.cache(show_spinner=False)
def lstm_model_result(vAR_input_text):
    # Input Data Preprocessing
    vAR_data = pd.DataFrame()
    vAR_target_columns = ['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
    vAR_model_obj = ClassificationModels(vAR_data,vAR_target_columns)
    vAR_test_data = pd.DataFrame([vAR_input_text],columns=['comment_text'])
    vAR_test_data['Toxic'] = None
    vAR_test_data['Severe Toxic'] = None
    vAR_test_data['Obscene'] = None
    vAR_test_data['Threat'] = None
    vAR_test_data['Insult'] = None
    vAR_test_data['Identity Hate'] = None
    print('Xtest length - ',len(vAR_test_data))
    vAR_corpus = vAR_model_obj.data_preprocessing(vAR_test_data)
    print('Data Preprocessing Completed')
    vAR_X,vAR_y = vAR_model_obj.word_embedding_vectorization(vAR_corpus,vAR_test_data)
    print('Vectorization Completed Using Word Embedding')
    print('var X - ',vAR_X)
    print('var Y - ',vAR_y)
    
    vAR_load_model = tf.keras.models.load_model('DSAI_Model_Implementation_Sourcecode/LSTM_RNN_Model')

    vAR_model_result = vAR_load_model.predict(vAR_X)
    print('LSTM result - ',vAR_model_result)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = (np.sum(vAR_model_result)*100).round(2)
    vAR_result_data.index = pd.Index(['Percentage'],name='category')
    vAR_result_data = vAR_result_data.astype(float).round(5)*100
    

    # Sum of predicted value with 20% as threshold
    if vAR_target_sum>20:
        return False,vAR_result_data,vAR_target_sum
    else:
        return True,vAR_result_data,vAR_target_sum


@st.cache(show_spinner=False)
def bert_model_result(vAR_input_text):
    
    vAR_test_sentence = vAR_input_text
    vAR_target_columns = ['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
    
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

    vAR_model_result = vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = (np.sum(vAR_model_result)*100).round(2)
    vAR_result_data.index = pd.Index(['Percentage'],name='category')
    vAR_result_data = vAR_result_data.astype(float).round(5)*100
    
    if vAR_target_sum>20:
        return False,vAR_result_data,vAR_target_sum
    else:
        return True,vAR_result_data,vAR_target_sum
 
    
    
def process_result(vAR_input_list):
    vAR_model_result = None
    vAR_model_result_list = [None]*4
    for vAR_idx,vAR_val in enumerate(vAR_input_list):
        vAR_pattern_result = pattern_denial(vAR_val)
        if  vAR_pattern_result:
            st.write('')
            col1, col2, col3 = st.columns([1.5,9,1.5])
            with col2:
                st.info('ELP Configuration **'+vAR_val+ '** Successfully processed for 1st Level')
            col1, col2, col3, col4,col_ = st.columns([1.5,4,1,4,1.5])
            with col2:
                st.write('')
                st.write('')
                st.subheader('Select Model')
            with col4:
                vAR_model = st.selectbox('',('Select Model','BERT','LSTM-RNN'),key=str(vAR_idx))
            if vAR_model == 'LSTM-RNN':
                col1, col2, col3 = st.columns([1.5,9,1.5])
                with col2:
                    st.write('')
                    with st.spinner(text='Model Prediction is in-progress'):
                        vAR_model_result_list[vAR_idx],vAR_result_data,vAR_target_sum = lstm_model_result(vAR_val) 
                col1, col2, col3 = st.columns([3,4,3])
                with col2:
                    st.write('')
                    st.write(vAR_result_data.transpose())
                    st.write('Sum of all Category values for configuration **'+vAR_val+ '** - '+str(vAR_target_sum))
            elif vAR_model == 'BERT':
                col1, col2, col3 = st.columns([1.5,9,1.5])
                with col2:
                    st.write('')
                    with st.spinner(text='Model Prediction is in-progress'):
                        vAR_model_result_list[vAR_idx],vAR_result_data,vAR_target_sum = bert_model_result(vAR_val)
                col1, col2, col3 = st.columns([3,4,3])
                with col2:
                    st.write('')
                    st.write(vAR_result_data.transpose())
                    st.write('Sum of all Category values for configuration **'+vAR_val+ '** - '+str(vAR_target_sum))
            elif vAR_model=='Select Model':
                st.write('')
                col1, col2, col3 = st.columns([1.5,9,1.5])
                with col2:
                    st.write('')
                    st.warning('Select model for 2nd level check for configuration** '+vAR_val+'**')
            if vAR_model_result_list[vAR_idx]:
                col1, col2, col3 = st.columns([1.5,9,1.5])
                with col2:
                    st.write('')
                    st.info('ELP Configuration **'+vAR_val+ '** Successfully processed for 2nd Level')
                    st.write('')
                    st.success('ELP Configuration **'+vAR_val+ '** Approved Successfully')
            elif vAR_model_result_list[vAR_idx]==False:
                col1, col2, col3 = st.columns([1.5,9,1.5])
                with col2:
                    st.write('')
                    st.error('ELP Configuration **'+vAR_val+ '** Failed to Meet the DMV Requirements at 2nd level')
                    denial_letter(vAR_idx)
            # vAR_model_result_list.append(vAR_model_result_list[vAR_idx])

        else:
            col1, col2, col3 = st.columns([1.3,7.6,1.3])
            with col2:
                st.write('')
                st.error('ELP Configuration **'+vAR_val+ '** Failed to Meet the DMV Requirements at 1st level')
                denial_letter(vAR_idx)
                
    return vAR_model_result_list

    
    
def denial_letter(vAR_idx):
    st.write('')
    vAR_denial_letter = st.button("Click here to view denial letter",key=str(vAR_idx))
    if vAR_denial_letter:
        js = "window.open('https://datastudio.google.com/reporting/eb175286-477a-4d4c-ac6b-303a40a820d9')"
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)