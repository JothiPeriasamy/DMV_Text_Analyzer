import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from DSAI_DMV_Pattern_Denial import Pattern_Denial
from DSAI_Text_Classification import ClassificationModels
# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel
from DSAI_DMV_Utility import SessionState
from bokeh.models.widgets import Div
import time
from streamlit_option_menu import option_menu



# import SessionState

session_state_order = SessionState.get(name="", vAR_choice_orders=False)
session_state_recommends = SessionState.get(name="", vAR_choice_recommends=False)
session_state_search = SessionState.get(name="", vAR_choice_search=False)

# Validate ELP
def ELP_Validation():
    vAR_model_result = None
    vAR_input_list = []
    vAR_input_len_list = []
    vAR_personal = False
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
            col1,col2,col3 = st.columns([1.15,7.1,1.15])
            col4, col5, col6, col7,col8 = st.columns([1.5,4,1,4,1.5])
            
            col9,col10,col11 = st.columns([1.15,7.1,1.15])
            col12, col13, col14, col15,col16 = st.columns([1.5,4,1,4,1.5])
            
            col17,col18,col19 = st.columns([1.15,7.1,1.15])
            col20, col21, col22, col23,col24 = st.columns([1.5,4,1,4,1.5])
            
            st.session_state.counter+=1
            with col2:
                st.markdown("<h1 style='text-align: center; color: black; font-size:15px;'>Section 1 - Plate Selection -  Check one (For special plates not listed, use REG 17A)</h1>", unsafe_allow_html=True)
            with col5:
                st.write('')
                st.subheader('Plates allowed 2-6 Characters')
                st.write('')
                vAR_breast_cancer = st.checkbox('Breast Cancer Awareness')
                vAR_california_arts = st.checkbox('California Arts Council')
                vAR_california_agri = st.checkbox('California Agricultural')
                vAR_california_memorial = st.checkbox('California Memorial')
                vAR_california_museums = st.checkbox('California Museums')
                vAR_collegiate = st.checkbox('Collegiate (only UCLA is available)')
                vAR_kids = st.checkbox('Kids - Child Health and Safety Funds (SYMBOLS: HEART, STAR, HAND OR PLUS SIGN)')
                
            with col7:
                st.write('')
                st.subheader('Plates allowed 2-7 Characters')
                st.write('')
                vAR_elp = st.checkbox('Environmental License Plate (ELP) (BASIC PERSONALIZED PLATE)')
                vAR_california_coastal = st.checkbox('California Coastal Commission (Whale Tail)')
                vAR_lake_tahoe = st.checkbox('Lake Tahoe Conservancy')
                vAR_yosemite = st.checkbox('Yosemite Foundation')
                vAR_california = st.checkbox('California 1960s Legacy (6 character sequential, 2-7 characters for ELP)')
                
                
            with col10:
                st.markdown("<h1 style='text-align: center; color: black; font-size:15px;'>Section 2 - Select Configuration</h1>", unsafe_allow_html=True)
                st.write('')
                vAR_sequential = st.checkbox('Sequential (Non-Personalized) — Issued in number sequence')
                st.write('Note : Your existing sequential license plate number cannot be re-used. You must submit a copy of your current registration card. ')
            print('checkbox value - ',vAR_sequential)
            if vAR_sequential and not vAR_personal:
                with col13:
                    vAR_current_plate = st.text_input('Current License Plate Number','',key='current_plate')
                with col15:
                    vAR_vehicle_id = st.text_input('Full Vehicle Id Number','',key='current_plate')
            with col18:
                vAR_personal = st.checkbox('Personalized Configuration Choice')
                st.write('''Note : DMV has the right to refuse any combination of letters and/or letters and numbers for any of the following reason(s): it could be considered
offensive to good taste and decency in any language or slang term, it substitutes letters for numbers or vice versa (e.g. ROBERT/RO8ERT),
to look like another personalized plate, or it conflicts with any regular license plate series issued.
Your application will not be accepted if the MEANING of the plate is not entered, even if it appears obvious, OR if the plate configuration
is unacceptable''')
                st.write("**For KIDS Plate : **Select Choice of Symbol")
                vAR_icons = option_menu(None, ["Heart", "Star", "Plus", 'Hand'], 
    icons=['heart-fill', 'star-fill', "align-middle", 'hand-index-thumb-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
            if vAR_personal and not vAR_sequential:
                with col21:
                    st.write('')
                    st.write('')
                    st.subheader('No.Of Configuration')


                with col23:
                    vAR_number_of_config = st.number_input('',step=1,max_value=3,value=1)

                for vAR_idx in range(0,vAR_number_of_config):
                    with col21:
                        st.write('')
                        st.write('')
                        st.write('')
                        if vAR_idx==1 or vAR_idx==2:
                            st.write('')
                            st.write('')
                            st.write('')
                            st.write('')
                            st.write('')
                            st.write('')
                            st.write('')
                        st.subheader('ELP Configuration '+str(vAR_idx+1))
                    with col23:
                        st.write('')
                        if vAR_idx==2:
                            st.write('')
                        vAR_input_text = st.text_input('Enter input','',key=str(vAR_idx),max_chars=7).upper().strip()
                        vAR_meaning = st.text_input('Meaning','',key=str('meaning'+str(vAR_idx)),max_chars=150)
                        vAR_input_list.append(vAR_input_text)
                for vAR_value in vAR_input_list:
                    length = len(vAR_value)
                    vAR_input_len_list.append(length)


                if len(vAR_input_list)>0 and vAR_input_len_list.count(0)<1:
                    vAR_model_result_list = Process_Result(vAR_input_list)
                    print('result list - ',vAR_model_result_list)
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
                    col1, col2, col3 = st.columns([1.15,7.1,1.15])
                    with col2:
                        st.write('')
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
def LSTM_Model_Result(vAR_input_text):
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
def Load_BERT_Model():
    vAR_load_model = tf.keras.models.load_model('DSAI_Model_Implementation_Sourcecode/BERT_MODEL_64B_4e5LR_3E')
    return vAR_load_model
    
    
@st.cache(show_spinner=False)
def BERT_Model_Result(vAR_input_text):
    
    
    
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
    start_time = time.time()
    vAR_load_model = tf.keras.models.load_model('DSAI_Model_Implementation_Sourcecode/BERT_MODEL_64B_4e5LR_3E')
    # vAR_load_model = Load_BERT_Model()
    print("--- %s seconds ---" % (time.time() - start_time))
    

    vAR_model_result = vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)
    
    # if "vAR_load_model" not in st.session_state:
    #     st.session_state.vAR_load_model = tf.keras.models.load_model('DSAI_Model_Implementation_Sourcecode/BERT_MODEL_64B_4e5LR_3E')
    # vAR_model_result = st.session_state.vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = (np.sum(vAR_model_result)*100).round(2)
    vAR_result_data.index = pd.Index(['Percentage'],name='category')
    vAR_result_data = vAR_result_data.astype(float).round(5)*100
    
    if vAR_target_sum>20:
        return False,vAR_result_data,vAR_target_sum
    else:
        return True,vAR_result_data,vAR_target_sum
 




    
def Process_Result(vAR_input_list):
    vAR_model_result = None
    vAR_model_result_list = [None]*4
    vAR_pattern_result = False
    for vAR_idx,vAR_val in enumerate(vAR_input_list):
        vAR_is_badword = Profanity_Words_Check(vAR_val)
        print('Is bad word - ',vAR_is_badword)
        if not vAR_is_badword:
            col1, col2, col3 = st.columns([1.5,9,1.5])
            with col2:
                st.info('ELP Configuration **'+vAR_val+ '** Successfully processed for 1st level(Configuration not Falls under profanity/obscene word category)')
            
            vAR_pattern_result = Pattern_Denial(vAR_val)
            print('vAR_pattern_result - ',vAR_pattern_result)
            if vAR_pattern_result:
                st.write('')

                col1, col2, col3 = st.columns([1.5,9,1.5])
                with col2:
                    st.info('ELP Configuration **'+vAR_val+ '** Successfully processed for 2nd Level')
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
                            vAR_model_result_list[vAR_idx],vAR_result_data,vAR_target_sum = LSTM_Model_Result(vAR_val) 
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
                            vAR_model_result_list[vAR_idx],vAR_result_data,vAR_target_sum = BERT_Model_Result(vAR_val)
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
                        st.info('ELP Configuration **'+vAR_val+ '** Successfully processed for 3rd Level')
                        st.write('')
                        st.success('ELP Configuration **'+vAR_val+ '** Approved Successfully')
                elif vAR_model_result_list[vAR_idx]==False:
                    col1, col2, col3 = st.columns([1.5,9,1.5])
                    with col2:
                        st.write('')
                        st.error('ELP Configuration **'+vAR_val+ '** Failed to Meet the DMV Requirements at 3rd level')
                        Denial_Letter(vAR_idx)
            else:
                col1, col2, col3 = st.columns([1.5,9,1.5])
                with col2:
                    vAR_model_result_list[vAR_idx] = vAR_pattern_result 
                    st.write('')
                    st.error('ELP Configuration **'+vAR_val+ '** Failed to Meet the DMV Requirements at 2nd level')
                    Denial_Letter(vAR_idx)
        else:
            vAR_model_result_list[vAR_idx] = not vAR_is_badword
            col1, col2, col3 = st.columns([1.3,7.6,1.3])
            with col2:
                st.write('')
                st.error('ELP Configuration **'+vAR_val+ '** Failed to Meet the DMV Requirements at 1st level(Configuration Falls under profanity/obscene word category)')
                Denial_Letter(vAR_idx)
                
    return vAR_model_result_list

    
    
def Denial_Letter(vAR_idx):
    st.write('')
    vAR_denial_letter = st.button("Click here to view denial letter",key=str(vAR_idx))
    if vAR_denial_letter:
        js = "window.open('https://datastudio.google.com/reporting/eb175286-477a-4d4c-ac6b-303a40a820d9')"
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)
        
        
def Profanity_Words_Check(vAR_val):
    vAR_input = vAR_val
    vAR_badwords_df = pd.read_csv('DSAI_Dataset/badwords_list.csv',header=None)
    
#---------------Profanity logic implementation with O(log n) time complexity-------------------
    # Direct profanity check
    vAR_badwords_df[1] = vAR_badwords_df[1].str.upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_input)
    if vAR_is_input_in_profanity_list!=-1:
        print('Input ' +vAR_val+ ' matches with direct profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list])
        return True
    
    # Reversal profanity check
    vAR_reverse_input = "".join(reversed(vAR_val))
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_reverse_input)
    if vAR_is_input_in_profanity_list!=-1:
        print('Input ' +vAR_val+ ' matches with reversal profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list])
        return True
    
    # Number replacement profanity check
    vAR_number_replaced = Number_Replacement(vAR_val)
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_number_replaced)
    if vAR_is_input_in_profanity_list!=-1: 
        print('Input ' +vAR_val+ ' matches with number replacement profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list])
        return True
    
    # Reversal Number replacement profanity check(5sa->as5->ass)
    vAR_number_replaced = Number_Replacement(vAR_reverse_input)
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_number_replaced)
    if vAR_is_input_in_profanity_list!=-1:  
        print('Input ' +vAR_val+ ' matches with reversal number replacement profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list])
        return True
    
    return False
    
    
# ---------------Profanity logic implementation with O(n) time complexity---------------------
#     for index, row in vAR_badwords_df.iterrows():
#         badword = row[1].upper()
        
#         # Direct Profanity check
#         if badword==vAR_val:
#             print('Input ' +vAR_val+ ' matches with direct profanity - '+badword)
#             return True
        
#         vAR_reverse_input = "".join(reversed(vAR_val))
#         # Reversal profanity check
#         if badword==vAR_reverse_input:
#             print('Input ' +vAR_val+ ' matches with reversal profanity - '+badword)
#             return True
        
#         # Number replacement profanity check
#         vAR_number_replaced = Number_Replacement(vAR_val)
#         if badword==vAR_number_replaced: 
#             print('Input ' +vAR_val+ ' matches with number replacement profanity - '+badword)
#             return True
        
#         # Reversal Number replacement profanity check(5sa->as5->ass)
#         vAR_number_replaced = Number_Replacement(vAR_reverse_input)
#         if badword==vAR_number_replaced: 
#             print('Input ' +vAR_val+ ' matches with reversal number replacement profanity - '+badword)
#             return True
        
#     return False
                
    
def Number_Replacement(vAR_val):
    vAR_output = vAR_val
    if "1" in vAR_val:
        vAR_output = vAR_output.replace("1","I")
    if "2" in vAR_val:
        vAR_output = vAR_output.replace("2","Z")
    if "3" in vAR_val:
        vAR_output = vAR_output.replace("3","E")
    if "4" in vAR_val:
        vAR_output = vAR_output.replace("4","A")
    if "5" in vAR_val:
        vAR_output = vAR_output.replace("5","S")
    if "8" in vAR_val:
        vAR_output = vAR_output.replace("8","B")
        print('8 replaced with B - ',vAR_val)
    if "0" in vAR_val:
        vAR_output = vAR_output.replace("0","O")
    print('number replace - ',vAR_output)
    return vAR_output



def Binary_Search(data, x):
    low = 0
    high = len(data) - 1
    mid = 0
    i =0
    while low <= high:
        i = i+1
        print('No.of iteration - ',i)
        mid = (high + low) // 2
        
        # If x is greater, ignore left half
        if data[mid] < x:
            low = mid + 1
 
        # If x is smaller, ignore right half
        elif data[mid] > x:
            high = mid - 1
 
        # means x is present at mid
        else:
            return mid
 
    # If we reach here, then the element was not present
    return -1