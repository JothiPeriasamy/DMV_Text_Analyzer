import re
import streamlit as st

def pattern_denial(input_text):
    result = re.findall("^A\d{1,2}$", input_text)
    col1, col2, col3 = st.columns([1.5,9,1.5])
    with col2:
        if len(result)>0:
            if not result:
                pass
                # st.info("Passed at State assembly pattern - Not in State assembly pattern")
            elif result[0]==input_text:
                st.write('')
                st.error("Denied - Similar to State assembly pattern")
                return False

            
        result = re.findall("^S[0-9]{1,2}[a-zA-Z]{1}$", input_text)

        if not result:
            pass
            # st.info("Passed at State senate pattern - Not in State senate pattern")
        else:
            st.write('')
            st.error("Denied - Similar to State senate pattern")
            return False

        
        result = re.findall("^C\d{1,2}\w{1}\d{1}$", input_text)

        if len(result)==0:
            pass
            # st.info("Passed at US Congress pattern - Not in US Congress pattern")
        else:
            st.write('')
            st.error("Denied - Similar to US Congress pattern")
            return False
    
    
        result = re.findall("^[a-zA-Z]{1}\d{2}[a-zA-Z]{1}\d{2}$", input_text)

        if len(result)==0:
            pass
            # st.info("Passed at US Congress pattern - Not in US Congress pattern")
        else:
            st.write('')
            st.error("Denied - Similar to OHV pattern")
            return False
        
        
        result = re.findall("^H\d{4}$", input_text)

        if len(result)==0:
            pass
            # st.info("Passed at US Congress pattern - Not in US Congress pattern")
        else:
            st.write('')
            st.error("Denied - Similar to HONORARY CONSUL pattern")
            return False
        
        
        result = re.findall("^[a-zA-Z]{1}\d{5}$", input_text)
        
        if len(result)==0:
            pass
            # st.info("Passed at US Congress pattern - Not in US Congress pattern")
        else:
            st.write('')
            st.error("Denied - Similar to COMMERCIAL pattern")
            return False
    
    
    
    
    return True
    