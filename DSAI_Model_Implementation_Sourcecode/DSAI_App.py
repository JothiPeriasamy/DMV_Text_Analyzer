import streamlit as st

from DSAI_DMV_Utility.DSAI_Utility import All_Initialization,CSS_Property
from DSAI_Text_Classification import ClassificationModels
from DSAI_DMV_Layout import ELP_Validation
import traceback
import sys
from PIL import Image

if __name__ == "__main__":
    
    img = Image.open('DSAI_Model_Implementation_Sourcecode/DSAI_DMV_Utility/DMV_Logo.png')
    st.set_page_config(page_title="DMV Vanity Plate Analyzer", layout="wide",page_icon=img)
    try:
        # Applying CSS properties for web page
        CSS_Property("DSAI_Model_Implementation_Sourcecode/DSAI_DMV_Utility/style.css")
        # Initializing Basic Componentes of Web Page
        All_Initialization()
        # ELP Validation Function call
        ELP_Validation()
        
    except BaseException as e:
        col1, col2, col3 = st.columns([1.5,9,1.5])
        with col2:
            st.write('')
            st.error('In Error block - '+str(e))
            traceback.print_exception(*sys.exc_info())