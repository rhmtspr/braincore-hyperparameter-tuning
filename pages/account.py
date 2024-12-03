import streamlit as st
from auth import auth_utils

if 'user_info' not in st.session_state:
    st.header('Access is forbidden')
    st.markdown('##### Please login first.')
else:
    # Show user information
    st.header('User information:')

    # Sign out
    st.header('Sign out:')
    st.button(label='Sign Out',on_click=auth_utils.sign_out,type='primary')

    # Delete Account
    st.header('Delete account:')
    password = st.text_input(label='Confirm your password',type='password')
    st.button(label='Delete Account',on_click=auth_utils.delete_account,args=[password],type='primary')