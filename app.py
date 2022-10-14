import streamlit as st 
import streamlit.components.v1 as stc
import requests 





def main():
	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	st.title("Customer Churn Analysis")

	if choice == "Home":
		st.subheader("Home")

		# Nav  Search Form











		
	else:
		st.subheader("About")




if __name__ == '__main__':
	main()