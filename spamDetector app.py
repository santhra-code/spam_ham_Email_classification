import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('spam123.pkl', 'rb'))  # Replace with your model filename
cv = pickle.load(open('vector123.pkl', 'rb'))      # Replace with your vectorizer filename

# Define the main function for the Streamlit app
def main():
    # App title and description
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.subheader("Classification")

    # Text input area for user to enter an email
    user_input = st.text_area("Enter an email to classify", height=150)

    # Button to classify the input
    if st.button("Classify"):
        if user_input:
            # Preprocess the input and classify
            data = [user_input]
            st.write(f"Input data: {data}")  # Display input text (for debugging purposes)
            
            # Transform input using the loaded vectorizer
            vec = cv.transform(data).toarray()
            
            # Predict using the loaded model
            prediction = model.predict(vec)
            
            # Display the result
            if prediction[0] == 1:
                st.success("The email is classified as **Spam**.")
            else:
                st.success("The email is classified as **Ham**.")

# Run the main function
if __name__ == "__main__":
    main()
