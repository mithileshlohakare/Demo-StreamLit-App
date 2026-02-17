import streamlit as st
import pandas as pd
import numpy as np

# st.title("Hello streamlit!")
# st.write("streamlit, this is your first streamlit app")
# st.text("Let's get started")

# df = pd.DataFrame(np.random.randn(10, 2), columns=["A", "B"])
# st.line_chart(df)
# st.bar_chart(df)

# st.sidebar.title("navigation")
# st.image("https://pixabay.com/images/search/red%20flower/", caption="Red Flower")
# st.video("https://www.youtube.com/watch?v=jJAmUJZtAMY")

# st.subheader("Upload a File")
# uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "jpg", "png"])
# if uploaded_file is not None:
#     if uploaded_file.type == "text/csv":
#         data = pd.read_csv(uploaded_file)
#         st.write("Uploaded CSV:")
#         st.dataframe(data)
#     elif uploaded_file.type.startswith("image"):
#         st.image(uploaded_file, caption="Uploaded Image")
#     else:
#         st.write("File uploaded:", uploaded_file.name)

# st.sidebar.title("Navigation")
# option = st.sidebar.radio(
#     "Choose a topic:",
#     ("Data Table", "Line Chart", "Bar Chart", "Area Chart")
# )

# if option == "Data Table":
#     st.subheader("Data Table")
#     st.dataframe(df)

# elif option == "Line Chart":
#     st.subheader("Line Chart")
#     st.line_chart(df)

# elif option == "Bar Chart":
#     st.subheader("Bar Chart")
#     st.bar_chart(df)

# elif option == "Area Chart":
#     st.subheader("Area Chart")
#     st.area_chart(df)


# # st.text_input("Enter your name", key="name")
# # st.text_area("Enter your message", key="message")
# # st.number_input("Enter a number", key="number")
# # st.slider("Select a value", 0, 100, key="slider")
# # st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"], key="selectbox")
# # st.multiselect("Select multiple options", ["Option A", "Option B", "Option C"], key="multiselect")
# # st.radio("Choose one", ["Choice 1", "Choice 2", "Choice 3"], key="radio")
# # st.checkbox("Check me", key="checkbox")

# st.subheader("Interactive Inputs")

# # Text input
# name = st.text_input("Enter your name:")
# if name:
#     st.write(f"Hello, {name}! 👋")

# # Text area
# message = st.text_area("Write a message:")
# if message:
#     st.write("Your message:", message)

# # Number input
# age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
# st.write("Age entered:", age)

# # Slider
# rating = st.slider("Rate this app:", min_value=0, max_value=10, value=5)
# st.write("Your rating:", rating)

# # Selectbox (single selection)
# color = st.selectbox("Choose a color:", ["Red", "Green", "Blue", "Yellow"])
# st.write("Selected color:", color)

# # Multiselect (multiple selections)
# hobbies = st.multiselect("Select your hobbies:", ["Reading", "Traveling", "Gaming", "Cooking"])
# st.write("Your hobbies:", hobbies)

# # Radio buttons
# gender = st.radio("Select your gender:", ["Male", "Female", "Other"])
# st.write("Selected gender:", gender)

# # Checkbox
# subscribe = st.checkbox("Subscribe to newsletter")
# if subscribe:
#     st.write("✅ You are subscribed!")
# else:
#     st.write("❌ Not subscribed")

st.title("Streamlit Form Example")

# Create a form
st.subheader("Login Form")

# Username input
username = st.text_input("Username")

# Password input (masked)
password = st.text_input("Password", type="password")

# Login button
login_button = st.button("Login")

# Check credentials (basic example)
if login_button:
    if username == "admin" and password == "1234":
        st.success("Login successful! 🎉")
    else:
        st.error("Invalid username or password")