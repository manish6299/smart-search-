import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the course data from CSV
df = pd.read_csv('analytics.csv')

# Generate embeddings for course titles and descriptions
course_embeddings = model.encode(df['Title'] + ' ' + df['Description'], convert_to_tensor=True)

# Function to search for courses based on a query
def search_courses(query, top_n=5):
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, course_embeddings)[0].numpy()

    if np.any(np.isnan(similarities)) or np.any(np.isinf(similarities)):
        raise ValueError("Similarity scores contain NaN or infinity values")

    top_results = np.argsort(similarities)[::-1][:top_n]

    results = []
    for idx in top_results:
        course_info = {
            'Title': df.iloc[idx]['Title'],
            'Description': df.iloc[idx]['Description'],
            'Link': df.iloc[idx]['Link'],
            'Relevance Score': similarities[idx].item()
        }
        results.append(course_info)

    return results

# Streamlit app configuration
st.set_page_config(page_title="Smart Course Finder", layout="centered")

# App title and description
st.title("🔍 Smart Course Finder for Analytics Vidhya")
st.markdown("""
Find the best free courses that match your learning interests. Enter a keyword or topic to discover the most relevant courses available.
""")

# Input field for the query
query = st.text_input("Enter your search query:", placeholder="e.g., Machine Learning, Data Science, Generative AI")

# Search button and display results
if st.button("Search") and query:
    with st.spinner("Searching for courses..."):
        try:
            top_courses = search_courses(query)
            if top_courses:
                st.success("Top courses found:")
                for course in top_courses:
                    st.subheader(f"📘 {course['Title']}")
                    st.write(f"**Description**: {course['Description']}")
                    st.write(f"**Relevance Score**: {course['Relevance Score']:.2f}")
                    st.markdown(f"[🔗 View Course]({course['Link']})", unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.warning("No courses found for the given query. Try different keywords.")
        except ValueError as e:
            st.error(f"Error: {e}")

# Footer with credits
st.markdown("""
---
*Built with ❤️ using Streamlit and SentenceTransformers*
""")
