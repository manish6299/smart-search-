import gradio as gr
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load course data from CSV
df = pd.read_csv('analytics.csv')  # Ensure this CSV file exists with the correct columns

# Generate embeddings for course titles and descriptions
course_embeddings = model.encode(df['Title'] + ' ' + df['Description'], convert_to_tensor=True)

def search_courses(query, top_n=5):
    # Ensure top_n is valid
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    # Generate query embedding
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(query_embedding, course_embeddings)[0]

    # Ensure similarities are valid numbers
    similarities = similarities.numpy()  # Convert tensor to numpy array for easier manipulation
    if np.any(np.isnan(similarities)) or np.any(np.isinf(similarities)):
        raise ValueError("Similarity scores contain NaN or infinity values")

    # Sort the similarities in descending order and select top_n courses
    top_results = np.argsort(similarities)[::-1][:top_n]  # Ensure proper slicing

    # Display top results
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

# Gradio interface function
def gradio_search(query):
    top_courses = search_courses(query)
    results = "\n".join([f"Title: {course['Title']}\nLink: {course['Link']}\nScore: {course['Relevance Score']:.4f}\n"
                         for course in top_courses])
    return results

# Create Gradio interface
iface = gr.Interface(fn=gradio_search,
                     inputs=gr.Textbox(label="Enter your search query"),
                     outputs=gr.Textbox(label="Top Courses"),
                     live=True)

# Launch Gradio app
iface.launch()
