import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import streamlit as st
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()


# Load data
@st.cache_data
def load_books_data():
    books = pd.read_csv("dataset/books_with_emotions.csv")

    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover.jpg",
        books["large_thumbnail"],
    )
    return books

# Load vector database
def load_vector_db(openai_api_key):
    # Set the API key for OpenAI
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    persist_directory = "vector_db"
    
    # Check if database already exists on disk
    if os.path.exists(persist_directory):
        st.info("📚 Loading existing vector database from disk...")
        embeddings = OpenAIEmbeddings()
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        st.info("🔨 Creating new vector database (this may take a few minutes)...")
        # Load documents and create new database
        raw_documents = TextLoader("dataset/tagged_description.txt").load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Save to disk
        vector_store = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=persist_directory
        )
        
        st.success("✅ Vector database created and saved to disk!")
        return vector_store

def retrieve_semantic_recommendations(
        query: str,
        db_books,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    books = load_books_data()
    
    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books_with_db(query: str, category: str, tone: str, db_books):
    recommendations = retrieve_semantic_recommendations(query, db_books, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

def clear_vector_db():
    """Clear the vector database from disk"""
    persist_directory = "vector_db"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        return True
    return False

# Streamlit UI
def main():
    st.set_page_config(
        layout="wide",
        page_title="Recomind",
        page_icon="🧠",
    )

    # Hide hamburger menu and customize footer
    hide_menu = """
        <style>
        #MainMenu {
            visibility: hidden;
        }
        footer {
            visibility: visible;
        }
        footer:after {
            content: 'With 🫶️ from Shubham Shankar.';
            display: block;
            position: relative;
            color: grey;
            padding: 5px;
            top: 3px;
        }
        </style>
    """
    st.markdown(hide_menu, unsafe_allow_html=True)

    # Branding
    st.image("icon.jpg", width=85)
    st.title("🧠 Recomind")
    st.subheader("Discover Your Perfect Book with Smart Recommendations 📚")
    st.write("Leveraging Semantic Search, Text Classification, and Sentiment Analysis for Smarter Choices.")
    st.markdown('---')

    # Intro
    st.write(
        """
        Welcome to **:green[Recomind]** by **:red[Shubham Shankar]**! 🚀

        This app helps you **discover amazing books** like never before.  
        Simply describe what you're looking for, and our AI will find **perfect book recommendations** that match your interests, preferred genre, and even emotional tone.

        ---
        ### 🧠 Why Use This?

        Finding the right book can be overwhelming with millions of options available. **Recomind** solves this by:

        - *"What books match my current mood?"*
        - *"I want something similar to my favorite author..."*
        - *"Show me books in my preferred genre with a specific emotional tone"*

        Instead of endless browsing, **Recomind** uses advanced AI to understand your preferences and find books that truly resonate with you.

        Great for:
        - 📚 Finding your next read
        - 🎭 Mood-based reading
        - 🔍 Genre exploration
        - 💡 Discovering new authors

        ---
        Powered by advanced **:green[LangChain]**, **:rainbow[OpenAI]**, and **:orange[HugginFace]**.
        """
    )

    st.markdown('---')

    # Instructions
    st.write(
        """
        ### 🧭 How to Use:

        🔑 **Setup**
        - Enter your OpenAI API key to get started.

        📝 **Describe Your Book**
        - Tell us what kind of book you're looking for (e.g., "A story about Mindset").

        🎯 **Refine Your Search**
        - Choose a specific category (fiction, non-fiction, etc.).
        - Select an emotional tone (happy, suspenseful, etc.).

        📖 **Discover**
        - Get personalized book recommendations with covers and descriptions.

        """
    )

    st.markdown('---')
    
    # Initialize session state for API key
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    
    # API Key Input Section
    st.markdown("### 🔑 Setup")
    st.markdown("To use this book recommendation system, you need to provide your OpenAI API key.")
    
    # Check if API key is already set
    if not st.session_state.openai_api_key:
        st.info("ℹ️ **How to get an OpenAI API key:**")
        st.markdown("""
        1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Sign in or create an account
        3. Click on "Create new secret key"
        4. Copy the key and paste it below
        """)
        
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-...",
            help="Your API key will be stored securely in your session and won't be saved permanently."
        )
        
        if st.button("🔐 Set API Key", type="primary"):
            if api_key and api_key.startswith("sk-"):
                st.session_state.openai_api_key = api_key
                st.success("✅ API key set successfully! You can now use the book recommender.")
                st.rerun()
            else:
                st.error("❌ Please enter a valid OpenAI API key that starts with 'sk-'")
        
        st.stop()  # Stop execution until API key is provided
    
    # Main application (only shown after API key is set)
    st.success("🔐 API key is configured and ready to use!")
    
    # Load data
    books = load_books_data()

    categories = ["All"] + sorted(books["simple_categories"].unique())
    tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    
    # Input section
    st.markdown("### 📖 Find Your Next Book")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        user_query = st.text_input(
            "Please enter a description of a book:",
            placeholder="e.g., A story about Mindset"
        )
    
    with col2:
        category_dropdown = st.selectbox(
            "Select a category:",
            options=categories,
            index=0
        )
    
    with col3:
        tone_dropdown = st.selectbox(
            "Select an emotional tone:",
            options=tones,
            index=0
        )
    
    # Search button
    if st.button("🔍 Find Recommendations", type="primary", use_container_width=True):
        if user_query.strip():
            try:
                # Combined spinner for the entire process
                with st.spinner("Finding the perfect books for you..."):
                    db_books = load_vector_db(st.session_state.openai_api_key)
                    results = recommend_books_with_db(user_query, category_dropdown, tone_dropdown, db_books)
                
                if results:
                    st.markdown("### 📖 Recommended Books")
                    
                    # Display results in a grid
                    cols = st.columns(4)
                    for idx, (image_url, caption) in enumerate(results):
                        with cols[idx % 4]:
                            st.image(image_url, caption=caption, use_container_width=True)
                else:
                    st.warning("No books found matching your criteria. Try adjusting your search parameters.")
                    
            except Exception as e:
                st.error(f"An error occurred while finding recommendations: {str(e)}")
                st.error("Please check if your OpenAI API key is valid and has sufficient credits.")
        else:
            st.warning("Please enter a book description to get recommendations.")
    
    # API Key and Database Management
    st.markdown("---")
    with st.expander("🔧 Settings & Management"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔑 API Key Management**")
            st.markdown("**Current API Key:** " + st.session_state.openai_api_key[:10] + "..." + st.session_state.openai_api_key[-4:])
            if st.button("🔄 Change API Key"):
                st.session_state.openai_api_key = ""
                st.rerun()
        
        with col2:
            st.markdown("**📚 Database Management**")
            persist_directory = "vector_db"
            if os.path.exists(persist_directory):
                st.success("✅ Vector database exists on disk")
                if st.button("🗑️ Clear Database"):
                    if clear_vector_db():
                        st.success("✅ Database cleared successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear database")
            else:
                st.info("ℹ️ No database found on disk")
    
    # Add some styling
    st.markdown("---")
    
    # Footer
    st.error(
        """
        Connect with me on [**Github**](https://github.com/RATHOD-SHUBHAM) and [**LinkedIn**](https://www.linkedin.com/in/shubhamshankar/). 🧠✨
        """,
        icon="🧑‍💻",
    )
    st.markdown('---')

if __name__ == "__main__":
    main() 