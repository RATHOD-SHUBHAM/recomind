# 🧠 Recomind

**Discover Your Perfect Book with Smart Recommendations 📚**

AI-powered book recommendation system using semantic search, Text-Classification and Sentiment Analysis.

This project focuses on improving the effectiveness of a book recommendation system by addressing two major real-world challenges:

    * The long-tail of noisy, sparse, and inconsistent book categories, and

    * The cold-start problem, where new or low-data books lack sufficient user interaction to be recommended.

## 🧠 Problem We’re Solving
Traditional recommendation systems—especially collaborative filtering—fail when books have:

    * No or missing metadata (e.g. categories, descriptions)

    * Sparse user interactions (e.g. low rating_count)

    * Overly fine-grained or messy category labels (e.g. 300+ unique genres)

This project aims to automatically classify books into a curated set of simplified, high-level categories, enabling:

    * Better generalization across the catalog

    * Improved discoverability of books with little interaction history

    * Richer item representations for hybrid and content-based recommenders

## 🔧 Components Used
1. Data Exploration and Cleaning (EDA)
    - Missing value detection for fields like description, num_pages, rating_count

    - Outlier identification in fields like average_rating and published_year

    - Word-level analysis of description to validate content completeness

2. Category Simplification
    - The raw categories column contains multi-valued, inconsistent labels (e.g., "Fiction; Adventure", "YA; Young Adult").

    - We normalize this into a controlled set of "simple categories" to reduce dimensionality and sparsity.

3. Cold-Start Mitigation via Text Classification
    - Books missing simple_categories are fed through a text classification model trained on their description.

    - Models used may include traditional TF-IDF + Logistic Regression or transformer-based encoders (like BERT).

    - This enables predictions even for books with no ratings or interaction data.

4. Metadata Feature Engineering
Computed fields like:

    - words_in_description — used to measure content richness

    - age_of_book — provides a temporal context

    - title_and_subtitle — creates unified textual features

    - tagged_description — fuses isbn13 and text to preserve traceability during embedding or vectorization

## 🚀 How This Helps in Real Life
    * Publishers can tag and classify books in real-time before reader reviews roll in.

    * New books can be effectively recommended based on their content alone.

    * Recommender systems become more robust by leveraging content, not just past user behavior.

    * Readers benefit from more diverse and personalized recommendations, including from the long tail of less-known titles.

# 🆚 How This Differs from Traditional Systems
#### Traditional Recommenders	
    - Rely mostly on user interactions	
    - Struggle with new books
    - Use noisy, sparse category data
    - Fail with missing metadata
    - Hard to explain or audit

#### This Project
    - Leverages textual content directly
    - Predicts meaningful categories immediately
    - Simplifies and standardizes genre labels
    - Handles missing description, categories
    - Uses interpretable features (title, description)

This system bridges the gap between collaborative filtering and content-based recommendations, improving both cold-start performance and category-level explainability for large-scale book platforms.

---

# 5 Stage Project
    1. Prepare and Clean the dataset.
    2. Create Embeddings and store in DB.
    3. Perform Text-Classification.
    4. Perform Sentiment Analysis.
    5. Create UI

---

# Structure
BookRecommendationSystem/
├── main.py    # Main app
├── Dockerfile               # Docker config
├── docker-compose.yml       # Docker compose
├── .dockerignore           # Docker ignore
├── requirements.txt         # Dependencies
├── pyproject.toml          # UV config
├── README.md               # Simple docs
├── dataset/                # Your data
└── vector_db/              # Created at runtime

---

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
uv sync

# Run the application
uv run streamlit run main.py
```

### Docker
```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t recomind:latest .
docker tag recomind:latest user_name/recomind:latest
docker push user_name/recomind:latest

# Run
docker run -p 8501:8501 user_name/recomind:latest
```

## 🎯 How to Use

1. Enter your OpenAI API key
2. Describe what kind of book you're looking for
3. Choose category and emotional tone
4. Get personalized recommendations

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI**: OpenAI, LangChain, Chroma
- **Data**: Pandas, NumPy
- **Package Manager**: uv

## 👨‍💻 Author

**Shubham Shankar**
- GitHub: [RATHOD-SHUBHAM](https://github.com/RATHOD-SHUBHAM)
- LinkedIn: [shubhamshankar](https://www.linkedin.com/in/shubhamshankar/)

---

Made with 🫶️ by Shubham Shankar 