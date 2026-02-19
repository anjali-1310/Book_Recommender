import streamlit as st
import pandas as pd
import pickle

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Book Recommendation System",
    layout="wide"
)

st.title("📚 Book Recommendation System")
st.write("User-User Collaborative Filtering Model")

# ===============================
# LOAD DATA (CACHED FOR SPEED)
# ===============================
@st.cache_data
def load_books():
    books = pd.read_csv("books.csv", encoding="latin-1")
    books = books.dropna(subset=["Image-URL-M"])
    return books

@st.cache_data
def load_ratings():
    return pd.read_csv("final_ratings.csv")

books = load_books()
ratings = load_ratings()

# ===============================
# LOAD RECOMMENDATIONS
# ===============================
with open("recs.pkl", "rb") as f:
    recs = pickle.load(f)

# ===============================
# CREATE MAPPINGS
# ===============================
book_mapping = ratings[['book_idx', 'ISBN']].drop_duplicates().set_index('book_idx')

title_map = books.set_index('ISBN')['Book-Title']
author_map = books.set_index('ISBN')['Book-Author']
image_map = books.set_index('ISBN')['Image-URL-M']

# ===============================
# DISPLAY RECOMMENDATIONS
# ===============================
st.subheader("✨ Top Recommended Books")

cols = st.columns(5)

for i, (book_idx, score) in enumerate(recs[:10]):
    col = cols[i % 5]

    if book_idx in book_mapping.index:
        isbn = book_mapping.loc[book_idx, 'ISBN']
        title = title_map.get(isbn, "Unknown Title")
        author = author_map.get(isbn, "Unknown Author")
        img = image_map.get(isbn, None)
    else:
        title = "Unknown Title"
        author = ""
        img = None

    with col:
        if img:
            st.image(img, use_container_width=True)
        else:
            st.write("No Image Available")

        st.markdown(f"**{title[:40]}**")
        st.caption(author)
        st.caption(f"⭐ Predicted Score: {score:.2f}")
