import streamlit as st
import asyncio
import json
import pandas as pd
import os

from extractor import crawl_and_extract

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------------------
# App Title and Description
# ------------------------------
st.title("📰 AI News Summarizer")
st.markdown("""
This app extracts and summarizes news articles automatically using **Crawl4AI** and **Hugging Face Transformers**.  
Simply paste a news article URL and click **Summarize** to get:
- Headline / Title  
- Body Text (main story)  
- Author  
- Publication Date & Time  
- Category  
- Tags / Keywords  
- AI-Generated Summary
""")

# ------------------------------
# Input Section
# ------------------------------
url = st.text_input("🔗 Enter a news article URL:")

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = []

# ------------------------------
# Summarize Button
# ------------------------------
if st.button("🚀 Summarize Article"):
    if not url.strip():
        st.warning("⚠️ Please enter a valid URL.")
    else:
        with st.spinner("Fetching and summarizing article... This may take a moment ⏳"):
            try:
                data = asyncio.run(crawl_and_extract(url))
                st.session_state.results.append(data)

                # Display extracted information
                st.success("✅ Article processed successfully!")
                st.subheader("🧩 Extracted Information")

                st.write(f"**Headline:** {data.get('Headline', 'N/A')}")
                st.write(f"**Author:** {data.get('Author', 'N/A')}")
                st.write(f"**Published:** {data.get('Publication date & time', 'N/A')}")
                st.write(f"**Category:** {data.get('Category', 'N/A')}")
                st.write(f"**Tags:** {', '.join(data.get('Tags', [])) if data.get('Tags') else 'N/A'}")

                st.subheader("📝 Summary / Excerpt")
                st.write(data.get("Excerpt", "No summary available."))

                with st.expander("📜 Full Article Text"):
                    st.text_area("Full Text", data.get("Body", ""), height=300)

            except Exception as e:
                st.error(f"❌ Error occurred while processing: {e}")

# ------------------------------
# Display Stored Results
# ------------------------------
if st.session_state.results:
    st.divider()
    st.subheader("🧾 Processed Articles History")

    df = pd.DataFrame(st.session_state.results)
    st. dataframe(df[["Headline", "Author", "Publication date & time", "Category", "Excerpt"]])

    # Save options
    st.markdown("### 💾 Save Data")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download as CSV"):
            csv_path = "summarized_articles.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            with open(csv_path, "rb") as f:
                st.download_button("⬇️ Download CSV", f, file_name="summarized_articles.csv")

    with col2:
        if st.button("Download as JSON"):
            json_path = "summarized_articles.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.results, f, indent=2, ensure_ascii=False)
            with open(json_path, "rb") as f:
                st.download_button("⬇️ Download JSON", f, file_name="summarized_articles.json")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("🧠 Built by Mian Muhammad Usman — powered by Crawl4AI + Hugging Face Transformers.")

