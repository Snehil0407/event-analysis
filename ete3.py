import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from PIL import Image
import cv2
import io
import os
from wordcloud import WordCloud
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Event Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    div[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    .stSelectbox label, .stDataFrame label {
        font-weight: 600;
        color: #0e1117;
    }
    div[data-testid="stSidebarNav"] {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    h1, h2, h3 {
        color: #0e1117;
    }
    .stAlert {
        background-color: #ffffff;
        border: none;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def generate_dataset():
    np.random.seed(42)
    colleges = ["College A", "College B", "College C", "College D", "College E"]
    states = ["State X", "State Y", "State Z"]
    tracks = ["AI", "Cybersecurity", "IoT", "Blockchain"]
    days = ["Day 1", "Day 2", "Day 3", "Day 4"]
    
    # Create base feedback list
    feedback_list = [
        "Excellent workshop! The hands-on coding sessions were incredibly helpful.",
        "Great speakers and valuable industry insights. Would love more networking opportunities.",
        "The AI track was fascinating, but some sessions were too advanced for beginners.",
        "Perfect organization and timing. The venue was fantastic!",
        "Interactive sessions were engaging, but could use more practical examples.",
        "Outstanding keynote speakers. Really enjoyed the tech demos.",
        "The blockchain workshop exceeded my expectations. Very informative!",
        "Good event overall, but needed better time management between sessions.",
        "Loved the diversity of topics. The mobile dev track was particularly useful.",
        "Amazing networking opportunities. Met some brilliant developers!",
        "Technical content was solid, but slides could be more detailed.",
        "The cybersecurity panel discussion was eye-opening. Great expertise!",
        "Well-structured workshops, but would appreciate more Q&A time.",
        "The cloud computing track provided excellent real-world applications.",
        "Fantastic learning experience! Looking forward to next year's event."
    ]
    
    # Repeat the feedback list to get exactly 400 entries
    feedback_data = (feedback_list * 27)[:400]  # This will give us exactly 400 entries

    data = {
        "Participant ID": np.arange(1, 401),
        "Name": [f"Student {i}" for i in range(1, 401)],
        "College": np.random.choice(colleges, 400),
        "State": np.random.choice(states, 400),
        "Track": np.random.choice(tracks, 400),
        "Day": np.random.choice(days, 400),
        "Feedback": feedback_data
    }
    # Calculate feedback length after creating the dictionary
    data["Feedback Length"] = [len(feedback) for feedback in data["Feedback"]]
    df = pd.DataFrame(data)
    return df

def main():
    st.title("📊 Event Analytics Dashboard")
    st.markdown("---")

    # Load data
    df = generate_dataset()

    # Create three columns for filters with better styling
    st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
        <h3 style='margin-bottom: 1rem; color: #0e1117;'>📌 Filter Options</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        track_filter = st.selectbox("🎯 Track", ["All"] + list(df["Track"].unique()))
    with col2:
        state_filter = st.selectbox("🗺️ State", ["All"] + list(df["State"].unique()))
    with col3:
        college_filter = st.selectbox("🏫 College", ["All"] + list(df["College"].unique()))
    
    # Apply filters
    filtered_df = df.copy()
    if track_filter != "All":
        filtered_df = filtered_df[filtered_df["Track"] == track_filter]
    if state_filter != "All":
        filtered_df = filtered_df[filtered_df["State"] == state_filter]
    if college_filter != "All":
        filtered_df = filtered_df[filtered_df["College"] == college_filter]
    
    # Display metrics
    st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0;'>
        <h3 style='margin-bottom: 1rem; color: #0e1117;'>📈 Key Metrics</h3>
        </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Participants", len(filtered_df))
    with m2:
        st.metric("Unique Tracks", filtered_df["Track"].nunique())
    with m3:
        st.metric("States Represented", filtered_df["State"].nunique())
    with m4:
        st.metric("Colleges Participating", filtered_df["College"].nunique())

    # Data display with styling
    st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0;'>
        <h3 style='margin-bottom: 1rem; color: #0e1117;'>📋 Detailed Data</h3>
        </div>
    """, unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

    # Visualizations
    st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0;'>
        <h3 style='margin-bottom: 1rem; color: #0e1117;'>📊 Visual Analytics</h3>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("🎯 Track-wise Participation")
        fig1, ax1 = plt.subplots(figsize=(3,2))
        track_counts = filtered_df["Track"].value_counts()
        sns.barplot(x=track_counts.index, y=track_counts.values, palette="viridis", ax=ax1)
        plt.xticks(rotation=45, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("📅 Day-wise Distribution")
        day_counts = filtered_df["Day"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(3,2))
        ax2.pie(day_counts.values, labels=day_counts.index, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(day_counts)))
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("📝 Feedback Length Distribution")
        fig3, ax3 = plt.subplots(figsize=(3,2))
        sns.histplot(data=filtered_df, x="Feedback Length", bins=20, ax=ax3)
        plt.xticks(rotation=45, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("🗺️ State-wise Participation")
        state_counts = filtered_df["State"].value_counts()
        fig4, ax4 = plt.subplots(figsize=(3,2))
        ax4.plot(state_counts.index, state_counts.values, marker="o", linestyle="-", color="orange")
        plt.xticks(rotation=45, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig4)
        st.markdown('</div>', unsafe_allow_html=True)

    # Full width box plot
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("📊 Feedback Length by Track")
    fig5, ax5 = plt.subplots(figsize=(3,2))
    sns.boxplot(data=filtered_df, x="Track", y="Feedback Length", palette="viridis", ax=ax5)
    plt.xticks(rotation=45, fontsize=5)
    plt.tight_layout()
    st.pyplot(fig5)
    st.markdown('</div>', unsafe_allow_html=True)

def save_uploaded_image(uploaded_file, day, track):
    # Use st.session_state to store images in memory during the session
    if 'gallery_images' not in st.session_state:
        st.session_state.gallery_images = {}
    
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = uploaded_file.name.split('.')[-1]
    filename = f"{day}_{track}_{timestamp}.{file_extension}"
    
    # Store the image data in session state
    st.session_state.gallery_images[filename] = {
        'data': uploaded_file.getvalue(),
        'day': day,
        'track': track
    }
    
    return filename

def load_gallery_images(day, track="All Tracks"):
    if 'gallery_images' not in st.session_state:
        return []
    
    images = []
    for filename, image_data in st.session_state.gallery_images.items():
        if image_data['day'] == day:
            if track == "All Tracks" or track in filename:
                images.append(filename)
    return images

def get_image_data(filename):
    if filename in st.session_state.gallery_images:
        return st.session_state.gallery_images[filename]['data']
    return None

def delete_image(filename):
    if filename in st.session_state.gallery_images:
        del st.session_state.gallery_images[filename]

def text_analysis_page():
    st.title("📝 Feedback Analysis")
    
    # Add some spacing and style
    st.markdown("""
        <style>
        .block-container {
            padding: 2rem;
        }
        div[data-testid="stHorizontalBlock"] {
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load and filter data
    df = generate_dataset()
    
    # Create a clean selection box for tracks
    st.markdown("### 🎯 Select Track for Analysis")
    track_options = ["All Tracks"] + list(df["Track"].unique())
    selected_track = st.selectbox("", track_options)  # Empty label as we have a header above
    
    st.markdown("---")  # Add a separator
    
    if selected_track != "All Tracks":
        track_df = df[df["Track"] == selected_track]
    else:
        track_df = df
    
    # Filter feedback for selected track
    if not track_df.empty:
        track_feedback = " ".join(track_df["Feedback"].dropna())
        
        if track_feedback:
            # Key Insights at the top
            st.markdown("### 🔍 Key Metrics")
            
            # Calculate metrics first
            positive_words = ['excellent', 'great', 'good', 'amazing', 'fantastic', 'helpful', 'informative', 'enjoyed', 'perfect', 'loved']
            negative_words = ['improve', 'needed', 'could', 'but', 'however', 'difficult', 'hard', 'complex', 'confusing']
            
            def get_sentiment_score(text):
                words = text.lower().split()
                positive_count = sum(1 for word in words if word in positive_words)
                negative_count = sum(1 for word in words if word in negative_words)
                return positive_count - negative_count
            
            track_df['sentiment_score'] = track_df['Feedback'].apply(get_sentiment_score)
            
            # Word frequency calculation
            from collections import Counter
            import re
            words = re.findall(r'\b\w+\b', track_feedback.lower())
            word_freq = Counter(words)
            common_words = pd.DataFrame(word_freq.most_common(10), columns=['Word', 'Frequency'])
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("Total Feedback", len(track_df))
            with metrics_col2:
                st.metric("Avg Sentiment", f"{track_df['sentiment_score'].mean():.2f}")
            with metrics_col3:
                st.metric("Avg Length", f"{int(track_df['Feedback Length'].mean())}")
            with metrics_col4:
                st.metric("Most Used Word", common_words.iloc[0]['Word'])
            
            st.markdown("---")
            
            # Visual Analysis Section
            st.markdown("### 📊 Visual Analysis")
            
            # First Row - Word Cloud and Sentiment
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🌥️ Word Cloud")
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis'
                ).generate(track_feedback)
                
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### 😊 Sentiment Distribution")
                fig = px.histogram(track_df, x='sentiment_score', 
                                 title='',  # Remove title as we have header above
                                 color_discrete_sequence=['#2ecc71'],
                                 labels={'sentiment_score': 'Sentiment Score',
                                        'count': 'Number of Feedbacks'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Second Row - Word Frequency and Length Analysis
            st.markdown("### 📈 Detailed Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### 🔤 Most Common Words")
                fig = px.bar(common_words, x='Word', y='Frequency',
                            title='',  # Remove title as we have header above
                            color='Frequency',
                            color_continuous_scale='Viridis')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                st.markdown("#### 📏 Feedback Length Distribution")
                fig = px.box(track_df, y='Feedback Length',
                           title='',  # Remove title as we have header above
                           points='all',
                           color_discrete_sequence=['#9b59b6'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Topic Analysis Section
            st.markdown("### 🎯 Topic Analysis")
            topics = {
                'Technical Content': ['code', 'programming', 'technical', 'workshop', 'hands-on'],
                'Organization': ['organization', 'time', 'schedule', 'management'],
                'Networking': ['network', 'connect', 'interaction', 'meet'],
                'Learning': ['learn', 'knowledge', 'understand', 'skill'],
                'Presentation': ['speaker', 'presentation', 'slides', 'demo']
            }
            
            topic_scores = []
            for topic, keywords in topics.items():
                score = sum(1 for word in track_feedback.lower().split() 
                          if word in keywords)
                topic_scores.append({'Topic': topic, 'Mentions': score})
            
            topic_df = pd.DataFrame(topic_scores)
            
            # Center the pie chart
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                fig = px.pie(topic_df, values='Mentions', names='Topic',
                            title='',  # Remove title as we have header above
                            color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ No feedback available for analysis")
    else:
        st.error("❌ No data available for the selected track")

def image_processing_page():
    st.title("📸 Event Image Gallery & Processing")
    
    # Create tabs for gallery and processing
    tab1, tab2 = st.tabs(["Day-wise Gallery", "Image Processing"])
    
    with tab1:
        st.header("Day-wise Image Gallery")
        
        # Day and track selection
        col1, col2 = st.columns(2)
        with col1:
            selected_day = st.selectbox("Select Day", ["Day 1", "Day 2", "Day 3", "Day 4"])
        with col2:
            selected_track = st.selectbox("Select Track", ["All Tracks", "AI", "Cybersecurity", "IoT", "Blockchain"])
        
        # Image upload for gallery
        st.subheader("Add Images to Gallery")
        gallery_upload = st.file_uploader("Upload event images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="gallery_upload")
        
        if gallery_upload:
            day_track = st.selectbox("Select Day and Track for uploaded images", 
                                   [f"{day} - {track}" for day in ["Day 1", "Day 2", "Day 3", "Day 4"] 
                                    for track in ["AI", "Cybersecurity", "IoT", "Blockchain"]])
            
            if st.button("Add to Gallery"):
                day, track = day_track.split(" - ")
                for uploaded_file in gallery_upload:
                    saved_path = save_uploaded_image(uploaded_file, day.replace(" ", "_"), track)
                st.success(f"Images added to {day_track}")
                st.rerun()  # Refresh to show new images
        
        # Display gallery images
        st.subheader(f"Images for {selected_day}")
        
        # Load images for selected day and track
        gallery_images = load_gallery_images(selected_day.replace(" ", "_"), selected_track)
        
        if gallery_images:
            # Create a grid layout
            cols = st.columns(3)
            for idx, image_path in enumerate(gallery_images):
                with cols[idx % 3]:
                    try:
                        image_data = get_image_data(image_path)
                        if image_data:
                            img = Image.open(io.BytesIO(image_data))
                            # Resize image to maintain consistent size
                            img.thumbnail((300, 300))
                            st.image(img, caption=f"{selected_day} - {os.path.basename(image_path).split('_')[1]}")
                        
                            # Add delete button for each image
                            if st.button("🗑️ Delete", key=f"delete_{idx}_{image_path}"):
                                if delete_image(image_path):
                                    st.success("Image deleted successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete image.")
                        
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
        else:
            if 'gallery_images' in st.session_state and len(st.session_state.gallery_images) > 0:
                st.info(f"No images found for {selected_day} - {selected_track}")
            else:
                # Show demo images only if no real images exist
                gallery_cols = st.columns(3)
                for i, col in enumerate(gallery_cols):
                    demo_image = np.zeros((200, 300, 3), dtype=np.uint8)
                    demo_image[:, :] = [200, 200, 200]
                    cv2.putText(demo_image, f"{selected_day}", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    demo_image_pil = Image.fromarray(demo_image)
                    col.image(demo_image_pil, caption=f"Demo Photo {i+1}")
    
    with tab2:
        st.header("Custom Image Processing")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="process_upload")
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image")
            
            # Image processing options
            st.subheader("Processing Options")
            
            # Create three columns for more options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Basic adjustments
                brightness = st.slider("Brightness", -100, 100, 0)
                contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0)
            
            with col2:
                # Filter options
                filter_option = st.selectbox("Apply Filter", 
                    ["None", "Grayscale", "Blur", "Edge Detection", "Sepia", "Sharpen", "Emboss"])
                
                # Additional filter parameters
                if filter_option == "Blur":
                    blur_strength = st.slider("Blur Strength", 1, 15, 5, step=2)
                elif filter_option == "Edge Detection":
                    edge_threshold = st.slider("Edge Threshold", 50, 200, 100)
                elif filter_option == "Sharpen":
                    sharpen_strength = st.slider("Sharpen Strength", 0.0, 2.0, 1.0)
            
            with col3:
                # Advanced options
                rotation = st.slider("Rotation", -180, 180, 0)
                flip_option = st.selectbox("Flip", ["None", "Horizontal", "Vertical"])
                resize_percent = st.slider("Resize %", 10, 200, 100)
            
            # Process image
            try:
                # Convert to numpy array for processing
                img_array = np.array(image)
                
                # Resize image
                if resize_percent != 100:
                    width = int(img_array.shape[1] * resize_percent / 100)
                    height = int(img_array.shape[0] * resize_percent / 100)
                    img_array = cv2.resize(img_array, (width, height))
                
                # Apply rotation
                if rotation != 0:
                    center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
                    matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                    img_array = cv2.warpAffine(img_array, matrix, (img_array.shape[1], img_array.shape[0]))
                
                # Apply flip
                if flip_option == "Horizontal":
                    img_array = cv2.flip(img_array, 1)
                elif flip_option == "Vertical":
                    img_array = cv2.flip(img_array, 0)
                
                # Apply brightness and contrast
                img_array = cv2.convertScaleAbs(img_array, alpha=contrast, beta=brightness)
                
                # Apply saturation
                if saturation != 1.0:
                    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                    hsv[:, :, 1] = hsv[:, :, 1] * saturation
                    img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
                # Apply selected filter
                if filter_option == "Grayscale":
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                elif filter_option == "Blur":
                    img_array = cv2.GaussianBlur(img_array, (blur_strength, blur_strength), 0)
                elif filter_option == "Edge Detection":
                    img_array = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), edge_threshold, edge_threshold * 2)
                elif filter_option == "Sepia":
                    kernel = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
                    img_array = cv2.transform(img_array, kernel)
                elif filter_option == "Sharpen":
                    kernel = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]]) * sharpen_strength
                    img_array = cv2.filter2D(img_array, -1, kernel)
                elif filter_option == "Emboss":
                    kernel = np.array([[-2,-1,0],
                                     [-1, 1,1],
                                     [ 0, 1,2]])
                    img_array = cv2.filter2D(img_array, -1, kernel) + 128
                
                # Display processed image
                st.subheader("Processed Image")
                st.image(img_array, caption=f"Processed with {filter_option}")
                
                # Add download button for processed image
                if len(img_array.shape) == 2:  # If grayscale
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                processed_img = Image.fromarray(img_array)
                buf = io.BytesIO()
                processed_img.save(buf, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

def main_page():
    st.sidebar.title("Navigation")
    pages = {
        "📊 Dashboard": main,
        "📝 Text Analysis": text_analysis_page,
        "📸 Image Gallery": image_processing_page
    }
    
    st.sidebar.markdown("---")
    choice = st.sidebar.radio("Select Page", list(pages.keys()))
    st.sidebar.markdown("---")
    
    # Add info section in sidebar
    st.sidebar.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px;'>
        <h4 style='color: #0e1117;'>ℹ️ About</h4>
        <p style='font-size: 0.9em; color: #6c757d;'>This dashboard provides analytics for event participation and feedback analysis.</p>
        </div>
    """, unsafe_allow_html=True)
    
    pages[choice]()

if __name__ == "__main__":
    main_page()
