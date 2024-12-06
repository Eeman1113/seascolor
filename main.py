import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import colorsys
from sklearn.cluster import KMeans
import io

def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space."""
    return colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)

def classify_undertone(h, s, v):
    """
    Classify skin undertones based on HSV color space.
    This is a simplified approximation and should not be used for precise skin tone analysis.
    """
    if 0.05 <= h <= 0.15:  # Warm orange-yellow range
        return 'Warm'
    elif 0.55 <= h <= 0.75:  # Cool blue-purple range
        return 'Cool'
    else:
        return 'Neutral'

def classify_seasonal_type(h, s, v):
    """
    Classify seasonal color types based on HSV values.
    This is a highly simplified approximation.
    """
    if 0.05 <= h <= 0.15 and s > 0.5 and v > 0.5:
        return 'Spring'
    elif 0.15 <= h <= 0.35 and s < 0.5 and v > 0.5:
        return 'Summer'
    elif 0.35 <= h <= 0.55 and s > 0.5 and v > 0.5:
        return 'Autumn'
    elif (h <= 0.05 or h >= 0.75) and s < 0.3 and v < 0.5:
        return 'Winter'
    else:
        return 'Undefined'

def analyze_image_colors(img):
    """
    Analyze colors in an image, create visualizations, and perform clustering.
    """
    # Convert image to RGB if needed
    img = img.convert('RGB')
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Reshape the image to be a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(pixels)
    
    # Get cluster centers and labels
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Convert RGB to HSV for further analysis
    hsv_colors = np.array([rgb_to_hsv(color) for color in colors])
    
    # Classify undertones and seasonal types
    undertones = [classify_undertone(h, s, v) for h, s, v in hsv_colors]
    seasonal_types = [classify_seasonal_type(h, s, v) for h, s, v in hsv_colors]
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Color Gradient Plot
    plt.subplot(2, 2, 1)
    luminance = np.dot(colors, [0.299, 0.587, 0.114])
    sorted_indices = np.argsort(luminance)
    sorted_colors = colors[sorted_indices] / 255.0
    
    plt.imshow(sorted_colors[np.newaxis, :], aspect='auto', extent=[0, 1, 0, 1])
    plt.title('Color Gradient (Sorted by Luminance)')
    plt.xlabel('Color Progression')
    plt.xticks([])
    plt.yticks([])
    
    # 2. Color Bar Plot
    plt.subplot(2, 2, 2)
    cluster_counts = np.unique(labels, return_counts=True)[1]
    plt.bar(range(len(colors)), cluster_counts, color=colors/255)
    plt.title('Color Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Pixel Count')
    
    # 3. Pie Chart of Undertones
    plt.subplot(2, 2, 3)
    undertone_counts = np.unique(undertones, return_counts=True)
    plt.pie(undertone_counts[1], labels=undertone_counts[0], autopct='%1.1f%%')
    plt.title('Undertone Distribution')
    
    # 4. Pie Chart of Seasonal Types
    plt.subplot(2, 2, 4)
    seasonal_counts = np.unique(seasonal_types, return_counts=True)
    plt.pie(seasonal_counts[1], labels=seasonal_counts[0], autopct='%1.1f%%')
    plt.title('Seasonal Type Distribution')
    
    plt.tight_layout()
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Prepare color cluster details
    cluster_details = []
    for i, (color, undertone, season) in enumerate(zip(colors, undertones, seasonal_types)):
        cluster_details.append({
            'Cluster': i,
            'RGB Color': color,
            'Undertone': undertone,
            'Seasonal Type': season
        })
    
    return buf, cluster_details

def main():
    st.title('Image Color Analysis Tool')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Analyze colors
        st.write('Analyzing image colors...')
        plot_buffer, cluster_details = analyze_image_colors(image)
        
        # Display analysis plot
        st.image(plot_buffer, caption='Color Analysis Visualization')
        
        # Display cluster details
        st.subheader('Color Cluster Details')
        for cluster in cluster_details:
            st.write(f"**Cluster {cluster['Cluster']}:**")
            st.write(f"  - RGB Color: {cluster['RGB Color']}")
            st.write(f"  - Undertone: {cluster['Undertone']}")
            st.write(f"  - Seasonal Type: {cluster['Seasonal Type']}")

if __name__ == '__main__':
    main()
