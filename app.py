import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from config import SENTENCE_TRANSFORMER_MODEL, SENTIMENT_MODEL
import numpy as np
import os

# --- CRITICAL: Streamlit Cloud Auto-Install Playwright Hooks ---
@st.cache_resource(show_spinner="⚙️ Preparing Headless Scraping Engine...")
def ensure_playwright_browsers():
    """Forces the Playwright Chromium binaries to install if they are missing on deployment."""
    os.system("playwright install chromium")
    os.system("playwright install-deps chromium")
    return True

# Ensure this runs exactly once on Boot when the app restarts on Streamlit servers
ensure_playwright_browsers()

@st.cache_resource(show_spinner="⏳ Warming up heavy NLP Models (One-time load)...")
def get_analyzer():
    from semantic_analyzer import SemanticReviewAnalyzer
    return SemanticReviewAnalyzer()

# Page configuration
st.set_page_config(
    page_title="Semantic Product Review Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern dark theme
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1428 100%);
        color: #e0e0e0;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
        border-bottom: 1px solid rgba(100, 200, 255, 0.1);
    }
    
    .main {
        max-width: 1400px;
        padding: 2rem;
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        min-height: 200px;
        background-color: rgba(20, 30, 60, 0.8) !important;
        border: 1px solid rgba(100, 200, 255, 0.3) !important;
        color: #e0e0e0 !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(100, 200, 255, 0.8) !important;
        box-shadow: 0 0 20px rgba(100, 200, 255, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.5);
    }
    
    /* Sentiment colors */
    .sentiment-positive {
        color: #00ff88;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .sentiment-negative {
        color: #ff3366;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
    }
    
    .sentiment-neutral {
        color: #ffaa00;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 170, 0, 0.5);
    }
    
    /* Aspect card styling */
    .aspect-card {
        background: linear-gradient(135deg, rgba(20, 50, 100, 0.6) 0%, rgba(30, 60, 120, 0.4) 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100, 200, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .aspect-card:hover {
        border-color: rgba(100, 200, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        transform: translateX(4px);
    }
    
    .aspect-card h4 {
        color: #00d4ff;
        margin-bottom: 0.5rem;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .aspect-card p {
        color: #b0b0b0;
        margin: 0.5rem 0;
        font-size: 14px;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    /* Main title - Improved visibility */
    h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff !important;
        text-shadow: 
            0 0 8px rgba(0, 180, 255, 0.9),
            0 0 16px rgba(0, 120, 255, 0.6);
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
        line-height: 1.3;
        position: relative;
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        background: rgba(10, 14, 39, 0.4);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(100, 200, 255, 0.15);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Add a subtle glow effect on hover */
    h1:hover {
        text-shadow: 
            0 0 12px rgba(0, 200, 255, 1),
            0 0 24px rgba(0, 150, 255, 0.8);
        box-shadow: 0 6px 25px rgba(0, 150, 255, 0.3);
    }
    
    h2 {
        font-size: 1.8rem;
        color: #00d4ff;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.3rem;
        color: #00ff88;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(20, 50, 100, 0.4) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(100, 200, 255, 0.2) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(20, 50, 100, 0.6) !important;
        border-color: rgba(100, 200, 255, 0.4) !important;
    }
    
    /* Metric styling */
    .metric-box {
        background: linear-gradient(135deg, rgba(20, 50, 100, 0.6) 0%, rgba(30, 60, 120, 0.4) 100%);
        border: 1px solid rgba(100, 200, 255, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #b0b0b0;
        font-size: 14px;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100, 200, 255, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Warning/Info boxes */
    .stWarning, .stInfo {
        background-color: rgba(255, 170, 0, 0.1) !important;
        border-left: 4px solid #ffaa00 !important;
        border-radius: 8px !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 30, 60, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 212, 255, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 212, 255, 0.8);
    }
    </style>
""", unsafe_allow_html=True)


def display_sentiment_emoji(sentiment):
    """Return emoji based on sentiment."""
    sentiment_lower = sentiment.lower()
    
    # Handle granular labels
    if "very positive" in sentiment_lower:
        return "🌟"
    elif "highly positive" in sentiment_lower:
        return "✨"
    elif "moderately positive" in sentiment_lower:
        return "😊"
    elif "slightly positive" in sentiment_lower:
        return "🙂"
    elif "leaning positive" in sentiment_lower:
        return "😐+"
    elif "leaning negative" in sentiment_lower:
        return "😐-"
    elif "slightly negative" in sentiment_lower:
        return "😕"
    elif "moderately negative" in sentiment_lower:
        return "😟"
    elif "highly negative" in sentiment_lower:
        return "⚠️"
    elif "very negative" in sentiment_lower:
        return "❌"
    # Fallback to basic categories
    elif "positive" in sentiment_lower:
        return "✨"
    elif "negative" in sentiment_lower:
        return "⚠️"
    elif "mixed" in sentiment_lower:
        return "🔀"
    return "◆"

def get_sentiment_color(sentiment):
    """Return color based on sentiment with gradient support."""
    sentiment_lower = sentiment.lower()
    
    # Granular color mapping (gradient from red to green)
    if "very positive" in sentiment_lower:
        return "#00ff88"  # Vibrant green
    elif "highly positive" in sentiment_lower:
        return "#33ff99"
    elif "moderately positive" in sentiment_lower:
        return "#66ffaa"
    elif "slightly positive" in sentiment_lower:
        return "#99ffbb"
    elif "leaning positive" in sentiment_lower:
        return "#ccffcc"
    elif "leaning negative" in sentiment_lower:
        return "#ffcccc"
    elif "slightly negative" in sentiment_lower:
        return "#ffaaaa"
    elif "moderately negative" in sentiment_lower:
        return "#ff8888"
    elif "highly negative" in sentiment_lower:
        return "#ff5555"
    elif "very negative" in sentiment_lower:
        return "#ff3366"  # Vibrant red
    # Fallback
    elif "positive" in sentiment_lower:
        return "#00ff88"
    elif "negative" in sentiment_lower:
        return "#ff3366"
    elif "mixed" in sentiment_lower:
        return "#9966ff"
    return "#ffaa00"  # Neutral orange

def get_sentiment_color_by_score(score):
    """Return color based on numerical score with smooth gradient."""
    # Interpolate between red (0) and green (1)
    if score >= 0.5:
        # Positive range: yellow to green
        t = (score - 0.5) * 2  # 0 to 1
        r = int(255 * (1 - t * 0.7))  # 255 to ~76
        g = 255
        b = int(136 * (1 - t * 0.5))  # 136 to ~68
    else:
        # Negative range: red to yellow
        t = score * 2  # 0 to 1
        r = 255
        g = int(51 + 204 * t)  # 51 to 255
        b = int(102 * (1 - t * 0.6))  # 102 to ~40
    
    return f"#{r:02x}{g:02x}{b:02x}"

def create_wordcloud(text):
    """Generate a word cloud from text."""
    if not text.strip():
        return None
        
    # Set the background color to transparent
    plt.rcParams['savefig.facecolor'] = 'none'
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=None,  # Set to None for transparent background
        mode='RGBA',  # Use RGBA mode for transparency
        max_words=100,
        colormap='cool',  # Using cool colormap for better visibility
        prefer_horizontal=0.7,
        max_font_size=150,
        min_font_size=10,
        contour_width=0
    ).generate(text)
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_alpha(0)
    
    # Display the generated word cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Adjust layout to remove any extra whitespace
    plt.tight_layout(pad=0)
    
    return fig

def create_sentiment_gauge(score, sentiment):
    """Create an enhanced sentiment gauge chart."""
    if sentiment == 'negative':
        score = 1 - score
    
    fig = go.Figure(data=[
        go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score", 'font': {'size': 20, 'color': '#00d4ff'}},
            number={'font': {'size': 32, 'color': get_sentiment_color(sentiment)}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'rgba(100, 200, 255, 0.3)'},
                'bar': {'color': get_sentiment_color(sentiment)},
                'bgcolor': 'rgba(20, 50, 100, 0.3)',
                'borderwidth': 2,
                'bordercolor': 'rgba(100, 200, 255, 0.5)',
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(255, 51, 102, 0.2)'},
                    {'range': [33, 66], 'color': 'rgba(255, 170, 0, 0.2)'},
                    {'range': [66, 100], 'color': 'rgba(0, 255, 136, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': '#00d4ff', 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(10, 14, 39, 0)',
        plot_bgcolor='rgba(10, 14, 39, 0)',
        font={'color': '#e0e0e0', 'family': 'Arial, sans-serif'},
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_sentiment_spectrum(score, granular_label):
    """Create a horizontal sentiment spectrum bar showing position."""
    fig = go.Figure()
    
    # Add background spectrum bars (10 tiers)
    tiers = [
        (0.0, 0.1, "Very Negative", "#ff3366"),
        (0.1, 0.2, "Highly Negative", "#ff5555"),
        (0.2, 0.3, "Moderately Negative", "#ff8888"),
        (0.3, 0.4, "Slightly Negative", "#ffaaaa"),
        (0.4, 0.5, "Leaning Negative", "#ffcccc"),
        (0.5, 0.6, "Leaning Positive", "#ccffcc"),
        (0.6, 0.7, "Slightly Positive", "#99ffbb"),
        (0.7, 0.8, "Moderately Positive", "#66ffaa"),
        (0.8, 0.9, "Highly Positive", "#33ff99"),
        (0.9, 1.0, "Very Positive", "#00ff88"),
    ]
    
    for start, end, label, color in tiers:
        fig.add_trace(go.Bar(
            x=[end - start],
            y=["Sentiment"],
            orientation='h',
            name=label,
            marker=dict(color=color),
            text=label,
            textposition='inside',
            textfont=dict(size=9, color='#000'),
            base=start,
            hovertemplate=f"{label}: {start*100:.0f}%-{end*100:.0f}%<extra></extra>",
            showlegend=False
        ))
    
    # Add marker for current sentiment
    fig.add_trace(go.Scatter(
        x=[score],
        y=["Sentiment"],
        mode='markers+text',
        marker=dict(
            size=20,
            color='#ffffff',
            symbol='diamond',
            line=dict(color='#000', width=2)
        ),
        text=[f"{score*100:.1f}%"],
        textposition="top center",
        textfont=dict(size=14, color='#00d4ff', family='Arial Black'),
        name='Current',
        hovertemplate=f"{granular_label}: {score*100:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(10, 14, 39, 0)',
        plot_bgcolor='rgba(10, 14, 39, 0)',
        font={'color': '#e0e0e0'},
        height=150,
        margin=dict(l=0, r=0, t=20, b=20),
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    )
    
    return fig

def display_humor_analysis(humor_data):
    """Display humor and sarcasm detection results."""
    if not humor_data.get('has_humor'):
        return
    
    st.markdown("### 😄 Humor Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        humor_type = humor_data.get('humor_type', 'none').title()
        confidence = humor_data.get('confidence', 0)
        
        st.markdown(f"""
        <div class="aspect-card" style="border-left-color: #9966ff;">
            <h4>🎭 {humor_type} Detected</h4>
            <p><strong>Confidence:</strong> {confidence:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show indicators
        indicators = humor_data.get('indicators', [])
        if indicators:
            st.markdown("**Indicators:**")
            for indicator in indicators:
                st.markdown(f"- {indicator}")
    
    with col2:
        # Humor confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Humor Level", 'font': {'size': 16, 'color': '#9966ff'}},
            number={'suffix': "%", 'font': {'size': 24, 'color': '#9966ff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'rgba(153, 102, 255, 0.3)'},
                'bar': {'color': '#9966ff'},
                'bgcolor': 'rgba(20, 50, 100, 0.3)',
                'borderwidth': 2,
                'bordercolor': 'rgba(153, 102, 255, 0.5)',
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(10, 14, 39, 0)',
            plot_bgcolor='rgba(10, 14, 39, 0)',
            font={'color': '#e0e0e0'},
            height=200,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_contradictions(contradiction_data):
    """Display contradictory statement analysis."""
    if not contradiction_data.get('has_contradictions'):
        return
    
    st.markdown("### ⚖️ Contradictory Statements")
    
    pairs = contradiction_data.get('contradictory_pairs', [])
    contradiction_score = contradiction_data.get('contradiction_score', 0)
    
    st.markdown(f"""
    <div class="metric-box" style="background: linear-gradient(135deg, rgba(255, 153, 0, 0.2) 0%, rgba(255, 102, 0, 0.1) 100%);">
        <div class="metric-value" style="background: linear-gradient(135deg, #ff9900 0%, #ff6600 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {contradiction_score:.0%}
        </div>
        <div class="metric-label">Contradiction Strength</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    for idx, pair in enumerate(pairs, 1):
        marker = pair['marker'].upper()
        before_sent = pair['before_sentiment']
        after_sent = pair['after_sentiment']
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(20, 50, 100, 0.3); padding: 1rem; border-radius: 8px; border-left: 3px solid {get_sentiment_color_by_score(before_sent['score'])};">
                <p style="color: #b0b0b0; margin: 0;">{pair['before']}</p>
                <p style="margin: 0.5rem 0 0 0;"><span style="color: {get_sentiment_color_by_score(before_sent['score'])};">
                    {before_sent['label'].upper()} ({before_sent['score']:.0%})
                </span></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0;">
                <strong style="color: #ff9900; font-size: 1.2rem;">{marker}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(20, 50, 100, 0.3); padding: 1rem; border-radius: 8px; border-left: 3px solid {get_sentiment_color_by_score(after_sent['score'])};">
                <p style="color: #b0b0b0; margin: 0;">{pair['after']}</p>
                <p style="margin: 0.5rem 0 0 0;"><span style="color: {get_sentiment_color_by_score(after_sent['score'])};">
                    {after_sent['label'].upper()} ({after_sent['score']:.0%})
                </span></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

def create_aspect_sentiment_chart(aspects):
    """Create a horizontal bar chart showing sentiment for each aspect."""
    if not aspects:
        return None
    
    # Prepare data
    aspect_names = [a['aspect'].title() for a in aspects]
    scores = [a['score'] * 100 for a in aspects]
    colors = [get_sentiment_color_by_score(a['score']) for a in aspects]
    granular_labels = [a.get('granular_sentiment', 'Unknown') for a in aspects]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=aspect_names,
        x=scores,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
        ),
        text=[f"{s:.1f}%" for s in scores],
        textposition='outside',
        textfont=dict(size=12, color='#e0e0e0'),
        hovertemplate='<b>%{y}</b><br>Sentiment: %{customdata}<br>Score: %{x:.1f}%<extra></extra>',
        customdata=granular_labels
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(10, 14, 39, 0)',
        plot_bgcolor='rgba(10, 14, 39, 0)',
        font={'color': '#e0e0e0'},
        height=max(300, len(aspects) * 50),
        margin=dict(l=0, r=80, t=20, b=40),
        xaxis=dict(
            title="Sentiment Score (%)",
            range=[0, 100],
            showgrid=True,
            gridcolor='rgba(100, 200, 255, 0.1)',
            zeroline=True,
            zerolinecolor='rgba(255, 170, 0, 0.5)'
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed'
        )
    )
    
    return fig

def display_aspect_analysis(aspects):
    """Display aspect-based analysis results."""
    if not aspects:
        st.warning("No specific aspects detected in the review.")
        return
    
    st.subheader("🔍 Aspect-Based Analysis")
    
    # Show aspect sentiment chart
    st.markdown("### Aspect Sentiment Breakdown")
    aspect_chart = create_aspect_sentiment_chart(aspects)
    if aspect_chart:
        st.plotly_chart(aspect_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Detected Aspects")
        for idx, aspect in enumerate(aspects, 1):
            sentiment = aspect.get('sentiment', 'neutral')
            granular = aspect.get('granular_sentiment', sentiment.title())
            score = aspect.get('score', 0.5)
            confidence = aspect.get('confidence', 0.0)
            occurrences = aspect.get('occurrences', 1)
            
            emoji = display_sentiment_emoji(granular)
            color = get_sentiment_color(granular)
            
            st.markdown(f"""
            <div class="aspect-card" style="border-left-color: {color};">
                <h4>#{idx} {aspect['aspect'].title()} {emoji}</h4>
                <p><strong>Sentiment:</strong> <span style="color: {color};">
                    {granular} ({score:.1%})
                </span></p>
                <p><strong>Confidence:</strong> {confidence:.0%} | <strong>Occurrences:</strong> {occurrences}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Sentiment Distribution")
        # Sentiment distribution pie chart
        if aspects:
            sentiment_counts = {}
            for aspect in aspects:
                granular = aspect.get('granular_sentiment', 'Unknown')
                sentiment_counts[granular] = sentiment_counts.get(granular, 0) + 1
            
            if sentiment_counts:
                df_sentiment = pd.DataFrame({
                    'Sentiment': list(sentiment_counts.keys()),
                    'Count': list(sentiment_counts.values())
                })
                
                fig = px.pie(
                    df_sentiment,
                    values='Count',
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={
                        'positive': '#00ff88',
                        'neutral': '#ffaa00',
                        'negative': '#ff3366'
                    }
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(10, 14, 39, 0)',
                    plot_bgcolor='rgba(10, 14, 39, 0)',
                    font={'color': '#e0e0e0'},
                    showlegend=True,
                    height=350,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont={'size': 12, 'color': '#000'}
                )
                
                st.plotly_chart(fig, use_container_width=True)

import asyncio
from scraper_service import ScraperService
from verdict_engine import verdict_engine

def display_dashboard(aggregated_data):
    """Display the unified dashboard for aggregated reviews."""
    if not aggregated_data:
        st.warning("No data to display.")
        return
        
    unified_score = aggregated_data['unified_score']
    verdict = aggregated_data['verdict']
    recent_trends = aggregated_data.get('recent_trends', {})
    platform_stats = aggregated_data.get('platform_stats', {})
    
    st.markdown("---")
    st.header("📊 Unified Product Dashboard")
    
    if recent_trends.get('has_issue'):
        st.error(f"⚠️ **RECENT BATCH ALERT**: {recent_trends['message']} (Recent Score: {recent_trends['recent_score']:.2f} vs Historical: {recent_trends['historical_score']:.2f})")
    elif 'message' in recent_trends and "tracking closely" in recent_trends['message'].lower():
        st.success(f"✅ **Consistent Quality**: {recent_trends['message']}")
        
    if platform_stats.get('discrepancy_alert'):
        st.error(f"🚨 **CROSS-PLATFORM WARNING**: {platform_stats['discrepancy_alert']}")
        
    # Hero Section: Verdict & Score
    col1, col2 = st.columns([2, 1])
    
    # Color logic for verdict
    if unified_score >= 0.65:
        verdict_color = "#00ff88"
    elif unified_score >= 0.5:
        verdict_color = "#ffaa00"
    else:
        verdict_color = "#ff3366"
        
    with col1:
        st.markdown(f"""
        <div style="padding: 2rem; border-radius: 12px; background: rgba(20, 50, 100, 0.4); border-left: 5px solid {verdict_color};">
            <h2 style="margin-top: 0; color: #e0e0e0;">Final Verdict</h2>
            <h1 style="color: {verdict_color}; margin: 0; font-size: 3rem; text-shadow: none;">{verdict}</h1>
            <p style="color: #b0b0b0; font-size: 1.1rem; margin-top: 0.5rem;">Based on {aggregated_data['total_reviews']} analyzed reviews</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-box" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div class="metric-value" style="font-size: 4rem;">{unified_score:.0%}</div>
            <div class="metric-label">Combined Rating</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Cross Platform Split Ratings
    amz_score = platform_stats.get('amazon_score')
    fk_score = platform_stats.get('flipkart_score')
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_amz, col_fk = st.columns(2)
    
    if amz_score is not None:
        with col_amz:
            st.markdown(f"""
            <div style="padding: 1.5rem; border-radius: 8px; background: rgba(255, 153, 0, 0.1); border: 1px solid rgba(255, 153, 0, 0.3); text-align: center;">
                <h3 style="color: #ff9900; margin-bottom: 0.5rem;">Amazon Rating</h3>
                <h2 style="font-size: 2.5rem; margin: 0; color: #e0e0e0;">{amz_score:.0%}</h2>
                <p style="color: #aaa; margin-top: 0.5rem;">{platform_stats['amazon_count']} Reviews Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
            
    if fk_score is not None:
        with col_fk:
            st.markdown(f"""
            <div style="padding: 1.5rem; border-radius: 8px; background: rgba(40, 116, 240, 0.1); border: 1px solid rgba(40, 116, 240, 0.3); text-align: center;">
                <h3 style="color: #2874f0; margin-bottom: 0.5rem;">Flipkart Rating</h3>
                <h2 style="font-size: 2.5rem; margin: 0; color: #e0e0e0;">{fk_score:.0%}</h2>
                <p style="color: #aaa; margin-top: 0.5rem;">{platform_stats['flipkart_count']} Reviews Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Recent Sentiment Trend Line
    if recent_trends.get('trend_data') and len(recent_trends['trend_data']) > 1:
        st.subheader("📈 Recent Sentiment Trend")
        
        trend_df = pd.DataFrame(recent_trends['trend_data'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df['date'],
            y=trend_df['score'] * 100,
            mode='lines+markers',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=8, color='#0a0e27', line=dict(color='#00ff88', width=2)),
            hovertemplate='Date: %{x}<br>Score: %{y:.1f}%<extra></extra>'
        ))
        
        # Add historical average line for comparison
        if 'historical_score' in recent_trends:
            fig.add_hline(
                y=recent_trends['historical_score'] * 100, 
                line_dash="dash", 
                line_color="#ffaa00",
                annotation_text="Historical Avg",
                annotation_position="bottom right"
            )
            
        fig.update_layout(
            paper_bgcolor='rgba(10, 14, 39, 0)',
            plot_bgcolor='rgba(10, 14, 39, 0.4)',
            font={'color': '#e0e0e0'},
            height=250,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor='rgba(100, 200, 255, 0.1)'),
            yaxis=dict(
                title="Sentiment (%)", 
                range=[0, 100], 
                showgrid=True, 
                gridcolor='rgba(100, 200, 255, 0.1)'
            ),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
    
    # Pros and Cons
    st.subheader("💡 Top Pros & Cons")
    pro_col, con_col = st.columns(2)
    
    with pro_col:
        st.markdown("<h3 style='color: #00ff88;'>👍 Pros (Most Liked)</h3>", unsafe_allow_html=True)
        if aggregated_data['pros']:
            for pro in aggregated_data['pros']:
                st.markdown(f"""
                <div class="aspect-card" style="border-left-color: #00ff88; padding: 1rem; margin: 0.5rem 0;">
                    <strong style="font-size: 1.1rem; color: #e0e0e0;">{pro['aspect']}</strong> 
                    <span style="color: #00ff88;">({pro['score']:.0%})</span>
                    <br><span style="font-size: 0.9rem; color: #aaa;">Mentions: {max(pro['positive_mentions'], pro['total_mentions'])}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No standout positive features identified.")
            
    with con_col:
        st.markdown("<h3 style='color: #ff3366;'>👎 Cons (Most Disliked)</h3>", unsafe_allow_html=True)
        if aggregated_data['cons']:
            for con in aggregated_data['cons']:
                st.markdown(f"""
                <div class="aspect-card" style="border-left-color: #ff3366; padding: 1rem; margin: 0.5rem 0;">
                    <strong style="font-size: 1.1rem; color: #e0e0e0;">{con['aspect']}</strong> 
                    <span style="color: #ff3366;">({con['score']:.0%})</span>
                    <br><span style="font-size: 0.9rem; color: #aaa;">Mentions: {max(con['negative_mentions'], con['total_mentions'])}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No standout negative features identified.")
            
    st.markdown("---")
    
    # General Extracted Aspects Chart
    st.subheader("🎯 Feature Rankings")
    tags = aggregated_data['ranked_tags']
    if tags:
        df = pd.DataFrame(tags)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df['aspect'],
            x=df['score'] * 100,
            orientation='h',
            marker=dict(
                color=[get_sentiment_color_by_score(s) for s in df['score']],
                line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
            ),
            text=[f"{s*100:.1f}%" for s in df['score']],
            textposition='outside',
            textfont=dict(color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(10, 14, 39, 0)',
            plot_bgcolor='rgba(10, 14, 39, 0)',
            font={'color': '#e0e0e0'},
            height=max(300, len(tags) * 50),
            margin=dict(l=0, r=80, t=20, b=40),
            xaxis=dict(
                title="Sentiment Score (%)",
                range=[0, 100],
                showgrid=True,
                gridcolor='rgba(100, 200, 255, 0.1)',
            ),
            yaxis={'autorange': 'reversed'}
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit app function."""
    analyzer = get_analyzer()
    # Header section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("✨ Semantic Product Review Analyzer")
        st.markdown("*Live Web Scraper & Contextual NLP Aggregator*")
    
    st.markdown("---")
    
    # Model info in expander
    with st.expander("ℹ️ About This Tool", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🤖 Technology Stack
            This tool leverages state-of-the-art NLP models to provide deep insights into product reviews.
            
            **Models Used:**
            - **Embeddings:** `all-MiniLM-L6-v2` 
            - **Sentiment:** `distilbert-base-uncased-finetuned-sst-2-english` 
            - **NLP:** `spaCy` with custom aspect extraction
            - **Scraping:** `Playwright`
            """)
        with col2:
            st.markdown("""
            ### 🔄 Analysis Pipeline
            1. **Live Scraping** - Fetches reviews from Amazon (more sites coming soon).
            2. **Aspect Extraction** - Identifying key product aspects.
            3. **Granular Sentiment Analysis** - 10-tier classification (0-100%).
            4. **Aggregation** - Combining batch results into Pros & Cons.
            5. **Final Verdict** - Actionable buying advice.
            """)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🔗 URL Aggregator (Live Feed)", "📝 Single Review Analyzer"])
    
    with tab1:
        st.header("🌐 Enter Amazon Product URL")
        product_url = st.text_input(
            "Product URL",
            placeholder="e.g. https://www.amazon.in/dp/B08L5T...",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, _ = st.columns([1, 1, 2])
        with col_btn1:
            url_analyze_btn = st.button("🚀 Scrape & Analyze", type="primary", use_container_width=True)
            
        if url_analyze_btn and product_url.strip():
            with st.status("🔄 Live Analysis in Progress...", expanded=True) as status:
                st.write("🌐 Deploying Playwright Scraper to Amazon... (Fetching Diverse Reviews)")
                
                # Import necessary cross-platform modules
                import urllib.parse
                from product_matcher import ProductMatcher
                
                scraper = ScraperService()
                
                import sys
                if sys.platform == 'win32':
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                # Create a new event loop to run async playwright code
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # 1. Scrape Amazon
                scrape_result = loop.run_until_complete(scraper.scrape_amazon_reviews(product_url))
                amazon_reviews = scrape_result.get('reviews', [])
                product_title = scrape_result.get('product_title', "")
                
                # 2. Extract Title & Find Flipkart Match
                if not product_title:
                    try:
                        import requests
                        from bs4 import BeautifulSoup
                        import re
                        
                        # Out-of-band network fetch for pristine title tag
                        headers = {
                            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'en-IN,en-US;q=0.9,en;q=0.8'
                        }
                        resp = requests.get(product_url, headers=headers, timeout=10)
                        soup = BeautifulSoup(resp.content, 'html.parser')
                        if soup.title and "Amazon" in soup.title.text:
                            raw_title = soup.title.text.split(':')[-1].replace('Customer reviews', '').strip()
                            if len(raw_title) > 5 and raw_title.lower() not in ['electronics', 'computers', 'smartphones', 'clothing']:
                                product_title = raw_title
                                
                        # Final Fallback: URL Path string parser
                        if not product_title:
                            parsed = urllib.parse.urlparse(product_url)
                            path = parsed.path.strip('/')
                            if '/dp/' in parsed.path or '/product/' in parsed.path:
                                raw_path = path.split('/')[0].replace('-', ' ')
                                if raw_path.lower() not in ['electronics', 'computers', 'smartphones', 'clothing']:
                                    product_title = re.sub(r'(?i)\b(storage|subwoofer|environment|combo|bundle|deals|deal|mobile)\b', '', raw_path).strip()
                    except Exception as e:
                        st.write("Could not parse product title for cross-platform matching.")
                        
                flipkart_reviews = []
                if product_title:
                    st.write(f"🔍 Searching Yahoo for Flipkart equivalent: '{product_title}'...")
                    matcher = ProductMatcher()
                    flipkart_url = loop.run_until_complete(matcher.get_flipkart_url(product_title))
                    
                    if flipkart_url:
                        st.write("📦 Found Flipkart Match! Deploying Scraper to Flipkart...")
                        flipkart_reviews = loop.run_until_complete(scraper.scrape_flipkart_reviews(flipkart_url))
                    else:
                        st.write("⚠️ Could not locate corresponding Flipkart product.")
                
                loop.close()
                
                # 3. Combine reviews
                reviews = amazon_reviews + flipkart_reviews
                
                if not reviews:
                    status.update(label="❌ Scraping Failed", state="error", expanded=True)
                    st.error("Could not fetch reviews. Amazon may have triggered a Captcha block, or the URL is invalid.")
                else:
                    st.write(f"✅ Successfully scraped {len(reviews)} reviews.")
                    
                    from filter_service import FilterService
                    st.write("🛡️ Running Authenticity Filter (Removing bots/duplicates)...")
                    filter_engine = FilterService()
                    reviews, filtered_count = filter_engine.filter_reviews(reviews)
                    
                    if filtered_count > 0:
                        st.info(f"🗑️ Filtered out {filtered_count} suspected fake/duplicate reviews.")
                    
                    st.write(f"🧠 Running deep semantic inference on {len(reviews)} authentic reviews...")
                    
                    analyzed_reviews = analyzer.analyze_batch(reviews)
                    
                    st.write("✨ Generating Insights & Verdict...")
                    aggregated_data = verdict_engine.aggregate_batch(analyzed_reviews)
                    
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    
            if reviews:
                display_dashboard(aggregated_data)
                
                # Optional: Show raw scraped reviews
                with st.expander("Raw Fetched Reviews"):
                    st.json(reviews)
                    
        elif url_analyze_btn:
            st.warning("⚠️ Please enter a product URL to analyze.")

    with tab2:
        st.header("📝 Enter a Single Product Review")
        review_text = st.text_area(
            "Paste your product review here...",
            height=150,
            placeholder="Example: The camera quality is amazing but the battery drains quickly.",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_btn = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("🔄 ClearText", use_container_width=True)
        
        if clear_btn:
             st.rerun()
             
        if analyze_btn and review_text.strip():
            with st.spinner("🔬 Analyzing your review..."):
                analysis_result = analyzer.analyze_review(review_text)
                
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                overall_sentiment = analysis_result['overall_sentiment']
                granular_label = overall_sentiment.get('granular_label', overall_sentiment['label'])
                sentiment_emoji = display_sentiment_emoji(granular_label)
                sentiment_color = get_sentiment_color(granular_label)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    <h2 style="color: {sentiment_color};">
                        {sentiment_emoji} {granular_label.upper()}
                    </h2>
                    <p style="font-size: 1.2rem; color: #b0b0b0;">
                        <strong>Score:</strong> {overall_sentiment['score']:.1%} | 
                        <strong>Intensity:</strong> {overall_sentiment.get('intensity', 0):.0%}
                    </p>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{overall_sentiment['score']:.0%}</div>
                        <div class="metric-label">Overall Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 📊 Sentiment Spectrum")
                spectrum_fig = create_sentiment_spectrum(overall_sentiment['score'], granular_label)
                st.plotly_chart(spectrum_fig, use_container_width=True)
                
                st.markdown("### 🎯 Sentiment Gauge")
                fig_gauge = create_sentiment_gauge(overall_sentiment['score'], overall_sentiment['label'])
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                humor_data = analysis_result.get('humor_analysis', {})
                if humor_data:
                    display_humor_analysis(humor_data)
                    st.markdown("---")
                
                contradiction_data = analysis_result.get('contradiction_analysis', {})
                if contradiction_data:
                    display_contradictions(contradiction_data)
                    st.markdown("---")
                
                aspects = analysis_result.get('aspects', [])
                if aspects:
                    display_aspect_analysis(aspects)
                    st.markdown("---")
                
                st.subheader("📊 Word Cloud Visualization")
                wordcloud_fig = create_wordcloud(review_text)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig, use_container_width=True)
                
                st.markdown("---")
                with st.expander("🔧 View Raw Analysis Data", expanded=False):
                    st.json(analysis_result)
        elif analyze_btn:
            st.warning("⚠️ Please enter a review to analyze.")

if __name__ == "__main__":
    main()
