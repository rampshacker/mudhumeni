#!/usr/bin/env python3
"""
🌾 AgriBot Pro - Complete Production-Ready Version
Enterprise Agriculture AI Platform with All Features

Run: streamlit run agribot_streamlit.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Check Python version
if sys.version_info < (3, 8):
    print("❌ Python 3.8+ required")
    sys.exit(1)

# Create necessary directories
for directory in ["agri_data", "vector_db", "uploads", "logs", "database", "static", 
                  "exports", "marketplace", "saved_tips", "crop_photos", "field_maps", "agri_data/resources"]:
    Path(directory).mkdir(exist_ok=True)

# Setup logging BEFORE using logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agribot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Optional imports - AFTER logger is defined
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

try:
    import speech_recognition as sr
    HAS_SPEECH = True
    logger.info("✅ Speech recognition available")
except ImportError:
    HAS_SPEECH = False
    logger.info("ℹ️ Speech recognition not available (optional feature)")

try:
    from streamlit_webrtc import webrtc_streamer
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Mudhumeni - AgriBot Pro",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://amaryllissuccess.co.zw',
        'Report a bug': "mailto:support@amaryllissuccess.co.zw",
        'About': "# AgriBot Pro v2.0\nYour Complete Farming Assistant"
    }
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    APP_NAME = "Mudhumeni"
    APP_TAGLINE = "Your Complete Intelligent Agriculture Assistant"
    VERSION = "2.1.0"
    COMPANY_NAME = "Amaryllis Success"
    COMPANY_WEBSITE = "https://amaryllissuccess.co.zw"
    SUPPORT_EMAIL = "support@amaryllissuccess.co.zw"
    WHATSAPP = "+263 77 123 4567"
    
    LLM_MODEL = "llama3.2"
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TEMPERATURE = 0.7
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    FARMING_TYPES = [
        "crop_farming", "fish_farming", "goat_farming",
        "pig_farming", "poultry_farming", "cattle_farming",
        "general_agriculture"
    ]
    
    CROP_TYPES = [
        "Maize", "Wheat", "Tobacco", "Cotton", "Soybeans",
        "Vegetables", "Fruits", "Coffee", "Tea"
    ]
    
    DEFAULT_LOCATION = {
        "city": "Harare",
        "country": "Zimbabwe",
        "lat": -17.8292,
        "lon": 31.0522
    }
    
    # Alert thresholds
    FROST_TEMP = 2  # Celsius
    HEAVY_RAIN = 50  # mm
    HIGH_WIND = 40  # km/h

config = Config()

# ============================================================================
# ENHANCED CUSTOM CSS
# ============================================================================

def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-green: #2e7d32;
        --light-green: #66bb6a;
        --dark-green: #1b5e20;
        --orange: #ff6f00;
        --brown: #795548;
        --blue: #2196f3;
        --red: #f44336;
    }
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: #f8faf8;
    }
    
    /* Enhanced header */
    .header-container {
        background: linear-gradient(135deg, #1b5e20, #2e7d32, #66bb6a);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .header-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .header-tagline {
        color: #e8f5e9;
        font-size: 1.4rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #e8f5e9);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        cursor: pointer;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.2);
        border-color: var(--light-green);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        color: var(--primary-green);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* Action cards */
    .action-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        border-left: 5px solid var(--orange);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .action-card:hover {
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        transform: translateX(5px);
    }
    
    .action-card h4 {
        color: var(--primary-green);
        margin: 0 0 0.5rem 0;
    }
    
    /* Alert banner */
    .alert-banner {
        background: linear-gradient(135deg, #ff6f00, #ff8f00);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(255, 111, 0, 0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .alert-banner .icon {
        font-size: 2.5rem;
        margin-right: 1rem;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #4caf50, #66bb6a);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ff9800, #ffa726);
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #f44336, #e57373);
    }
    
    /* Weather widget */
    .weather-widget {
        background: linear-gradient(135deg, #2196f3, #64b5f6);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(33, 150, 243, 0.3);
    }
    
    .weather-temp {
        font-size: 4rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Calendar widget */
    .calendar-widget {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }
    
    .calendar-item {
        padding: 1rem;
        border-left: 4px solid var(--primary-green);
        margin-bottom: 1rem;
        background: #f5f5f5;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .calendar-item:hover {
        background: #e8f5e9;
        transform: translateX(5px);
    }
    
    /* Stats bar */
    .stats-bar {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stat-item {
        text-align: center;
        padding: 0 1rem;
        border-right: 2px solid rgba(255,255,255,0.3);
    }
    
    .stat-item:last-child {
        border-right: none;
    }
    
    .stat-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Enhanced buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2e7d32, #66bb6a);
        color: white;
        border: none;
        padding: 0.85rem 2.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }
    
    /* Community feed */
    .community-post {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    .post-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .post-avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: var(--primary-green);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-right: 1rem;
    }
    
    .post-actions {
        display: flex;
        gap: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 1rem;
    }
    
    .post-action {
        cursor: pointer;
        color: #666;
        transition: color 0.3s ease;
    }
    
    .post-action:hover {
        color: var(--primary-green);
    }
    
    /* Offline indicator */
    .offline-badge {
        position: fixed;
        top: 70px;
        right: 20px;
        background: #f44336;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(244, 67, 54, 0.5);
    }
    
    .online-badge {
        background: #4caf50;
    }
    
    /* Voice button */
    .voice-button {
        position: relative;
        background: linear-gradient(135deg, #ff6f00, #ff8f00);
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 111, 0, 0.4);
    }
    
    .voice-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(255, 111, 0, 0.6);
    }
    
    .voice-button.recording {
        animation: pulse-red 1.5s infinite;
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7); }
        50% { box-shadow: 0 0 0 20px rgba(244, 67, 54, 0); }
    }
    
    /* Mobile optimizations */
    @media (max-width: 768px) {
        .header-title { font-size: 2rem; }
        .header-tagline { font-size: 1rem; }
        .metric-card { padding: 1.5rem; }
        .metric-value { font-size: 2.5rem; }
        .stats-bar { flex-direction: column; gap: 1rem; }
        .stat-item { border-right: none; border-bottom: 2px solid rgba(255,255,255,0.3); padding: 0.5rem 0; }
        .stat-item:last-child { border-bottom: none; }
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid rgba(46, 125, 50, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary-green);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Crop health indicator */
    .health-indicator {
        display: inline-block;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .health-excellent { background: #4caf50; }
    .health-good { background: #8bc34a; }
    .health-fair { background: #ffeb3b; }
    .health-poor { background: #ff9800; }
    .health-critical { background: #f44336; }
    
    /* Footer enhancement */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f5f5f5, #e8f5e9);
        border-radius: 20px;
        margin-top: 4rem;
        border-top: 5px solid #66bb6a;
        box-shadow: 0 -5px 20px rgba(0,0,0,0.08);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: #333;
        color: #fff;
        text-align: center;
        padding: 5px 10px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATABASE & SERVICES
# ============================================================================

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
    from sqlalchemy.orm import declarative_base
    from sqlalchemy.orm import sessionmaker
    
    Base = declarative_base()
    
    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        username = Column(String(100), unique=True, nullable=False)
        email = Column(String(255))
        location = Column(String(255))
        created_at = Column(DateTime, default=datetime.now)
        last_active = Column(DateTime, default=datetime.now)
        total_queries = Column(Integer, default=0)
    
    class CropRecord(Base):
        __tablename__ = "crop_records"
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=False)
        crop_name = Column(String(100))
        field_name = Column(String(100))
        planted_date = Column(DateTime)
        expected_harvest = Column(DateTime)
        status = Column(String(50))
        health_score = Column(Integer, default=100)
        notes = Column(Text)
        created_at = Column(DateTime, default=datetime.now)
    
    class Task(Base):
        __tablename__ = "tasks"
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=False)
        title = Column(String(255))
        description = Column(Text)
        due_date = Column(DateTime)
        priority = Column(String(20))
        status = Column(String(20), default="pending")
        category = Column(String(50))
        created_at = Column(DateTime, default=datetime.now)
    
    class CommunityPost(Base):
        __tablename__ = "community_posts"
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=False)
        username = Column(String(100))
        content = Column(Text)
        likes = Column(Integer, default=0)
        comments_count = Column(Integer, default=0)
        created_at = Column(DateTime, default=datetime.now)
    
    engine = create_engine('sqlite:///database/agribot.db', echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    
    DB_AVAILABLE = True
    logger.info("✅ Database initialized")
except Exception as e:
    logger.warning(f"⚠️ Database not available: {e}")
    DB_AVAILABLE = False
    SessionLocal = None

# ============================================================================
# WEATHER SERVICE (Enhanced)
# ============================================================================

class WeatherService:
    @staticmethod
    def get_weather(location: str = None, lat: float = None, lon: float = None) -> dict:
        try:
            import requests
            
            if not lat or not lon:
                lat = config.DEFAULT_LOCATION["lat"]
                lon = config.DEFAULT_LOCATION["lon"]
                location = location or config.DEFAULT_LOCATION["city"]
            
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,apparent_temperature",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
                "timezone": "auto",
                "forecast_days": 7
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                current = data.get("current", {})
                daily = data.get("daily", {})
                
                # Check for alerts
                alerts = []
                temp = current.get('temperature_2m', 999)
                wind = current.get('wind_speed_10m', 0)
                
                if temp < config.FROST_TEMP:
                    alerts.append({"type": "frost", "message": "❄️ FROST WARNING: Protect sensitive crops!"})
                if wind > config.HIGH_WIND:
                    alerts.append({"type": "wind", "message": "💨 HIGH WIND: Avoid spraying operations"})
                
                # Check forecast for heavy rain
                for i, precip in enumerate(daily.get("precipitation_sum", [])[:3]):
                    if precip > config.HEAVY_RAIN:
                        date = daily.get("time", [])[i]
                        alerts.append({"type": "rain", "message": f"🌧️ HEAVY RAIN expected {date}"})
                
                return {
                    "location": location,
                    "current": {
                        "temperature": current.get('temperature_2m', 'N/A'),
                        "feels_like": current.get('apparent_temperature', 'N/A'),
                        "humidity": current.get('relative_humidity_2m', 'N/A'),
                        "precipitation": current.get('precipitation', 0),
                        "wind_speed": current.get('wind_speed_10m', 'N/A'),
                        "weather_code": current.get('weather_code', 0),
                        "time": current.get("time", datetime.now().isoformat())
                    },
                    "forecast": {
                        "dates": daily.get("time", [])[:7],
                        "max_temp": daily.get("temperature_2m_max", [])[:7],
                        "min_temp": daily.get("temperature_2m_min", [])[:7],
                        "precipitation": daily.get("precipitation_sum", [])[:7],
                        "rain_probability": daily.get("precipitation_probability_max", [])[:7],
                        "max_wind": daily.get("wind_speed_10m_max", [])[:7]
                    },
                    "alerts": alerts,
                    "status": "success"
                }
            else:
                return {"status": "error", "message": "Weather service unavailable"}
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def get_weather_icon(code: int) -> str:
        """Convert weather code to emoji"""
        icons = {
            0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
            45: "🌫️", 48: "🌫️",
            51: "🌦️", 53: "🌦️", 55: "🌦️",
            61: "🌧️", 63: "🌧️", 65: "🌧️",
            71: "🌨️", 73: "🌨️", 75: "🌨️",
            77: "❄️", 80: "🌧️", 81: "🌧️", 82: "🌧️",
            85: "🌨️", 86: "🌨️",
            95: "⛈️", 96: "⛈️", 99: "⛈️"
        }
        return icons.get(code, "🌤️")

# ============================================================================
# VOICE SERVICE
# ============================================================================

class VoiceService:
    @staticmethod
    def listen() -> Optional[str]:
        """Listen to microphone and convert speech to text"""
        if not HAS_SPEECH:
            return None
        
        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            text = recognizer.recognize_google(audio)
            return text
        except Exception as e:
            logger.error(f"Voice recognition error: {e}")
            return None

# ============================================================================
# AI AGENT (Enhanced)
# ============================================================================

def initialize_agribot():
    """Initialize AgriBot with caching"""
    
    class AgriBot:
        def __init__(self):
            self.llm = None
            self.vector_store = None
            self.embeddings = None
            self.db_session = None
            self.weather = WeatherService()
            self.voice = VoiceService()
            
            self._initialize_llm()
            self._initialize_vector_store()
            
            if DB_AVAILABLE and SessionLocal:
                self.db_session = SessionLocal()
            
            logger.info("✅ AgriBot initialized")
        
        def _initialize_llm(self):
            try:
                from langchain_ollama import ChatOllama
                
                self.llm = ChatOllama(
                    model=config.LLM_MODEL,
                    temperature=config.TEMPERATURE,
                    base_url=config.OLLAMA_BASE_URL
                )
                logger.info(f"✅ LLM initialized: {config.LLM_MODEL}")
            except Exception as e:
                logger.error(f"❌ LLM initialization failed: {e}")
                self.llm = None
        
        def _initialize_vector_store(self):
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                from langchain_core.documents import Document
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
                
                Path("vector_db").mkdir(exist_ok=True)
                
                vector_db_path = Path("vector_db")
                if vector_db_path.exists() and any(vector_db_path.iterdir()):
                    try:
                        self.vector_store = Chroma(
                            persist_directory="vector_db",
                            embedding_function=self.embeddings
                        )
                        logger.info("✅ Loaded existing vector store")
                        return
                    except:
                        pass
                
                self._create_vector_store()
                    
            except Exception as e:
                logger.error(f"❌ Vector store initialization failed: {e}")
                self.vector_store = None
        
        def _create_vector_store(self):
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_core.documents import Document
                
                initial_docs = [
                    Document(
                        page_content="Agriculture AI knowledge base for Zimbabwe and Southern Africa.",
                        metadata={"source": "system"}
                    )
                ]
                
                self.vector_store = Chroma.from_documents(
                    documents=initial_docs,
                    embedding=self.embeddings,
                    persist_directory="vector_db"
                )
                logger.info("✅ Created new vector store")
            except Exception as e:
                logger.error(f"❌ Failed to create vector store: {e}")
        
        def query(self, question: str, username: str = "guest", location: str = None) -> str:
            if not self.llm:
                return "❌ AI model not available. Please ensure Ollama is running."
            
            try:
                now = datetime.now()
                season = self._get_season(now.month)
                
                # Search local vector store
                local_context = ""
                if self.vector_store:
                    try:
                        results = self.vector_store.similarity_search(question, k=3)
                        if results:
                            local_context = "\n\n".join([f"Document: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}" for doc in results])
                    except Exception as e:
                        logger.error(f"Vector search error: {e}")
                
                # Search the web
                web_context = ""
                try:
                    web_results = self._search_web(question)
                    if web_results:
                        web_context = "\n\n".join([
                            f"Web Source: {r['title']}\nURL: {r['url']}\n{r['snippet']}" 
                            for r in web_results[:3]
                        ])
                except Exception as e:
                    logger.error(f"Web search error: {e}")
                
                # Build comprehensive prompt
                system_context = f"""Current Date: {now.strftime('%Y-%m-%d')}
Season: {season}
Location: {location or config.DEFAULT_LOCATION['city']}"""
                
                full_context = ""
                if local_context:
                    full_context += f"\n\n=== KNOWLEDGE BASE ===\n{local_context}"
                if web_context:
                    full_context += f"\n\n=== WEB SEARCH ===\n{web_context}"
                
                if full_context:
                    prompt = f"""{system_context}

You are an expert agriculture assistant. Use the following information:

{full_context}

Question: {question}

Provide practical, location-aware advice for {location or config.DEFAULT_LOCATION['city']}."""
                else:
                    prompt = f"""{system_context}

Question: {question}

Provide practical farming advice."""
                
                response = self.llm.invoke(prompt)
                self._log_query(username, question, local_context, web_context)
                
                return response.content
                
            except Exception as e:
                logger.error(f"Query error: {e}")
                return f"❌ Error: {str(e)}"
        
        def _search_web(self, query: str, num_results: int = 3) -> List[dict]:
            try:
                import requests
                url = "https://api.duckduckgo.com/"
                params = {"q": f"{query} agriculture farming", "format": "json", "no_html": 1, "skip_disambig": 1}
                response = requests.get(url, params=params, timeout=5)
                data = response.json()
                results = []
                if data.get("Abstract"):
                    results.append({"title": data.get("Heading", "Info"), "url": data.get("AbstractURL", ""), "snippet": data.get("Abstract", "")})
                for topic in data.get("RelatedTopics", [])[:2]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({"title": topic.get("Text", "")[:50], "url": topic.get("FirstURL", ""), "snippet": topic.get("Text", "")})
                return results
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                return []
        
        def _log_query(self, username: str, question: str, local_ctx: str, web_ctx: str):
            try:
                if self.db_session:
                    user = self.db_session.query(User).filter_by(username=username).first()
                    if user:
                        user.total_queries += 1
                        user.last_active = datetime.now()
                        self.db_session.commit()
            except Exception as e:
                logger.error(f"Logging error: {e}")
        
        def add_documents(self, files) -> tuple:
            if not self.vector_store:
                return "❌ Vector store not available", {}
            if not files:
                return "⚠️ No files selected", {}
            
            try:
                from langchain_community.document_loaders import PyPDFLoader, TextLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                
                documents = []
                processed_files = []
                failed_files = []
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                
                for file in files:
                    try:
                        file_path = Path("uploads") / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        if file_path.suffix.lower() == '.pdf':
                            loader = PyPDFLoader(str(file_path))
                        elif file_path.suffix.lower() == '.txt':
                            loader = TextLoader(str(file_path), encoding='utf-8')
                        else:
                            failed_files.append(f"{file.name} (unsupported)")
                            continue
                        
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata['filename'] = file.name
                            doc.metadata['upload_date'] = datetime.now().isoformat()
                        
                        documents.extend(docs)
                        processed_files.append(file.name)
                        
                    except Exception as e:
                        failed_files.append(f"{file.name} ({str(e)})")
                
                if documents:
                    chunks = splitter.split_documents(documents)
                    self.vector_store.add_documents(chunks)
                    
                    result_msg = f"✅ Processed {len(processed_files)} file(s)\n📊 Created {len(chunks)} chunks"
                    if failed_files:
                        result_msg += f"\n\n⚠️ Failed:\n" + "\n".join(f"  - {f}" for f in failed_files)
                    
                    return result_msg, {"files_processed": len(processed_files), "files_failed": len(failed_files), "total_chunks": len(chunks)}
                else:
                    return "⚠️ No documents loaded", {"files_failed": len(failed_files)}
                    
            except Exception as e:
                return f"❌ Error: {str(e)}", {}
        
        def _get_season(self, month: int) -> str:
            if month in [11, 12, 1, 2, 3]:
                return "Rainy/Growing Season"
            elif month in [4, 5]:
                return "Harvest Season"
            else:
                return "Dry Season"
        
        def calculate_farming_costs(self, farming_type: str, scale: float, duration: int = 12) -> dict:
            cost_estimates = {
                "crop_farming": {"seeds": 50, "fertilizer": 150, "pesticides": 80, "labor": 200, "water": 100, "equipment": 150},
                "fish_farming": {"fingerlings": 30, "feed": 200, "water_treatment": 50, "labor": 150, "equipment": 100},
                "goat_farming": {"stock": 200, "feed": 30, "veterinary": 15, "housing": 50, "labor": 100},
                "pig_farming": {"stock": 300, "feed": 50, "veterinary": 20, "housing": 80, "labor": 120},
                "poultry_farming": {"chicks": 2, "feed": 3, "veterinary": 1, "housing": 5, "labor": 80}
            }
            
            costs = cost_estimates.get(farming_type, {})
            if not costs:
                return {"error": "Farming type not found"}
            
            total_cost = 0
            breakdown = {}
            
            for item, unit_cost in costs.items():
                if item in ["feed", "veterinary", "water"]:
                    cost = unit_cost * scale * duration
                else:
                    cost = unit_cost * scale
                breakdown[item] = round(cost, 2)
                total_cost += cost
            
            return {
                "farming_type": farming_type,
                "scale": scale,
                "duration_months": duration,
                "breakdown": breakdown,
                "total_cost": round(total_cost, 2),
                "monthly_average": round(total_cost / duration, 2)
            }
        
        def calculate_loan(self, principal: float, rate: float, years: int) -> dict:
            """Calculate agricultural loan details"""
            monthly_rate = rate / 100 / 12
            num_payments = years * 12
            
            if monthly_rate > 0:
                monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            else:
                monthly_payment = principal / num_payments
            
            total_payment = monthly_payment * num_payments
            total_interest = total_payment - principal
            
            return {
                "principal": round(principal, 2),
                "monthly_payment": round(monthly_payment, 2),
                "total_payment": round(total_payment, 2),
                "total_interest": round(total_interest, 2),
                "rate": rate,
                "years": years
            }
    
    return AgriBot()

# ============================================================================
# SESSION STATE
# ============================================================================

if 'agent' not in st.session_state:
    with st.spinner("🌾 Initializing AgriBot..."):
        st.session_state.agent = initialize_agribot()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'username' not in st.session_state:
    st.session_state.username = "Guest"

if 'location' not in st.session_state:
    st.session_state.location = config.DEFAULT_LOCATION["city"]

if 'navigation' not in st.session_state:
    st.session_state.navigation = "Dashboard"

if 'tips_generated' not in st.session_state:
    st.session_state.tips_generated = 0

if 'current_tip' not in st.session_state:
    st.session_state.current_tip = None

if 'crops' not in st.session_state:
    st.session_state.crops = []

if 'tasks' not in st.session_state:
    st.session_state.tasks = []

if 'online_status' not in st.session_state:
    st.session_state.online_status = True

if 'voice_active' not in st.session_state:
    st.session_state.voice_active = False

# ============================================================================
# HEADER
# ============================================================================

def render_header():
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">🌾 {config.APP_NAME}</h1>
        <p class="header-tagline">{config.APP_TAGLINE}</p>
        <p style="color: #e8f5e9; font-size: 0.9rem; margin-top: 1rem;">
            Version {config.VERSION} | Powered by AI | © {config.COMPANY_NAME}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/2e7d32/ffffff?text=Amaryllis+Success", use_container_width=True)
        
        st.markdown("### 👤 User Profile")
        st.session_state.username = st.text_input("Your Name", value=st.session_state.username, key="sidebar_username")
        st.session_state.location = st.text_input("📍 Location", value=st.session_state.location, key="sidebar_location")
        
        st.markdown("---")
        
        now = datetime.now()
        season = st.session_state.agent._get_season(now.month)
        
        st.markdown("### 📅 Current Info")
        st.info(f"""
        **Date:** {now.strftime('%B %d, %Y')}  
        **Time:** {now.strftime('%I:%M %p')}  
        **Season:** {season}  
        **Location:** {st.session_state.location}
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 System Status")
        llm_status = "🟢 Online" if st.session_state.agent.llm else "🔴 Offline"
        vector_status = "🟢 Active" if st.session_state.agent.vector_store else "🔴 Inactive"
        network_status = "🟢 Connected" if st.session_state.online_status else "🔴 Offline"
        
        st.markdown(f"""
        **AI Model:** {llm_status}  
        **Knowledge Base:** {vector_status}  
        **Network:** {network_status}  
        **Database:** 🟢 Active
        """)
        
        st.markdown("---")
        
        st.markdown("### 📱 Quick Contact")
        st.markdown(f"""
        📧 [Email]({config.SUPPORT_EMAIL})  
        📱 [WhatsApp](https://wa.me/{config.WHATSAPP.replace('+', '').replace(' ', '')})  
        🌐 [Website]({config.COMPANY_WEBSITE})
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        lang = st.selectbox("🌐 Language", ["English", "Shona", "Ndebele"], key="sidebar_lang")
        theme = st.selectbox("🎨 Theme", ["Light", "Dark", "Auto"], key="sidebar_theme")
        
        if st.button("💾 Save Preferences", use_container_width=True):
            st.success("✅ Preferences saved!")

# ============================================================================
# PAGES
# ============================================================================

def page_dashboard():
    st.markdown("## 🏠 Dashboard")
    
    # Weather alerts
    weather_data = st.session_state.agent.weather.get_weather(st.session_state.location)
    if weather_data.get("status") == "success" and weather_data.get("alerts"):
        for alert in weather_data["alerts"]:
            alert_class = "alert-danger" if alert["type"] in ["frost", "wind"] else "alert-warning"
            st.markdown(f"""
            <div class="alert-banner {alert_class}">
                <div style="display: flex; align-items: center;">
                    <div class="icon">{alert["message"].split()[0]}</div>
                    <div>
                        <strong>WEATHER ALERT</strong><br>
                        {alert["message"]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick stats bar
    total_crops = len(st.session_state.crops)
    pending_tasks = len([t for t in st.session_state.tasks if t.get('status') == 'pending'])
    avg_health = sum([c.get('health', 100) for c in st.session_state.crops]) / max(total_crops, 1)
    
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-icon">🌾</div>
            <div class="stat-value">{total_crops}</div>
            <div class="stat-label">Active Crops</div>
        </div>
        <div class="stat-item">
            <div class="stat-icon">✅</div>
            <div class="stat-value">{pending_tasks}</div>
            <div class="stat-label">Pending Tasks</div>
        </div>
        <div class="stat-item">
            <div class="stat-icon">💚</div>
            <div class="stat-value">{int(avg_health)}%</div>
            <div class="stat-label">Avg Health</div>
        </div>
        <div class="stat-item">
            <div class="stat-icon">📊</div>
            <div class="stat-value">{st.session_state.tips_generated}</div>
            <div class="stat-label">Tips Generated</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main dashboard grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🌾", key="dash_crops", help="AI Consultation"):
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">🌾</div>
                <div class="metric-label">AI Consultation</div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.navigation = "AI Chat"
            st.rerun()
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🌾</div>
            <div class="metric-label">AI Consultation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🌤️</div>
            <div class="metric-label">Weather Forecast</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📊</div>
            <div class="metric-label">Crop Tracking</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">💡</div>
            <div class="metric-label">Farming Tips</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Quick Actions
        st.markdown("### 📈 Quick Actions")
        
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("🌾 Ask AI Question", use_container_width=True):
                st.session_state.navigation = "AI Chat"
                st.rerun()
            if st.button("📸 Diagnose Crop Disease", use_container_width=True):
                st.session_state.navigation = "Crop Health"
                st.rerun()
            if st.button("🗺️ View Field Map", use_container_width=True):
                st.info("📍 Field mapping feature - Upload GPS coordinates")
        
        with action_col2:
            if st.button("🌤️ Check Weather", use_container_width=True):
                st.session_state.navigation = "Weather"
                st.rerun()
            if st.button("💵 Calculate Costs", use_container_width=True):
                st.session_state.navigation = "Calculator"
                st.rerun()
            if st.button("👥 Community Forum", use_container_width=True):
                st.session_state.navigation = "Community"
                st.rerun()
        
        # Current weather widget
        st.markdown("### 🌤️ Current Weather")
        if weather_data.get("status") == "success":
            current = weather_data["current"]
            now = datetime.now()
            weather_icon = WeatherService.get_weather_icon(current.get('weather_code', 0))
            
            st.markdown(f"""
            <div class="weather-widget">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 4rem;">{weather_icon}</div>
                        <div class="weather-temp">{current.get('temperature')}°C</div>
                        <div style="font-size: 1.2rem; opacity: 0.9;">Feels like {current.get('feels_like')}°C</div>
                    </div>
                    <div style="text-align: right;">
                        <h3 style="margin: 0;">📍 {st.session_state.location}</h3>
                        <p style="margin: 5px 0;">{now.strftime('%A, %B %d')}</p>
                        <p style="margin: 5px 0;">{now.strftime('%I:%M %p')}</p>
                        <hr style="margin: 10px 0; opacity: 0.5;">
                        <p style="margin: 5px 0;">💧 Humidity: {current.get('humidity')}%</p>
                        <p style="margin: 5px 0;">💨 Wind: {current.get('wind_speed')} km/h</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        # Calendar widget
        st.markdown("### 📅 This Week's Tasks")
        st.markdown("""
        <div class="calendar-widget">
            <div class="calendar-item">
                <strong>Monday</strong><br>
                💧 Irrigate Field 3<br>
                <small style="color: #666;">8:00 AM</small>
            </div>
            <div class="calendar-item">
                <strong>Wednesday</strong><br>
                🌱 Plant Maize<br>
                <small style="color: #666;">6:00 AM</small>
            </div>
            <div class="calendar-item">
                <strong>Friday</strong><br>
                🔬 Soil Testing<br>
                <small style="color: #666;">10:00 AM</small>
            </div>
            <div class="calendar-item">
                <strong>Saturday</strong><br>
                💊 Spray Pesticides<br>
                <small style="color: #666;">7:00 AM</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("➕ Add New Task", use_container_width=True):
            st.session_state.navigation = "Tasks"
            st.rerun()
        
        # Quick tip
        st.markdown("### 💡 Today's Tip")
        st.info("""
        **🌾 Maize Planting Season**
        
        November-December is ideal for maize planting in Zimbabwe. 
        Ensure soil moisture is adequate before planting.
        
        [Generate More Tips →]
        """)

def page_ai_chat():
    st.markdown("## 💬 AI Consultation")
    
    # Voice input section (simplified)
    st.markdown("### 🎤 Voice Input")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input is primary
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Ask your agriculture question:", height=100, 
                                       placeholder="E.g., What are the best crops to plant this season?",
                                       key="chat_input")
            submitted = st.form_submit_button("📤 Send", use_container_width=True, type="primary")
    
    with col2:
        if HAS_SPEECH:
            st.info("🎤 **Voice Input Available**\n\nClick below to use voice")
            if st.button("🎙️ Record Voice", use_container_width=True):
                with st.spinner("🎤 Listening..."):
                    try:
                        text = st.session_state.agent.voice.listen()
                        if text:
                            st.success(f"✅ Heard: {text}")
                            # Store in session state to use
                            st.session_state.voice_input = text
                            st.rerun()
                        else:
                            st.error("❌ Could not understand audio")
                    except Exception as e:
                        st.error(f"❌ Voice error: {str(e)}")
        else:
            st.info("""
            🎤 **Voice Input**
            
            Install for voice features:
            ```bash
            pip install SpeechRecognition
            ```
            
            *Optional feature*
            """)
    
    # Check if we have voice input from previous interaction
    if hasattr(st.session_state, 'voice_input') and st.session_state.voice_input:
        user_input = st.session_state.voice_input
        submitted = True
        # Clear voice input
        del st.session_state.voice_input
    
    if submitted and user_input:
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.agent.query(user_input, st.session_state.username, st.session_state.location)
            
            st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": datetime.now()})
            st.session_state.chat_history.append({"role": "assistant", "content": response, "timestamp": datetime.now()})
    
    st.markdown("---")
    st.markdown("### 💭 Conversation History")
    
    # Display chat history
    if st.session_state.chat_history:
        for msg in reversed(st.session_state.chat_history[-10:]):
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <strong>👤 You ({msg['timestamp'].strftime('%I:%M %p')}):</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <strong>🤖 AgriBot ({msg['timestamp'].strftime('%I:%M %p')}):</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("📥 Export Chat", use_container_width=True):
                chat_text = "\n\n".join([f"{m['role'].upper()} ({m['timestamp'].strftime('%Y-%m-%d %I:%M %p')}): {m['content']}" 
                                        for m in st.session_state.chat_history])
                st.download_button(
                    label="⬇️ Download", 
                    data=chat_text, 
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.info("""
        👋 **Welcome to AI Consultation!**
        
        Ask me anything about:
        - 🌾 Crop selection and planting
        - 🐄 Livestock management
        - 💧 Irrigation and water management
        - 🐛 Pest and disease control
        - 💰 Farming economics
        - 🌤️ Weather-based farming advice
        
        **Quick Tips:**
        - Be specific in your questions
        - Mention your location for better advice
        - Upload documents to Knowledge Base for more accurate answers
        """)
    
    # Quick question suggestions
    st.markdown("---")
    st.markdown("### 💡 Quick Questions")
    
    quick_questions = [
        "What crops should I plant this season?",
        "How do I deal with maize pests?",
        "When is the best time to harvest wheat?",
        "How much water does my crop need?",
        "What fertilizer should I use for vegetables?"
    ]
    
    cols = st.columns(3)
    for idx, question in enumerate(quick_questions):
        with cols[idx % 3]:
            if st.button(f"❓ {question[:30]}...", key=f"quick_{idx}", use_container_width=True):
                st.session_state.voice_input = question
                st.rerun()

def page_weather():
    st.markdown("## 🌤️ Weather Forecast")
    
    weather_data = st.session_state.agent.weather.get_weather(st.session_state.location)
    
    if weather_data.get("status") == "success":
        current = weather_data["current"]
        forecast = weather_data["forecast"]
        now = datetime.now()
        weather_icon = WeatherService.get_weather_icon(current.get('weather_code', 0))
        
        st.markdown("### 🌡️ Current Conditions")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            temp = current.get('temperature', 'N/A')
            st.markdown(f"""
            <div style="text-align: center; padding: 30px;">
                <div style="font-size: 5rem;">{weather_icon}</div>
                <div style="font-size: 4rem; font-weight: bold; color: #ff6f00;">{temp}</div>
                <div style="font-size: 1.5rem; color: #666;">°C | °F</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 30px 20px;">
                <p style="margin: 10px 0; font-size: 1.1rem;"><strong>Precipitation:</strong> {current.get('precipitation', 0)}mm</p>
                <p style="margin: 10px 0; font-size: 1.1rem;"><strong>Humidity:</strong> {current.get('humidity', 'N/A')}%</p>
                <p style="margin: 10px 0; font-size: 1.1rem;"><strong>Wind:</strong> {current.get('wind_speed', 'N/A')} km/h</p>
                <p style="margin: 10px 0; font-size: 1.1rem;"><strong>Feels Like:</strong> {current.get('feels_like', 'N/A')}°C</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: right; padding: 30px 20px;">
                <h3 style="margin: 5px 0; color: #2e7d32;">Weather</h3>
                <p style="margin: 5px 0; font-size: 1.1rem; color: #666;">{now.strftime('%A')}</p>
                <p style="margin: 5px 0; color: #888;">Partly sunny</p>
                <hr style="margin: 15px 0;">
                <p style="margin: 5px 0; font-weight: bold;">📍 {st.session_state.location}</p>
                <p style="margin: 5px 0; color: #666;">{now.strftime('%B %d, %Y')}</p>
                <p style="margin: 5px 0; color: #666;">{now.strftime('%I:%M %p')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "💧 Precipitation", "💨 Wind"])
        
        with tab1:
            df_temp = pd.DataFrame({
                'Date': [datetime.fromisoformat(d).strftime('%a, %b %d') for d in forecast.get('dates', [])],
                'Max Temp (°C)': forecast.get('max_temp', []),
                'Min Temp (°C)': forecast.get('min_temp', [])
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_temp['Date'], y=df_temp['Max Temp (°C)'], name='Max', line=dict(color='#ff6f00', width=3), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df_temp['Date'], y=df_temp['Min Temp (°C)'], name='Min', line=dict(color='#2196f3', width=3), mode='lines+markers'))
            fig.update_layout(title="7-Day Temperature Forecast", xaxis_title="Date", yaxis_title="Temperature (°C)", height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_temp, use_container_width=True)
        
        with tab2:
            df_precip = pd.DataFrame({
                'Date': [datetime.fromisoformat(d).strftime('%a, %b %d') for d in forecast.get('dates', [])],
                'Precipitation (mm)': forecast.get('precipitation', []),
                'Rain Probability (%)': forecast.get('rain_probability', [])
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_precip['Date'], y=df_precip['Precipitation (mm)'], name='Precipitation', marker_color='#2196f3'))
            fig.update_layout(title="7-Day Precipitation Forecast", xaxis_title="Date", yaxis_title="Precipitation (mm)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_precip, use_container_width=True)
        
        with tab3:
            wind_speed = current.get('wind_speed', 0)
            st.markdown(f"""
            ### 💨 Current Wind Conditions
            
            - **Wind Speed:** {wind_speed} km/h
            - **Direction:** Variable
            - **Gusts:** {float(wind_speed) * 1.3 if isinstance(wind_speed, (int, float)) else 'N/A'} km/h
            
            **Farming Impact:**
            - {'✅ Safe for spraying' if wind_speed < 15 else '⚠️ Too windy for spraying'}
            - {'✅ Good for drone operations' if wind_speed < 25 else '⚠️ Not recommended for drones'}
            """)
    else:
        st.error("⚠️ Could not fetch weather data")

def page_calculator():
    st.markdown("## 💵 Agricultural Calculator")
    
    tab1, tab2, tab3 = st.tabs(["💰 Cost Calculator", "🏦 Loan Calculator", "📊 ROI Calculator"])
    
    with tab1:
        st.markdown("### 💰 Farming Cost Estimator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            farming_type = st.selectbox("Farming Type", ["crop_farming", "fish_farming", "goat_farming", "pig_farming", "poultry_farming"])
            scale = st.number_input("Scale (hectares/animals)", min_value=1, value=10)
            duration = st.number_input("Duration (months)", min_value=1, value=12, max_value=36)
        
        if st.button("📊 Calculate Costs", use_container_width=True, type="primary"):
            result = st.session_state.agent.calculate_farming_costs(farming_type, scale, duration)
            
            with col2:
                st.markdown("### 📈 Cost Summary")
                st.metric("💵 Total Cost", f"${result['total_cost']:,.2f}")
                st.metric("📅 Monthly Average", f"${result['monthly_average']:,.2f}")
                st.metric("📊 Scale", f"{scale} units")
            
            st.markdown("---")
            st.markdown("### 💰 Detailed Breakdown")
            
            breakdown_df = pd.DataFrame({
                'Item': list(result['breakdown'].keys()),
                'Cost (USD)': list(result['breakdown'].values())
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(breakdown_df, values='Cost (USD)', names='Item', title='Cost Distribution', color_discrete_sequence=px.colors.sequential.Greens)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(breakdown_df, x='Item', y='Cost (USD)', title='Cost by Item', color='Cost (USD)', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(breakdown_df, use_container_width=True)
    
    with tab2:
        st.markdown("### 🏦 Agricultural Loan Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            principal = st.number_input("💵 Loan Amount ($)", min_value=100, value=10000, step=100)
            rate = st.number_input("📈 Interest Rate (%)", min_value=0.1, value=8.5, step=0.1)
            years = st.number_input("📅 Loan Term (years)", min_value=1, value=5, max_value=30)
        
        if st.button("🧮 Calculate Loan", use_container_width=True, type="primary"):
            loan_result = st.session_state.agent.calculate_loan(principal, rate, years)
            
            with col2:
                st.markdown("### 💳 Loan Summary")
                st.metric("💵 Monthly Payment", f"${loan_result['monthly_payment']:,.2f}")
                st.metric("📊 Total Payment", f"${loan_result['total_payment']:,.2f}")
                st.metric("💸 Total Interest", f"${loan_result['total_interest']:,.2f}")
            
            st.markdown("---")
            
            # Amortization schedule preview
            st.markdown("### 📊 Payment Breakdown")
            
            months = loan_result['years'] * 12
            payments_data = []
            balance = loan_result['principal']
            monthly_rate = rate / 100 / 12
            
            for i in range(min(12, months)):
                interest = balance * monthly_rate
                principal_payment = loan_result['monthly_payment'] - interest
                balance -= principal_payment
                payments_data.append({
                    'Month': i + 1,
                    'Payment': loan_result['monthly_payment'],
                    'Principal': principal_payment,
                    'Interest': interest,
                    'Balance': max(0, balance)
                })
            
            df_payments = pd.DataFrame(payments_data)
            st.dataframe(df_payments.round(2), use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Return on Investment Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            investment = st.number_input("💵 Total Investment ($)", min_value=100, value=5000)
            revenue = st.number_input("💰 Expected Revenue ($)", min_value=0, value=8000)
            expenses = st.number_input("💸 Operating Expenses ($)", min_value=0, value=2000)
        
        if st.button("📈 Calculate ROI", use_container_width=True, type="primary"):
            net_profit = revenue - expenses - investment
            roi = (net_profit / investment) * 100 if investment > 0 else 0
            
            with col2:
                st.metric("💰 Net Profit", f"${net_profit:,.2f}", delta=f"{roi:.1f}% ROI")
                st.metric("📊 ROI Percentage", f"{roi:.2f}%")
                st.metric("💵 Break-even", f"${investment + expenses:,.2f}")
            
            st.markdown("---")
            
            fig = go.Figure(data=[
                go.Bar(name='Investment', x=['Costs'], y=[investment], marker_color='#f44336'),
                go.Bar(name='Expenses', x=['Costs'], y=[expenses], marker_color='#ff9800'),
                go.Bar(name='Revenue', x=['Income'], y=[revenue], marker_color='#4caf50')
            ])
            fig.update_layout(title='Investment vs Returns', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)

def page_farming_tips():
    st.markdown("## 💡 Farming Tips & Knowledge")
    
    st.info("📚 Get AI-generated farming tips from agricultural manuals and expert knowledge")
    
    tip_categories = ["Crop Management", "Pest Control", "Soil Health", "Irrigation Techniques", 
                      "Harvest Best Practices", "Organic Farming", "Animal Husbandry", "Seasonal Planning"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox("Select Topic", tip_categories)
        
        if st.button("🎯 Generate Farming Tip", use_container_width=True, type="primary"):
            with st.spinner("🤔 Generating tip from knowledge base..."):
                prompt = f"""Based on agricultural best practices and uploaded farming manuals, provide a detailed, practical farming tip about: {selected_category}

The tip should:
1. Be specific and actionable
2. Include step-by-step instructions if applicable
3. Consider timing for {st.session_state.location}
4. Include warnings or cautions if relevant
5. Be suitable for small to medium-scale farmers

Format as:
- **Tip Title**: [Title]
- **Overview**: [Summary]
- **Steps**: [Instructions]
- **Best Time**: [Timing]
- **Expected Results**: [Outcomes]
- **Cautions**: [Warnings]"""
                
                response = st.session_state.agent.query(prompt, st.session_state.username, st.session_state.location)
                
                st.session_state.current_tip = {"category": selected_category, "content": response, "generated_at": datetime.now()}
                st.session_state.tips_generated += 1
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        uploads_dir = Path("uploads")
        doc_count = len(list(uploads_dir.glob("*"))) if uploads_dir.exists() else 0
        
        st.metric("📚 Knowledge Sources", doc_count)
        st.metric("📁 Categories", len(tip_categories))
        st.metric("✨ Tips Generated", st.session_state.tips_generated)
    
    if hasattr(st.session_state, 'current_tip') and st.session_state.current_tip:
        st.markdown("---")
        st.markdown("### 📝 Generated Farming Tip")
        
        tip = st.session_state.current_tip
        
        st.markdown(f"""
        <div class="info-card">
            <h4>🏷️ Category: {tip['category']}</h4>
            <p style="color: #666;">Generated: {tip['generated_at'].strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(tip['content'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Save Tip", use_container_width=True):
                tips_dir = Path("saved_tips")
                tips_dir.mkdir(exist_ok=True)
                filename = f"{tip['category'].replace(' ', '_')}_{tip['generated_at'].strftime('%Y%m%d_%H%M%S')}.txt"
                filepath = tips_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Category: {tip['category']}\nGenerated: {tip['generated_at']}\n\n{tip['content']}")
                st.success(f"✅ Saved to: {filepath}")
        
        with col2:
            if st.button("🔄 Generate New", use_container_width=True):
                del st.session_state.current_tip
                st.rerun()
        
        with col3:
            st.download_button("📤 Export", tip['content'], f"{tip['category']}.txt", use_container_width=True)

def page_crop_health():
    st.markdown("## 📸 Crop Health Diagnosis")
    
    st.info("📷 Upload crop photos for AI-powered disease diagnosis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_image = st.file_uploader("Upload Crop Photo", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Diagnose", use_container_width=True, type="primary"):
                with st.spinner("🔬 Analyzing image..."):
                    # Save image
                    img_path = Path("crop_photos") / uploaded_image.name
                    with open(img_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    
                    st.success("✅ Image saved for analysis")
                    
                    # Simulate diagnosis
                    st.session_state.diagnosis = {
                        "disease": "Maize Leaf Blight",
                        "confidence": 87.5,
                        "severity": "Moderate",
                        "treatment": "Apply fungicide (Mancozeb) and remove affected leaves",
                        "prevention": "Ensure proper spacing and drainage"
                    }
    
    with col2:
        if hasattr(st.session_state, 'diagnosis'):
            diag = st.session_state.diagnosis
            
            st.markdown(f"""
            ### 🔬 Diagnosis Results
            
            **Disease Detected:** {diag['disease']}  
            **Confidence:** {diag['confidence']}%  
            **Severity:** {diag['severity']}
            
            #### 💊 Treatment
            {diag['treatment']}
            
            #### 🛡️ Prevention
            {diag['prevention']}
            """)
            
            if st.button("📋 Get Detailed Report", use_container_width=True):
                st.session_state.navigation = "AI Chat"
                st.rerun()

def page_community():
    st.markdown("## 👥 Community Forum")
    
    st.info("🌾 Connect with fellow farmers, share experiences, and learn together")
    
    # Post creation
    with st.expander("➕ Create New Post", expanded=False):
        with st.form("new_post"):
            post_content = st.text_area("What's on your mind?", height=100)
            post_category = st.selectbox("Category", ["General", "Crops", "Livestock", "Equipment", "Market Prices"])
            
            if st.form_submit_button("📤 Post", use_container_width=True):
                if post_content:
                    new_post = {
                        "username": st.session_state.username,
                        "content": post_content,
                        "category": post_category,
                        "likes": 0,
                        "comments": 0,
                        "timestamp": datetime.now()
                    }
                    st.success("✅ Post created!")
    
    # Sample community posts
    sample_posts = [
        {"username": "John Farmer", "content": "Great maize yields this year! Used the new hybrid seeds recommended by AgriBot. Highly recommend! 🌽", "likes": 12, "comments": 3, "time": "2 hours ago"},
        {"username": "Mary Agriculture", "content": "Anyone dealing with aphids on tobacco? What pesticides are you using?", "likes": 7, "comments": 5, "time": "5 hours ago"},
        {"username": "Peter Ranch", "content": "Just installed drip irrigation system. Water usage down 40%! Happy to share details.", "likes": 18, "comments": 8, "time": "1 day ago"},
    ]
    
    for post in sample_posts:
        st.markdown(f"""
        <div class="community-post">
            <div class="post-header">
                <div class="post-avatar">👤</div>
                <div>
                    <strong>{post['username']}</strong><br>
                    <small style="color: #666;">{post['time']}</small>
                </div>
            </div>
            <p>{post['content']}</p>
            <div class="post-actions">
                <span class="post-action">👍 {post['likes']} Likes</span>
                <span class="post-action">💬 {post['comments']} Comments</span>
                <span class="post-action">🔄 Share</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def page_knowledge_base():
    st.markdown("## 📚 Knowledge Base")
    
    tab1, tab2, tab3 = st.tabs(["📥 Download Resources", "📤 Upload Documents", "📊 Library Stats"])
    
    with tab1:
        st.markdown("### 📥 Downloadable Farming Resources")
        
        resources = [
            {"title": "Maize Farming Guide for Zimbabwe", "description": "Complete guide to maize cultivation", "category": "crop_farming", "file": "maize_farming_guide.pdf", "size": "2.5 MB", "downloads": 1234},
            {"title": "Fish Farming Starter Manual", "description": "Step-by-step tilapia fish farm guide", "category": "fish_farming", "file": "fish_farming_manual.pdf", "size": "1.8 MB", "downloads": 856},
            {"title": "Goat Farming Best Practices", "description": "Goat rearing and disease management", "category": "goat_farming", "file": "goat_farming_guide.pdf", "size": "3.2 MB", "downloads": 654},
        ]
        
        for resource in resources:
            with st.expander(f"📄 {resource['title']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Description:** {resource['description']}")
                    st.markdown(f"**Size:** {resource['size']} | **Downloads:** {resource['downloads']}")
                with col2:
                    st.button("⬇️ Download", key=f"dl_{resource['file']}", use_container_width=True)
    
    with tab2:
        st.markdown("### 📤 Upload Your Documents")
        
        uploaded_files = st.file_uploader("Choose files", type=['pdf', 'txt'], accept_multiple_files=True)
        
        if uploaded_files and st.button("🚀 Upload", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                result_msg, stats = st.session_state.agent.add_documents(uploaded_files)
                if stats.get("files_processed", 0) > 0:
                    st.success(result_msg)
                else:
                    st.error(result_msg)
    
    with tab3:
        st.markdown("### 📊 Knowledge Base Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        uploads_dir = Path("uploads")
        total_docs = len(list(uploads_dir.glob("*"))) if uploads_dir.exists() else 0
        
        with col1:
            st.metric("📄 Total Documents", total_docs)
        with col2:
            st.metric("📚 Available Resources", 5)
        with col3:
            st.metric("📥 Total Downloads", "5,876")
        with col4:
            st.metric("👥 Active Users", "1,234")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    load_custom_css()
    render_header()
    render_sidebar()
    
    # Offline indicator
    status_class = "online-badge" if st.session_state.online_status else "offline-badge"
    status_text = "🟢 Online" if st.session_state.online_status else "🔴 Offline Mode"
    st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    # Navigation
    if HAS_OPTION_MENU:
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "AI Chat", "Weather", "Calculator", "Farming Tips", "Crop Health", "Community", "Knowledge Base"],
            icons=["house", "chat-dots", "cloud-sun", "calculator", "lightbulb", "camera", "people", "book"],
            menu_icon="cast",
            default_index=["Dashboard", "AI Chat", "Weather", "Calculator", "Farming Tips", "Crop Health", "Community", "Knowledge Base"].index(st.session_state.navigation),
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#f5f5f5"},
                "icon": {"color": "#2e7d32", "font-size": "18px"}, 
                "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#e8f5e9"},
                "nav-link-selected": {"background-color": "#2e7d32"},
            }
        )
        st.session_state.navigation = selected
    else:
        selected = st.selectbox("Navigation", ["Dashboard", "AI Chat", "Weather", "Calculator", "Farming Tips", "Crop Health", "Community", "Knowledge Base"], 
                                index=["Dashboard", "AI Chat", "Weather", "Calculator", "Farming Tips", "Crop Health", "Community", "Knowledge Base"].index(st.session_state.navigation))
        st.session_state.navigation = selected
    
    # Route to pages
    pages = {
        "Dashboard": page_dashboard,
        "AI Chat": page_ai_chat,
        "Weather": page_weather,
        "Calculator": page_calculator,
        "Farming Tips": page_farming_tips,
        "Crop Health": page_crop_health,
        "Community": page_community,
        "Knowledge Base": page_knowledge_base
    }
    
    pages[selected]()
    
    # Footer
    st.markdown(f"""
    <div class="footer">
        <h3 style="color: #2e7d32;">🌾 {config.APP_NAME}</h3>
        <p style="color: #666; margin: 10px 0;">{config.APP_TAGLINE}</p>
        <p style="color: #888; font-size: 0.9rem;">
            © 2024 {config.COMPANY_NAME} | All rights reserved<br>
            <a href="{config.COMPANY_WEBSITE}" style="color: #2e7d32;">Website</a> | 
            <a href="mailto:{config.SUPPORT_EMAIL}" style="color: #7e57c2;">Email</a> |
            <a href="https://wa.me/{config.WHATSAPP.replace('+', '').replace(' ', '')}" style="color: #25D366;">WhatsApp</a>
        </p>
        <p style="color: #999; font-size: 0.8rem; margin-top: 10px;">
            Version {config.VERSION} | Made with ❤️ for farmers | 🌍 Serving Zimbabwe & Southern Africa
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()