import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PLATFORM_CONFIG = {
    "Twitter/X": {"max_length": 280, "tone": "concise"},
    "Facebook": {"max_length": 800, "tone": "friendly"},
    "LinkedIn": {"max_length": 1200, "tone": "professional"},
    "Instagram": {"max_length": 300, "tone": "creative"},
    "General": {"max_length": 500, "tone": "neutral"}
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_platform" not in st.session_state:
    st.session_state.selected_platform = "General"
if "ratings" not in st.session_state:
    st.session_state.ratings = []

def generate_ai_response(user_prompt: str, platform: str) -> str:
    """Generate AI response using Groq API with LangChain"""
    try:
        platform_rules = PLATFORM_CONFIG[platform]
        
        # LangChain prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """You are a professional social media manager assistant. 
             Respond to the user's query in {tone} tone. 
             Keep response under {max_length} characters. 
             Include platform-specific best practices."""),
            ("user", "{input}")
        ])
        
        # Groq model initialization
        model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-specdec",
            temperature=0.7,
            max_tokens=min(platform_rules['max_length'], 1024)
        )
        
        # Create chain
        chain = prompt | model | StrOutputParser()
        
        response = chain.invoke({
            "input": user_prompt,
            "tone": platform_rules['tone'],
            "max_length": platform_rules['max_length']
        })
        
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def display_analytics():
    """Display chatbot performance metrics"""
    st.sidebar.subheader("Performance Analytics")
    st.sidebar.metric("Total Interactions", len(st.session_state.messages))
    if st.session_state.ratings:
        avg_rating = sum(st.session_state.ratings)/len(st.session_state.ratings)
        st.sidebar.metric("Average Rating", f"{avg_rating:.1f} ⭐")
    else:
        st.sidebar.metric("Average Rating", "N/A")

def main():
    st.set_page_config(page_title="Social Media AI Assistant", layout="wide")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        selected_platform = st.selectbox(
            "Select Social Platform",
            options=list(PLATFORM_CONFIG.keys()),
            index=list(PLATFORM_CONFIG.keys()).index(st.session_state.selected_platform)
        )
        st.session_state.selected_platform = selected_platform
        
        tone_preference = st.selectbox(
            "Tone Preference",
            ["Professional", "Friendly", "Casual", "Persuasive"],
            index=0
        )
        
        st.divider()
        display_analytics()
    
    # Main Chat Interface
    st.title(f"Social Media AI Assistant - {st.session_state.selected_platform}")
    
    # Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(f"{message['platform']} - {message['time']}")
    
    # User Input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to history
        user_msg = {
            "role": "user",
            "content": prompt,
            "platform": st.session_state.selected_platform,
            "time": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.messages.append(user_msg)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"{user_msg['platform']} - {user_msg['time']}")
        
        # Generate AI response
        with st.spinner("Analyzing and crafting response..."):
            ai_response = generate_ai_response(prompt, st.session_state.selected_platform)
        
        # Add AI response to history
        ai_msg = {
            "role": "assistant",
            "content": ai_response,
            "platform": st.session_state.selected_platform,
            "time": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.messages.append(ai_msg)
        
        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            st.caption(f"{ai_msg['platform']} - {ai_msg['time']}")
        
        # Feedback system
        if "rate_response" not in st.session_state:
            st.session_state.rate_response = None
        
        selected_rating = st.radio("Rate response:", [1, 2, 3, 4, 5], index=None, key="rate_response")
        if selected_rating:
            st.session_state.ratings.append(selected_rating)
            st.success("Thanks for your feedback!")
    
    # Add reset button
    if st.session_state.messages:
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.ratings = []
            st.rerun()

if __name__ == "__main__":
    main()
