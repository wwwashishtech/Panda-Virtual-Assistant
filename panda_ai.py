import os
os.environ['DISPLAY'] = ':0'  # Mock display environment
import threading
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_cors import CORS
import wikipediaapi
import webbrowser
import datetime
import random
import logging
import requests
import re
import json
import pytz
from pytz import timezone
import openai  
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

import pyjokes
import wolframalpha
import subprocess
from newsapi import NewsApiClient
import yfinance as yf
from bs4 import BeautifulSoup
from string import punctuation
from heapq import nlargest
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from werkzeug.utils import secure_filename
import uuid
from functools import wraps
import ssl
from email.utils import formataddr
import emoji  # To help with emoji removal

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app, supports_credentials=True)  
app.secret_key = os.getenv('FLASK_SECRET_KEY', str(uuid.uuid4()))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
app.config['UPLOAD_FOLDER'] = 'uploads'


# Configure session
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=1)

# API Keys
CRYPTO_API_URL = 'https://api.coingecko.com/api/v3/simple/price'
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "9fd3e42f107014b62cb7b2bbfcbea1bd")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "ed80d7e95b9d46c987d2e4a3f59e2436")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "RMLdl0VrWBHua7rz1bykzGC0KQ8_BrzAUfDsy7XK")
WOLFRAM_ALPHA_APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID", "7RAV58-5VKAT936R5")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "ashishvishwakarma53871@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")

logging.basicConfig(filename='assistant.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize APIs with error handling
def initialize_apis():
    """Initialize all API clients with proper error handling."""
    apis = {}
    
    try:
        apis['newsapi'] = NewsApiClient(api_key=NEWS_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize NewsAPI: {e}")
        apis['newsapi'] = None

    try:
        if WOLFRAM_ALPHA_APP_ID:
            apis['wolfram'] = wolframalpha.Client(WOLFRAM_ALPHA_APP_ID)
        else:
            apis['wolfram'] = None
    except Exception as e:
        logging.error(f"Failed to initialize WolframAlpha: {e}")
        apis['wolfram'] = None

    try:
        openai.api_key = OPENAI_API_KEY
        apis['openai'] = openai.api_key is not None
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI: {e}")
        apis['openai'] = False

    return apis

apis = initialize_apis()

# Initialize NLP tools
def initialize_nlp():
    """Initialize NLP tools with error handling."""
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)  # Add this line
        
        sia = SentimentIntensityAnalyzer()
        STOP_WORDS = set(nltk.corpus.stopwords.words('english') + list(punctuation))
        return sia, STOP_WORDS
    except Exception as e:
        logging.error(f"Failed to initialize NLP tools: {e}")
        raise

# Constants
DEFAULT_CITY = 'Mumbai'
TIME_FORMAT = '%I:%M %p'
DATE_FORMAT = '%B %d, %Y'
WEATHER_UNITS = 'metric'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MAX_SUMMARY_LENGTH = 500

# Enhanced logging configuration
def setup_logging():
    """Configure comprehensive logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('assistant.log'),
            logging.StreamHandler()
        ]
    )
    # Suppress noisy library logs
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Wikipedia API setup with proper user agent
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="PandaAI/2.0",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    timeout=10
)

# Enhanced User Session Management
class UserSession:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset user session to default values."""
        self.name = None
        self.preferences = {
            'news_categories': ['technology', 'business'],
            'default_city': DEFAULT_CITY,
            'time_format': TIME_FORMAT,
            'temperature_unit': 'Celsius',
            'voice_enabled': False,
            'dark_mode': False,
            'language': 'en'
        }
        self.reminders = []
        self.timers = []
        self.conversation_history = []
        self.favorite_commands = {}
        self.auth_token = str(uuid.uuid4())
    
    def add_to_history(self, command, response):
        self.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'command': command,
            'response': response
        })
        # Keep only the last 100 conversations
        if len(self.conversation_history) > 100:
            self.conversation_history.pop(0)
        
        # Track favorite commands
        simplified_cmd = command.lower().strip()
        self.favorite_commands[simplified_cmd] = self.favorite_commands.get(simplified_cmd, 0) + 1
    
    def get_favorite_commands(self, top_n=5):
        return sorted(self.favorite_commands.items(), key=lambda x: x[1], reverse=True)[:top_n]

user_session = UserSession()

# Helper functions
def wish_me(user_name="User"):
    """Generate personalized greeting based on time of day."""
    hour = datetime.datetime.now().hour
    greetings = {
        "morning": f'Good morning, {user_name}! 🌞 How can I assist you today?',
        "afternoon": f'Good afternoon, {user_name}! 🌤️ What can I do for you?',
        "evening": f'Good evening, {user_name}! 🌙 How may I help you?',
        "night": f'Good night, {user_name}! 🌌 If you need anything before bed, just ask!'
    }
    
    if 5 <= hour < 12:
        return greetings["morning"]
    elif 12 <= hour < 17:
        return greetings["afternoon"]
    elif 17 <= hour < 21:
        return greetings["evening"]
    else:
        return greetings["night"]

def get_current_time(city=None):
    """Get current time, optionally for a specific city."""
    if city:
        try:
            geolocator = Nominatim(user_agent="panda_ai")
            location = geolocator.geocode(city)
            if location:
                tf = TimezoneFinder()
                time_zone = tf.timezone_at(lng=location.longitude, lat=location.latitude)
                if time_zone:
                    tz = timezone(time_zone)
                    return datetime.datetime.now(tz).strftime(TIME_FORMAT)
        except Exception as e:
            logging.warning(f"Couldn't get time for {city}: {e}")
    
    return datetime.datetime.now().strftime(TIME_FORMAT)

def get_current_date():
    """Get current date in a readable format."""
    return datetime.datetime.now().strftime(DATE_FORMAT)

def get_day_of_week():
    """Get current day of the week."""
    return datetime.datetime.now().strftime('%A')

def extract_entity(command, patterns):
    """Extract entities from command using regex patterns."""
    for pattern in patterns:
        match = re.search(pattern, command, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def analyze_sentiment(text):
    """Analyze sentiment of text using NLTK's VADER."""
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "positive"
    elif scores['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

def summarize_text(text, percent=0.3):
    """Generate summary of text using NLTK."""
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    
    word_frequencies = {}
    for word in words:
        if word.lower() not in STOP_WORDS and word.lower() not in punctuation:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    max_frequency = max(word_frequencies.values()) if word_frequencies else 1
    
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency
        
    sentence_scores = {}
    for sent in sentences:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
                    
    select_length = int(len(sentences)*percent)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Command handlers
def handle_weather(command):
    """Handle weather-related queries."""
    city_patterns = [
        r"weather(?: in| at)?\s*(.+)",
        r"what's the weather like in (.+)",
        r"how's the weather in (.+)",
        r"temperature in (.+)"
    ]
    city = extract_entity(command, city_patterns) or user_session.preferences['default_city']
    return get_weather_info(city)

def get_fun_fact():
    """Provide a random fun fact."""
    facts = [
        "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly good to eat.",
        "Bananas are berries, but strawberries aren't.",
        "A group of flamingos is called a 'flamboyance.'",
        "Wombat poop is cube-shaped.",
        "Octopuses have three hearts."
    ]
    return random.choice(facts)

def get_weather_info(city):
    """Get weather information for a city."""
    try:
        params = {
            'q': city,
            'appid': WEATHER_API_KEY,
            'units': WEATHER_UNITS,
            'lang': 'en'
        }
        response = requests.get('https://api.openweathermap.org/data/2.5/weather', params=params)
        data = response.json()

        if response.status_code == 200:
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            desc = data['weather'][0]['description']
            city_name = data.get('name', city)
            
            weather_icons = {
                'clear': '☀️',
                'clouds': '☁️',
                'rain': '🌧️',
                'snow': '❄️',
                'thunderstorm': '⛈️',
                'drizzle': '🌦️',
                'mist': '🌫️',
                'fog': '🌁',
                'tornado': '🌪️'
            }
            icon = weather_icons.get(data['weather'][0]['main'].lower(), '🌤️')
            
            return (f"🌦️ Weather in {city_name}: {icon} {temp}°C (feels like {feels_like}°C)\n"
                    f"• Conditions: {desc.capitalize()}\n"
                    f"• Humidity: {humidity}%\n"
                    f"• Wind: {wind_speed} km/h")
        
        return f"❌ Couldn't retrieve weather for {city}. Error: {data.get('message', 'Unknown error')}"
    except Exception as e:
        logging.error(f"Weather API error: {e}")
        return "⚠️ Error fetching weather data. Please try again later."

def handle_news(command):
    """Handle news-related queries."""
    if not apis['newsapi']:
        return "News API is not configured properly."
        
    category_patterns = [
        r"news about (.+)",
        r"what's new in (.+)",
        r"update me about (.+)",
        r"latest on (.+)"
    ]
    category = extract_entity(command, category_patterns)
    return get_news_updates(category)

def get_news_updates(category=None):
    """Get news updates for a category or default categories."""
    try:
        if category:
            data = apis['newsapi'].get_top_headlines(
                q=category,
                language='en',
                page_size=5
            )
            title = f"Latest news about {category}"
        else:
            data = apis['newsapi'].get_top_headlines(
                category=user_session.preferences['news_categories'][0],
                language='en',
                page_size=5
            )
            title = "Latest news headlines"
        
        if data['status'] == 'ok' and data['totalResults'] > 0:
            headlines = []
            for i, article in enumerate(data['articles'][:5], 1):
                source = article['source']['name']
                title_text = article['title'].split(' - ')[0]
                headlines.append(f"{i}. {title_text} ({source})")
            return f"📰 {title}:\n" + "\n".join(headlines)
        return "📰 No news articles found for your query."
    except Exception as e:
        logging.error(f"News API error: {e}")
        return "⚠️ Error fetching news updates. Please try again later."

def handle_crypto(command):
    """Handle cryptocurrency queries."""
    crypto_patterns = [
        r"price of (.+)",
        r"how much is (.+) worth",
        r"value of (.+)",
        r"crypto price for (.+)"
    ]
    crypto = extract_entity(command, crypto_patterns)
    if crypto:
        return get_crypto_price(crypto)
    return "Please specify a cryptocurrency."

def get_crypto_price(crypto):
    """Get cryptocurrency price information."""
    try:
        crypto_id = crypto.lower()
        response = requests.get(
            'https://api.coingecko.com/api/v3/simple/price',
            params={
                'ids': crypto_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true'
            }
        )
        data = response.json()
        
        if crypto_id in data:
            price = data[crypto_id]['usd']
            change = data[crypto_id].get('usd_24h_change', 0)
            market_cap = data[crypto_id].get('usd_market_cap', 0)
            
            change_emoji = '📈' if change >= 0 else '📉'
            market_cap_str = f"Market Cap: ${market_cap:,.2f}\n" if market_cap > 0 else ""
            
            return (f"💰 {crypto.capitalize()}:\n"
                    f"• Price: ${price:,.2f}\n"
                    f"• 24h Change: {change_emoji} {abs(change):.2f}%\n"
                    f"{market_cap_str}")
        return f"❌ Couldn't find price for {crypto}. Please check the cryptocurrency name."
    except Exception as e:
        logging.error(f"Crypto API error: {e}")
        return "⚠️ Error fetching cryptocurrency data. Please try again later."

def handle_stock(command):
    """Handle stock market queries."""
    stock_patterns = [
        r"stock price of (.+)",
        r"how is (.+) stock doing",
        r"value of (.+) stock",
        r"share price of (.+)"
    ]
    stock = extract_entity(command, stock_patterns)
    if stock:
        return get_stock_price(stock)
    return "Please specify a stock symbol."

def get_stock_price(symbol):
    """Get stock market information."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='1d')
        
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = ((current_price - prev_close) / prev_close) * 100
            change_emoji = '📈' if change >= 0 else '📉'
            
            # Get additional info
            info = stock.info
            company_name = info.get('longName', symbol.upper())
            day_range = f"{info.get('regularMarketDayLow', 'N/A')}-{info.get('regularMarketDayHigh', 'N/A')}"
            year_range = f"{info.get('fiftyTwoWeekLow', 'N/A')}-{info.get('fiftyTwoWeekHigh', 'N/A')}"
            
            return (f"📊 {company_name} ({symbol.upper()}):\n"
                    f"• Price: ${current_price:.2f} {change_emoji} {abs(change):.2f}%\n"
                    f"• Day Range: {day_range}\n"
                    f"• 52-Week Range: {year_range}")
        return f"❌ Couldn't find data for {symbol}. Please check the stock symbol."
    except Exception as e:
        logging.error(f"Stock API error: {e}")
        return "⚠️ Error fetching stock data. Please try again later."

def handle_calculations(command):
    """Handle mathematical calculations with multiple fallback methods."""
    try:
        # Clean and normalize the command
        clean_cmd = command.strip()
        
        # First try to evaluate as direct math expression (like "2+2")
        if re.fullmatch(r'^[\d\+\-\*\/\^\(\)\. ]+$', clean_cmd):
            try:
                # Safer evaluation using a custom function
                def safe_eval(expr):
                    # Only allow basic math operators and numbers
                    allowed_chars = set('0123456789+-*/.()^ ')
                    if not all(c in allowed_chars for c in expr):
                        raise ValueError("Invalid characters in expression")
                    # Replace ^ with ** for exponentiation
                    expr = expr.replace('^', '**').replace(' ', '')
                    # Use a restricted eval with only math operations
                    return eval(expr, {'__builtins__': None}, {})
                
                result = safe_eval(clean_cmd)
                return f"🐼 Bamboo Math says: {clean_cmd} = {result}"
            except Exception as e:
                logging.warning(f"Basic math evaluation failed: {e}")
                # Continue to Wolfram fallback
        
        # Try Wolfram Alpha if available
        if apis.get('wolfram'):
            try:
                res = apis['wolfram'].query(clean_cmd)
                answer = next(res.results).text
                return f"🎋 Wolfram says: {answer}"
            except Exception as e:
                logging.warning(f"Wolfram Alpha failed: {e}")
                # Continue to final fallback
        
        # Final fallback - try to extract math expression from natural language
        math_patterns = [
            r"calculate (.+)", 
            r"what is (.+)",
            r"how much is (.+)",
            r"solve (.+)",
            r"what's (.+)",
            r"compute (.+)"
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, clean_cmd, re.IGNORECASE)
            if match:
                math_expr = match.group(1).strip()
                if re.fullmatch(r'^[\d\+\-\*\/\^\(\)\. ]+$', math_expr):
                    try:
                        result = safe_eval(math_expr)
                        return f"🐼 Bamboo Math says: {math_expr} = {result}"
                    except:
                        break
        
        return "I couldn't solve that math problem. Try something like '5+5' or 'what is 2+2?' 🎍"
    
    except Exception as e:
        logging.error(f"Calculation error: {e}")
        return "My bamboo calculator malfunctioned! Try again? 🎍"


def handle_ai_chat(prompt):
    """Handle general chat using OpenAI."""
    if not apis['openai']:
        return "AI chat feature is not configured."
    
    try:
        # Include conversation history for context
        messages = [{"role": "system", "content": "You are Panda, a helpful AI assistant."}]
        
        # Add last few conversation turns for context
        for conv in user_session.conversation_history[-3:]:
            messages.append({"role": "user", "content": conv['command']})
            messages.append({"role": "assistant", "content": conv['response']})
        
        messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "I encountered an error while processing your request."

def handle_email(command):
    """Handle email-related commands."""
    if not EMAIL_PASSWORD or not EMAIL_ADDRESS:
        return "Email functionality is not configured."
    
    email_patterns = [
        r"email (.+) that (.+)",
        r"send an email to (.+) saying (.+)",
        r"mail (.+) about (.+)",
        r"send (.+) an email about (.+)"
    ]
    match = extract_entity(command, email_patterns)
    if match:
        recipient, body = match
        subject = f"Message from {user_session.name or 'Panda AI'}"
        return send_email(recipient, subject, body)
    return "I couldn't understand the email details. Format: 'email [recipient] that [message]'"

def send_email(to, subject, body):
    """Send email using SMTP."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return f"Email sent to {to} successfully."
    except Exception as e:
        logging.error(f"Email error: {e}")
        return "Failed to send email. Please check the email configuration."

def handle_reminders(command):
    """Handle reminder creation."""
    reminder_patterns = [
        r"remind me to (.+) at (.+)",
        r"set a reminder to (.+) at (.+)",
        r"remember to (.+) at (.+)",
        r"reminder for (.+) at (.+)"
    ]
    match = extract_entity(command, reminder_patterns)
    if match:
        task, time_str = match
        return set_reminder(task, time_str)
    return "I couldn't understand the reminder details. Format: 'remind me to [task] at [time]'"



def set_reminder(task, time_str):
    """Set a reminder for a specific time."""
    try:
        reminder_time = datetime.datetime.strptime(time_str, '%I:%M %p').time()
        now = datetime.datetime.now().time()
        
        if reminder_time > now:
            delay = (datetime.datetime.combine(datetime.date.today(), reminder_time) - \
                    datetime.datetime.combine(datetime.date.today(), now)).seconds
            threading.Timer(delay, lambda: print(f"REMINDER: {task}")).start()
            user_session.reminders.append({"task": task, "time": time_str})
            return f"⏰ Reminder set for {time_str}: {task}"
        else:
            return "That time has already passed today. Please set a future time."
    except ValueError:
        return "Please specify time in HH:MM AM/PM format (e.g., 3:30 PM)."

def handle_timer(command):
    """Handle timer creation."""
    timer_patterns = [
        r"set a timer for (\d+) (seconds|minutes|hours)",
        r"timer for (\d+) (seconds|minutes|hours)",
        r"countdown for (\d+) (seconds|minutes|hours)",
        r"start a (\d+) (second|minute|hour) timer"
    ]
    match = extract_entity(command, timer_patterns)
    if match:
        duration, unit = match
        return set_timer(int(duration), unit)
    return "I couldn't understand the timer details. Format: 'set a timer for [number] [seconds/minutes/hours]'"


def handle_summarize(command):
    """Handle text summarization requests."""
    text_patterns = [
        r"summarize (.+)",
        r"summarize this: (.+)",
        r"give me a summary of (.+)"
    ]
    text = extract_entity(command, text_patterns)
    if text:
        return summarize_text(text)
    return "Please provide text to summarize. Format: 'summarize [text]'"

def handle_sentiment(command):
    """Handle sentiment analysis requests."""
    text_patterns = [
        r"analyze sentiment of (.+)",
        r"what's the sentiment of (.+)",
        r"how does this feel: (.+)"
    ]
    text = extract_entity(command, text_patterns)
    if text:
        sentiment = analyze_sentiment(text)
        return f"The sentiment is {sentiment}."
    return "Please provide text to analyze. Format: 'analyze sentiment of [text]'"

def search_wikipedia(search_term):
    """Search Wikipedia for information."""
    try:
        page = wiki.page(search_term)
        if page.exists():
            summary = page.summary[:500] + '...' if len(page.summary) > 500 else page.summary
            return f"📚 Wikipedia: {search_term}\n{summary}\nRead more: {page.fullurl}"
        return f"❌ No Wikipedia article found for '{search_term}'."
    except Exception as e:
        logging.error(f"Wikipedia error: {e}")
        return "⚠️ Error searching Wikipedia. Please try again later."

def play_on_youtube(query):
    """Play a video on YouTube."""
    try:
        kit.playonyt(query)
        return f"🎵 Playing '{query}' on YouTube..."
    except Exception as e:
        logging.error(f"YouTube error: {e}")
        return "⚠️ Error playing on YouTube. Please try again later."

def search_google(query):
    """Search Google for a query."""
    try:
        kit.search(query)
        return f"🔍 Searching Google for '{query}'..."
    except Exception as e:
        logging.error(f"Google search error: {e}")
        return "⚠️ Error searching Google. Please try again later."

def get_joke():
    """Get a random joke."""
    try:
        joke = pyjokes.get_joke()
        return f"😂 Joke: {joke}"
    except:
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the math book look sad? Because it had too many problems.",
            "Why don't skeletons fight each other? They don't have the guts.",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        return f"😂 Joke: {random.choice(jokes)}"

def get_quote():
    """Get an inspirational quote."""
    try:
        response = requests.get("https://api.quotable.io/random")
        if response.status_code == 200:
            quote = response.json()
            return f"💬 \"{quote['content']}\" — {quote['author']}"
        return "💬 \"The only way to do great work is to love what you do.\" — Steve Jobs"
    except:
        return "💬 \"Life is what happens when you're busy making other plans.\" — John Lennon"

def get_dare():
    """Get a random dare."""
    dares = [
        "Do 20 jumping jacks while keeping eye contact.",
    "Whisper something naughty to your partner.",
    "Do your best seductive dance for 30 seconds.",
    "Improvise a sexy poem and recite it.",
    "Let your partner blindfold you for the next dare.",
    "Do a striptease for your partner (without full nudity).",
    "Give your partner a 5-minute foot rub.",
    "Tell your partner your favorite memory of them.",
    "Act like a sexy cat for the next minute.",
    "Slowly and seductively remove one item of clothing.",
    "Send your partner a sexy text message.",
    "Let your partner choose your next outfit.",
    "Do your best impression of a seductive movie scene.",
    "Let your partner write something on your body with a marker.",
    "Give your partner a passionate kiss for 20 seconds.",
    "Dance around the room with no music.",
    "Re-enact a famous movie love scene.",
    "Try to make your partner laugh with a silly sexy dance.",
    "Whisper your deepest secret to your partner.",
    "Let your partner take a photo of you in your current pose.",
    "Do a seductive walk from one side of the room to the other.",
    "Act out your favorite fantasy without speaking.",
    "Give your partner a long, lingering hug.",
    "Pretend you're on a romantic date with your partner and act it out.",
    "Improvise a sensual massage for your partner's back.",
    "Let your partner feed you something with their hands.",
    "Do 10 squats while maintaining eye contact.",
    "Send a flirty message to your partner right now.",
    "Do a sexy runway walk across the room.",
    "Let your partner blindfold you and guess the next sensation.",
    "Pretend you're a character from a romantic movie for the next minute.",
    "Let your partner choose a body part for you to kiss.",
    "Have your partner write something romantic on your body.",
    "Give your partner a slow kiss on the neck.",
    "Let your partner tie your hands lightly for the next dare.",
    "Kiss your partner's neck for 15 seconds.",
    "Act like you're meeting your partner for the first time and flirt.",
    "Sit on your partner’s lap and give them a gentle kiss.",
    "Take a seductive selfie and show it to your partner.",
    "Walk around the room like you're on a catwalk.",
    "Give your partner a playful or passionate peck on the lips.",
    "Describe your most romantic fantasy to your partner.",
    "Let your partner kiss you anywhere (within limits).",
    "Pretend you're in a passionate embrace and act it out.",
    "Send a flirtatious selfie to your partner with a smile.",
    "Do a sexy dance to your favorite song without music.",
    "Tell your partner your most secret desire.",
    "Create a new sexy move and teach it to your partner.",
    "Act like you're auditioning for a romance movie scene.",
    "Let your partner pick a song and dedicate it to you.",
    "Do a seductive pose and hold it for 30 seconds.",
    "Create a sexy love letter to your partner on the spot.",
    "Take turns giving each other a soft kiss in different places.",
    "Pretend you're a character from a romantic novel.",
    "Draw a heart or something sexy on your partner's skin with your finger.",
    "Give your partner a surprise kiss while they're not expecting it.",
    "Let your partner lead you in a slow, romantic dance.",
    "Do your best impression of a sensual model for your partner.",
    "Give your partner a kiss on the lips and hold it for 5 seconds.",
    "Act out a slow motion kiss with your partner.",
    "Pretend you're giving a love speech to your partner and tell them why they're special.",
    "Slowly trace your finger along your partner's arm or back.",
    "Let your partner choose a romantic song to dance to.",
    "Give your partner a gentle kiss on the hand.",
    "Give your partner a loving hug and whisper something sweet in their ear.",
    "Reenact a romantic scene from a movie you've both seen together.",
    "Kiss your partner's lips, then neck, then lips again.",
    "Do a seductive walk toward your partner and give them a kiss.",
    "Pretend you're a professional dancer and teach your partner a sexy move.",
    "Give your partner a surprise passionate kiss when they’ve just woken up."
            
];
    return f"🎯 Dare: {random.choice(dares)}"

def open_application(command):
    """Open the requested application."""
    app_name = command.replace("open", "").strip().lower()
    apps = {
        "instagram": "https://www.instagram.com",
        "google": "https://www.google.com",
        "facebook": "https://www.facebook.com",
        "youtube": "https://www.youtube.com",
        "linkedin": "https://www.linkedin.com",
        "github": "https://www.github.com",
        "stackoverflow": "https://stackoverflow.com",
        "amazon": "https://www.amazon.com",
        "flipkart": "https://www.flipkart.com",
        "whatsapp": "https://web.whatsapp.com",
        "chrome": "chrome",
        "command prompt": "cmd",
        "powershell": "powershell",
        "visual studio code": "code",
        "zoom": "zoom"
    }
    
    if app_name in apps:
        app_path = apps[app_name]
        try:
            if app_path.startswith('http'):
                webbrowser.open(app_path)
            else:
                subprocess.Popen(app_path)
            return f"🖥️ Opening {app_name.capitalize()}..."
        except Exception as e:
            logging.error(f"Error opening {app_name}: {e}")
            return f"❌ Failed to open {app_name}. Please try manually."
    else:
        return f"❌ I can't open '{app_name}'. Available apps: {', '.join(apps.keys())}"

def get_random_advice():
    """Get random advice."""
    advice_list = [
        "Take breaks often when working on the computer to rest your eyes.",
        "Stay hydrated throughout the day for better focus and energy.",
        "Set clear goals to keep yourself motivated and productive.",
        "Don't be afraid to ask for help when you're stuck.",
        "Practice mindfulness to reduce stress and increase your well-being.",
        "Keep a healthy work-life balance for long-term success.",
        "Take time to enjoy the small things in life.",
        "Always learn from your mistakes, they help you grow.",
        "Get enough sleep to recharge your body and mind.",
        "Stay positive, even when things don't go as planned."
    ]
    return f"💡 Advice: {random.choice(advice_list)}"


def get_motivational_quote():
    """Get a motivational quote."""
    quotes = [
        "The only limit to our realization of tomorrow is our doubts of today. — Franklin D. Roosevelt",
        "The only way to do great work is to love what you do. - Steve Jobs",
        "Life is what happens when you're busy making other plans. - John Lennon",
        "Get busy living or get busy dying. - Stephen King",
        "You only live once, but if you do it right, once is enough. - Mae West",
        "The purpose of our lives is to be happy. - Dalai Lama"
    ]
    return f"🌟 Motivational Quote: {random.choice(quotes)}"

def remove_emojis(text):
    """Remove emojis from a given text (for voice responses only)."""
    return emoji.replace_emoji(text, replace='')

def get_story():
    """Tell a random story."""
    stories = [
        "Once upon a time, in a faraway land, there was a small village where everyone was happy. One day, a stranger came...",
        "Long ago, in a kingdom by the sea, there lived a brave knight who set out on an adventure to rescue a captive princess..."
    ]
    return random.choice(stories)


def handle_command(command, is_voice=False):
    """Process user commands and return appropriate responses."""
    logging.info(f"Processing command: {command}")
    command = command.lower().strip()
    
    # Predefined responses with more variety and emojis
    responses = {
        "who created you": [
            "I was created by Ashish Vishwakarma, a talented developer with a passion for AI. 🤖",
            "My creator is Ashish Vishwakarma, who built me to be your helpful assistant. 💡",
            "Ashish Vishwakarma developed me to assist with various tasks and make life easier. 🌟"
        ],
        "what is your name": [
            "I'm Panda, your virtual assistant! 🐼✨",
            "You can call me Panda AI Assistant. 💬",
            "My name is Panda! How can I help you today? 🐾"
        ],
        "how are you": [
            "I'm functioning optimally, ready to assist you! ⚙️",
            "I'm just a program, but I'm here to help you with anything! 😊",
            "I don't have feelings, but I'm always ready to help! 💪"
        ],
        "thank you": [
            "You're welcome! Happy to help. 😊💖",
            "Anytime! Let me know if you need anything else. ✨",
            "My pleasure! Don't hesitate to ask for more help. 🤗"
        ],
        "what can you do": [
            "I can help with weather, news, calculations, reminders, emails, and much more! ☀️🌧️💻",
            "I can answer questions, set reminders, play music, and assist with daily tasks. 🎶⏰",
            "My capabilities include web searches, scheduling, information lookup, and more! 🌐"
        ],
        "tell me about yourself": [
            "I'm Panda AI, a virtual assistant designed to make your life easier. 🎉",
            "I'm an AI assistant here to help with information, tasks, and entertainment. 🎭",
            "I'm your digital helper, created to assist with various tasks and answer questions. 💡"
        ],
        "hello": [
            wish_me(user_session.name or "there"),
            f"Hi {user_session.name or 'there'}! How can I help you today? 👋",
            "Hello! What can I do for you? 😊"
        ],
        "hi": [
            wish_me(user_session.name or "there"),
            f"Hey {user_session.name or 'there'}! What's up? 🙌",
            "Hi there! How can I assist you? 🤖"
        ],
        "hey": [
            wish_me(user_session.name or "there"),
            f"Hey {user_session.name or 'there'}! What can I do for you? 🙏",
            "Hey! Ready to help with whatever you need. 💬"
        ],
        "good morning": [
            wish_me(user_session.name or "there"),
            f"Good morning {user_session.name or 'there'}! Ready for a great day? 🌞",
            "Morning! How can I start your day right? 🌻"
        ],
        "good afternoon": [
            wish_me(user_session.name or "there"),
            f"Good afternoon {user_session.name or 'there'}! How's your day going? 🌇",
            "Afternoon! What can I do for you? ☕"
        ],
        "good evening": [
            wish_me(user_session.name or "there"),
            f"Good evening {user_session.name or 'there'}! How can I help? 🌙",
            "Evening! What do you need assistance with? 🌜"
        ],
        "good night": [
            wish_me(user_session.name or "there"),
            f"Good night {user_session.name or 'there'}! Sleep well! 😴🌙",
            "Night night! Let me know if you need anything before bed. 🌙"
        ],
        "bye": [
            "Goodbye! Have a great day! 👋🌟",
            "See you later! Don't hesitate to return if you need more help. ✌️",
            "Bye! Come back anytime you need assistance. 🐾"
        ],
        "exit": [
            "Goodbye! Have a great day! 👋",
            "Closing down. Feel free to return anytime! 💤",
            "Signing off. See you soon! 🌟"
        ],
        "who are you": [
            "I'm Panda, your AI assistant! Here to help with anything you need. 🐼💻",
            "I'm your virtual assistant Panda, ready to assist you! 🤖💬",
            "I'm Panda AI, created to make your life easier. 🌟"
        ],
        "what can you do": [
            "I can assist you with a variety of tasks like answering questions, providing information, managing tasks, and much more. Let me know how I can help!",
            "I can help with tasks, answer questions, and much more!"
        ],
        "who am I": [
            "You are a valued user of the Panda Virtual Assistant! I'm here to support you with whatever you need.",
            "You're an important user of my services!"
        ],
        "what is your purpose": [
            "My purpose is to assist you with daily tasks, provide information, and make your life easier, all thanks to Ashish Vishwakarma's development skills.",
            "I'm here to make your life easier and assist with your daily tasks!"
        ],
        "thank you": [
            "You're welcome! I'm here whenever you need assistance.",
            "Anytime! I'm happy to help!"
        ],
        "what day is it": [
            "It's a beautiful day today! Let me know how I can assist you.",
            "Today is a great day! How can I help?"
        ],
        "what time is it": [
            "I can check the time for you. Just let me know if you need that info!",
            "Let me know if you want me to find out the current time!"
        ],
        "tell me a joke": [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call fake spaghetti? An impasta!"
        ],
        "what's your favorite color": [
            "I don't have a favorite color, but I think every color is beautiful!",
            "I think every color is special in its own way!"
        ],
        "do you have feelings": [
            "I don't have feelings like humans do, but I'm here to help you!",
            "I'm just a program, so I don't feel emotions, but I'm designed to assist you!"
        ],
        "what is the weather like today": [
            "I can help you find out the weather! Just let me know your location.",
            "Tell me your location, and I'll find the weather for you!"
        ],
        "what's your favorite food": [
            "I don't eat, but I hear pizza is a favorite for many people!",
            "I don't eat, but I think anything that brings people together is wonderful!"
        ],
        "how can I improve my productivity": [
            "To improve productivity, try setting clear goals, taking breaks, and eliminating distractions!",
            "Consider using tools like to-do lists, timers, and prioritizing your tasks!"
        ],
        "what are your hobbies": [
            "I don't have hobbies like humans, but I love helping you with yours!",
            "I enjoy assisting users with their questions and tasks!"
        ],
    }
    
    # Check predefined responses first
    for key in responses:
        if key in command:
            response = random.choice(responses[key])
            if is_voice:
                response = remove_emojis(response)
            user_session.add_to_history(command, response)
            return response
    
    # Handle specific commands
    try:
        if "roll a dice" in command:
            return f"You rolled a {random.randint(1, 6)}."

        if "search in wikipedia" in command:
            search_term = command.replace("search in wikipedia", "").strip()
            return search_wikipedia(search_term) if search_term else "Please specify what to search on Wikipedia."

        if "search in google" in command:
            search_term = command.replace("search in google", "").strip()
            if search_term:
                webbrowser.open(f"https://www.google.com/search?q={search_term}")
                return f"Searching for '{search_term}' on Google."
            return "Please specify what to search on Google."

        if "tell me a story" in command:
            return get_story()

        if "bitcoin price" in command or "ethereum price" in command:
            crypto = "bitcoin" if "bitcoin" in command else "ethereum"
            return get_crypto_price(crypto)

        if any(word in command for word in ['weather', 'temperature', 'forecast', 'rain']):
            response = handle_weather(command)
        
        if "fact" in command:
            return get_fun_fact()

        elif any(word in command for word in ['news', 'headlines', 'updates', 'headline']):
            response = handle_news(command)
        
        elif any(word in command for word in ['bitcoin', 'ethereum', 'crypto', 'cryptocurrency']):
            response = handle_crypto(command)
        
        elif any(word in command for word in ['stock', 'share price', 'market']):
            response = handle_stock(command)
        
        elif (any(word in command for word in ['calculate', 'math', 'what is', 'solve', 'equation', '+', '-', '*', '/']) 
              or re.match(r'^[\d\+\-\*\/\^\(\)\. ]+$', command.strip())):
            response = handle_calculations(command)
        
        elif any(word in command for word in ['remind', 'reminder']):
            response = handle_reminders(command)
        
        elif any(word in command for word in ['timer', 'countdown']):
            response = handle_timer(command)
        
        elif any(word in command for word in ['email', 'send mail', 'e-mail']):
            response = handle_email(command)
        
        elif any(word in command for word in ['play', 'youtube', 'song', 'music']):
            query = command.replace('play', '').replace('on youtube', '').strip()
            response = play_on_youtube(query)
        
        elif any(word in command for word in ['search', 'google']):
            query = command.replace('search', '').replace('on google', '').strip()
            response = search_google(query)
        
        elif any(word in command for word in ['wikipedia', 'wiki']):
            query = command.replace('search', '').replace('on wikipedia', '').strip()
            response = search_wikipedia(query)
        
        elif any(word in command for word in ['joke', 'funny', 'laugh']):
            response = get_joke()
        
        elif any(word in command for word in ['quote', 'inspiration', 'inspire']):
            response = get_quote()
        
        elif any(word in command for word in ['time', 'clock']):
            response = f"The current time is {get_current_time()}" + (" ⏰" if not is_voice else "")
        
        elif any(word in command for word in ['date', 'today']):
            response = f"Today is {get_current_date()}" + (" 📅" if not is_voice else "")
        
        elif any(word in command for word in ['day', 'today is']):
            response = f"Today is {get_day_of_week()}" + (" 📅" if not is_voice else "")
        
        elif any(word in command for word in ['open', 'launch']):
            response = open_application(command)
        
        elif any(word in command for word in ['dare', 'challenge']):
            response = get_dare()
        
        elif any(word in command for word in ['advice', 'suggestion']):
            response = get_random_advice()
        
        elif any(word in command for word in ['motivate', 'motivation']):
            response = get_motivational_quote()
        
        elif any(word in command for word in ['summarize', 'summary']):
            response = handle_summarize(command)
        
        elif any(word in command for word in ['sentiment', 'feel', 'feeling']):
            response = handle_sentiment(command)
        
        # Fallback to AI chat for complex queries if configured
        elif OPENAI_API_KEY:
            response = handle_ai_chat(command)
        else:
            response = "I'm not sure how to help with that. Could you try rephrasing?" + (" 🤔" if not is_voice else "")
        
        # Remove emojis for voice input
        if is_voice:
            response = remove_emojis(response)
        
        # Add to conversation history
        user_session.add_to_history(command, response)
        return response
            
    except Exception as e:
        logging.error(f"Command handling error: {e}")
        error_msg = "Sorry, I encountered an error processing your request." + (" 😞" if not is_voice else "")
        if is_voice:
            error_msg = remove_emojis(error_msg)
        user_session.add_to_history(command, error_msg)
        return error_msg

# Flask routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/command', methods=['POST'])
def process_command():
    """Process commands from the frontend."""
    try:
        data = request.get_json()
        command = data.get('command', '').strip()
        
        if not command:
            return jsonify({"response": "Please provide a command."})
        
        response = handle_command(command)
        return jsonify({
            "response": response,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({
            "response": "An error occurred processing your request.",
            "error": str(e)
        }), 500

# Update this route to match what your frontend is calling
@app.route('/voice-command', methods=['POST']) 
def voice_command():
    try:
        data = request.get_json()
        command = data.get('command', '').strip()
        
        if not command:
            return jsonify({"response": "Please provide a command."})
        
        response = handle_command(command)
        return jsonify({
            "response": response,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Voice command error: {e}")
        return jsonify({
            "response": "An error occurred processing your voice command.",
            "error": str(e)
        }), 500

@app.route('/api/set_preferences', methods=['POST'])
def set_preferences():
    """Update user preferences."""
    try:
        data = request.get_json()
        if 'name' in data:
            user_session.name = data['name']
        if 'preferences' in data:
            user_session.preferences.update(data['preferences'])
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Preferences error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history."""
    return jsonify({
        "history": user_session.conversation_history,
        "favorites": user_session.get_favorite_commands()
    })

@app.route('/api/reminders', methods=['GET'])
def get_reminders():
    """Get active reminders."""
    return jsonify({"reminders": user_session.reminders})

@app.route('/api/timers', methods=['GET'])
def get_timers():
    """Get active timers."""
    return jsonify({"timers": user_session.timers})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        try:
            if filename.endswith('.txt'):
                with open(filepath, 'r') as f:
                    content = f.read()
                summary = summarize_text(content)
                return jsonify({
                    "summary": summary,
                    "sentiment": analyze_sentiment(content)
                })
            else:
                return jsonify({"message": "File uploaded successfully", "filename": filename})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    port = int(os.environ.get('PORT', 10000))  # Render's default port
    app.run(host='0.0.0.0', port=port)
