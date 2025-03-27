# 🐼 Panda AI Virtual Assistant

Panda AI is an intelligent virtual assistant capable of handling tasks like weather updates, news, calculations, reminders, and AI-powered conversations.

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/panda_ai.git
cd panda_ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` file

Create an `.env` file with the following contents:

```bash
echo "OPENWEATHERMAP_API_KEY=your_api_key_here" > .env
echo "NEWSAPI_KEY=your_api_key_here" >> .env
echo "OPENAI_API_KEY=your_api_key_here" >> .env
echo "WOLFRAMALPHA_APP_ID=your_app_id_here" >> .env
echo "EMAIL_ADDRESS=your_email@gmail.com" >> .env
echo "EMAIL_PASSWORD=your_app_password" >> .env
```

### 4. Run the application

Start the application by running the following command:

```bash
python panda_ai.py
```

You can now access it at [http://localhost:5000](http://localhost:5000).

## 🎯 Supported Commands

### 🌍 General
- "Hello" / "Hi" / "Hey"
- "What's your name?"
- "Who created you?"
- "What can you do?"

### 🕒 Time & Date
- "What time is it?"
- "What's today's date?"
- "Time in [city]?"

### 🌤️ Weather
- "Weather in [city]"
- "Will it rain today?"

### 📰 News
- "Latest news"
- "News about [topic]"

### 💰 Finance
- "Bitcoin price"
- "Stock price of [company]"

### 🧮 Math
- "Calculate [expression]"
- "Solve [equation]"

### ⏰ Reminders
- "Remind me to [task] at [time]"
- "Set timer for [X] minutes"

### 📧 Email
- "Email [recipient] that [message]"

### 🎉 Entertainment
- "Tell me a joke"
- "Play [song] on YouTube"
- "Give me a dare"

### 📝 Productivity
- "Summarize [text]"
- "Analyze sentiment of [text]"

### 🤖 AI Chat
- "Explain [concept]" (Uses OpenAI API)

## 🔧 Configuration

Make sure to configure the following API keys by creating a `.env` file as shown in the Quick Start section:

- `OPENWEATHERMAP_API_KEY` – Get your API key from [OpenWeatherMap](https://openweathermap.org/api).
- `NEWSAPI_KEY` – Get your API key from [NewsAPI](https://newsapi.org/).
- `OPENAI_API_KEY` – Get your API key from [OpenAI](https://beta.openai.com/signup/).
- `WOLFRAMALPHA_APP_ID` – Get your App ID from [WolframAlpha](https://products.wolframalpha.com/api/).
- `EMAIL_ADDRESS` – Your email address (used for sending emails).
- `EMAIL_PASSWORD` – Your email password (or app-specific password).
