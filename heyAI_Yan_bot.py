import logging
import os
import requests
import asyncio

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

# Yandex GPT API URL
YANDEX_GPT_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_yandex_gpt_response(prompt_text: str) -> str:
    """
    Sends a request to the Yandex GPT API and returns the response.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_API_KEY}"
    }
    
    payload = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": "2000"
        },
        "messages": [
            {
                "role": "system",
                "text": "Ты — умный и полезный AI-агент."
            },
            {
                "role": "user",
                "text": prompt_text
            }
        ]
    }

    try:
        response = requests.post(YANDEX_GPT_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        
        # Extract the content from the response
        if "result" in data and "alternatives" in data["result"] and len(data["result"]["alternatives"]) > 0:
            return data["result"]["alternatives"][0]["message"]["text"]
        else:
            logger.error(f"Unexpected API response structure: {data}")
            return "Извините, я не смог обработать ответ от AI."

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Yandex GPT API: {e}")
        return "Произошла ошибка при обращении к AI. Пожалуйста, попробуйте позже."
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "Произошла непредвиденная ошибка."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler for the /start command.
    """
    await update.message.reply_text(
        "Привет! Я AI-агент, подключенный к YandexGPT. Задайте мне любой вопрос."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler for regular text messages.
    """
    user_message = update.message.text
    
    # Inform the user that the bot is thinking
    thinking_message = await update.message.reply_text("Думаю...")
    
    # Get response from Yandex GPT
    ai_response = get_yandex_gpt_response(user_message)
    
    # Edit the "thinking" message with the actual response
    await context.bot.edit_message_text(
        text=ai_response,
        chat_id=update.effective_chat.id,
        message_id=thinking_message.message_id
    )

def main() -> None:
    """
    Starts the bot.
    """
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    logger.info("Starting bot...")
    application.run_polling()

if __name__ == "__main__":
    main()