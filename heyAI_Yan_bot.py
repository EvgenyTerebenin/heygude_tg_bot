import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4

import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables
load_dotenv()


# Configuration
class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
    YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
    YANDEX_GPT_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    REQUEST_TIMEOUT = 30
    MAX_TOKENS = 2000
    TEMPERATURE = 0.6


# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class YandexGPTClient:
    """Client for Yandex GPT API interactions."""

    def __init__(self):
        self.api_key = Config.YANDEX_API_KEY
        self.folder_id = Config.YANDEX_FOLDER_ID
        self.api_url = Config.YANDEX_GPT_API_URL

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in a consistent format."""
        return datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    def _create_system_prompt(self, timestamp: str) -> str:
        """Create system prompt with proper JSON format instructions."""
        return f"""Ты — умный и полезный AI-агент. ВАЖНО: Всегда отвечай строго в следующем формате в виде строки, но чтоб его можно было распарсить как JSON:

{{
  "status": "success",
  "data": {{ 
    "text": "Основной текст ответа от модели",
    "metadata": {{
      "model": "yandexgpt",
      "timestamp": "{timestamp}",
      "tokens_used": количество использованных токенов
    }}
  }},
  "error": null
}}

Или в случае ошибки:

{{
  "status": "error",
  "data": null,
  "error": {{
    "code": "код ошибки",
    "message": "Описание ошибки",
    "details": {{
      "retry_after": 60
    }}
  }}
}}

ОБЯЗАТЕЛЬНО используй timestamp: "{timestamp}" в поле metadata.timestamp
ОБЯЗАТЕЛЬНО удали ``` из начала и конца ответа. Не используй блоки кода, не добавляй символы ``` или другое форматирование."""

    def _create_payload(self, system_prompt: str, user_text: str) -> Dict[str, Any]:
        """Create API request payload."""
        return {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": Config.TEMPERATURE,
                "maxTokens": str(Config.MAX_TOKENS)
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_text}
            ]
        }

    def _create_error_response(self, code: str, message: str,
                               timestamp: str, **details) -> str:
        """Create standardized error response."""
        error_data = {
            "status": "error",
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": {
                    "timestamp": timestamp,
                    **details
                }
            }
        }
        return json.dumps(error_data, ensure_ascii=False, indent=2)

    def _clean_markdown_json(self, text: str) -> str:
        """Remove markdown code block formatting from JSON response."""
        text = text.strip()

        if text.startswith("```"):
            lines = text.split('\n')
            # Remove first line with ```
            if len(lines) > 1:
                lines = lines[1:]
            # Remove last line with ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = '\n'.join(lines)

        return text.strip()

    def _fix_timestamp_in_response(self, response: str, correct_timestamp: str) -> str:
        """Ensure response contains correct timestamp."""
        try:
            parsed_json = json.loads(response)
            if ("data" in parsed_json and
                    "metadata" in parsed_json["data"]):
                parsed_json["data"]["metadata"]["timestamp"] = correct_timestamp
                return json.dumps(parsed_json, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            logger.warning("Could not parse AI response as JSON to fix timestamp")

        return response

    def _extract_response_content(self, data: Dict[str, Any], timestamp: str) -> str:
        """Extract content from API response."""
        try:
            alternatives = data["result"]["alternatives"]
            if not alternatives:
                raise KeyError("No alternatives in response")

            raw_response = alternatives[0]["message"]["text"]
            cleaned_response = self._clean_markdown_json(raw_response)
            return self._fix_timestamp_in_response(cleaned_response, timestamp)

        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response structure: {data}")
            return self._create_error_response(
                "api_structure_error",
                "Неожиданная структура ответа от Yandex API",
                timestamp,
                response_keys=list(data.keys()) if isinstance(data, dict) else "invalid_response"
            )

    def get_response(self, prompt_text: str) -> str:
        """Send request to Yandex GPT API and return response."""
        timestamp = self._get_current_timestamp()
        system_prompt = self._create_system_prompt(timestamp)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        payload = self._create_payload(system_prompt, prompt_text)

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()
            return self._extract_response_content(data, timestamp)

        except requests.exceptions.Timeout:
            logger.error("Timeout error calling Yandex GPT API")
            return self._create_error_response(
                "timeout_error",
                "Превышено время ожидания ответа от API",
                timestamp,
                retry_after=30
            )

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "unknown"
            logger.error(f"HTTP error calling Yandex GPT API: {e}")
            retry_after = 60 if status_code in [429, 503] else 30
            return self._create_error_response(
                "http_error",
                f"HTTP ошибка: {status_code}",
                timestamp,
                status_code=status_code,
                retry_after=retry_after
            )

        except requests.exceptions.ConnectionError:
            logger.error("Connection error calling Yandex GPT API")
            return self._create_error_response(
                "connection_error",
                "Ошибка подключения к API",
                timestamp,
                retry_after=60
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error calling Yandex GPT API: {e}")
            return self._create_error_response(
                "request_error",
                "Ошибка при выполнении запроса к API",
                timestamp,
                error_type=type(e).__name__,
                retry_after=30
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error from Yandex GPT API: {e}")
            return self._create_error_response(
                "json_decode_error",
                "Не удалось декодировать JSON ответ от API",
                timestamp,
                retry_after=30
            )

        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            return self._create_error_response(
                "unknown_error",
                "Произошла непредвиденная ошибка",
                timestamp,
                error_type=type(e).__name__,
                error_message=str(e),
                retry_after=60
            )


class TelegramBot:
    """Telegram bot handler."""

    def __init__(self):
        self.gpt_client = YandexGPTClient()
        self.application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup bot command and message handlers."""
        self.application.add_handler(CommandHandler("start", self._start_handler))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler)
        )

    async def _start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "Привет! Я AI-агент, подключенный к YandexGPT. Задайте мне любой вопрос."
        )

    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages."""
        user_message = update.message.text

        # Show thinking message
        thinking_message = await update.message.reply_text("Думаю...")

        # Get AI response
        ai_response = self.gpt_client.get_response(user_message)

        # Update message with response
        await context.bot.edit_message_text(
            text=ai_response,
            chat_id=update.effective_chat.id,
            message_id=thinking_message.message_id
        )

    def run(self):
        """Start the bot."""
        logger.info("Starting bot...")
        self.application.run_polling()


def main() -> None:
    """Main entry point."""
    bot = TelegramBot()
    bot.run()


if __name__ == "__main__":
    main()