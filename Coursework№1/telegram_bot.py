import os
import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import numpy as np
import io
from traditional_model import TraditionalClassifier
from simple_cnn import SimpleCNNClassifier
from resnet_model import ResNetClassifier
from config import Config

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class EuroSATTelegramBot:
    """Telegram бот для классификации изображений EuroSAT"""

    def __init__(self, token):
        self.token = token
        self.config = Config()

        # Инициализация моделей
        print("🔄 Инициализация моделей...")
        self.traditional = TraditionalClassifier()
        self.simple_cnn = SimpleCNNClassifier()
        self.resnet = ResNetClassifier()

        # Загрузка моделей
        self._load_models()

        # Статистика
        self.stats = {
            'total_images': 0,
            'predictions': {}
        }

    def _load_models(self):
        """Загрузка всех моделей"""
        try:
            self.traditional.load_model()
            print("✅ Traditional model loaded")
        except Exception as e:
            print(f"❌ Traditional model error: {e}")
            self.traditional = None

        try:
            self.simple_cnn.load_model()
            print("✅ Simple CNN model loaded")
        except Exception as e:
            print(f"❌ Simple CNN model error: {e}")
            self.simple_cnn = None

        try:
            self.resnet.load_model()
            print("✅ ResNet model loaded")
        except Exception as e:
            print(f"❌ ResNet model error: {e}")
            self.resnet = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        welcome_text = f"""
👋 Привет, {user.first_name}!

🌍 Я бот для классификации спутниковых снимков земной поверхности.

📸 Отправьте мне изображение, и я определю:
• Растительность (леса, поля, пастбища)
• Водоемы (реки, озера, моря)
• Городские территории (жилые, промышленные зоны)
• Инфраструктуру (дороги, магистрали)

📊 Я использую 3 разных модели для анализа:
1. Traditional (SIFT + SVM)
2. Simple CNN
3. ResNet (самая точная)

🎯 Отправьте мне любое спутниковое изображение!
        """
        await update.message.reply_text(welcome_text)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
🆘 *Помощь*

*Команды:*
/start - Начать работу с ботом
/help - Показать это сообщение
/stats - Показать статистику
/classes - Показать все классы
/best - Показать лучшую модель

*Как использовать:*
1. Просто отправьте мне изображение (спутниковый снимок)
2. Я проанализирую его тремя разными моделями
3. Вы увидите результаты и уверенность каждой модели

*Поддерживаемые форматы:* JPG, PNG, JPEG

*Примеры классов:*
🌲 Forest - Лес
🌾 AnnualCrop - Сельскохозяйственные культуры
🏭 Industrial - Промышленная зона
🏠 Residential - Жилая зона
🚗 Highway - Магистраль
🌊 SeaLake - Море/Озеро
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        stats_text = f"""
📊 *Статистика бота*

📸 Всего обработано изображений: {self.stats['total_images']}

🏆 Лучшие результаты по классам:
"""

        # Собираем статистику по классам
        for class_name in self.config.CLASSES:
            count = self.stats['predictions'].get(class_name, 0)
            if count > 0:
                stats_text += f"  • {class_name}: {count} раз\n"

        await update.message.reply_text(stats_text, parse_mode='Markdown')

    async def classes_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /classes"""
        classes_text = "🎯 *Все классы для классификации:*\n\n"

        for i, class_name in enumerate(self.config.CLASSES, 1):
            emoji = self._get_emoji_for_class(class_name)
            classes_text += f"{i}. {emoji} {class_name}\n"

        classes_text += "\n📌 *Описание:*\n"
        classes_text += "• AnnualCrop - Сельхозкультуры (пшеница, кукуруза)\n"
        classes_text += "• Forest - Лес\n"
        classes_text += "• HerbaceousVegetation - Травянистая растительность\n"
        classes_text += "• Highway - Магистраль/Шоссе\n"
        classes_text += "• Industrial - Промышленная зона\n"
        classes_text += "• Pasture - Пастбище\n"
        classes_text += "• PermanentCrop - Многолетние насаждения (сады, виноградники)\n"
        classes_text += "• Residential - Жилая зона\n"
        classes_text += "• River - Река\n"
        classes_text += "• SeaLake - Море/Озеро"

        await update.message.reply_text(classes_text, parse_mode='Markdown')

    async def best_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /best"""
        best_text = """
🏆 *Лучшая модель*

📈 На основе тестирования:
1. 🥇 *ResNet* - Наиболее точная модель (95-98%)
   - Использует предобученные веса ImageNet
   - Глубокие сверточные сети
   - Лучше всего для сложных изображений

2. 🥈 *Simple CNN* - Хороший баланс (85-90%)
   - Простая архитектура
   - Быстрее чем ResNet
   - Подходит для большинства задач

3. 🥉 *Traditional (SIFT+SVM)* - Базовая модель (70-75%)
   - Использует ручные признаки
   - Быстрая, но менее точная
   - Хороша для простых случаев

💡 *Рекомендация:* Используйте ResNet для максимальной точности!
        """
        await update.message.reply_text(best_text, parse_mode='Markdown')

    def _get_emoji_for_class(self, class_name):
        """Получить эмодзи для класса"""
        emoji_map = {
            'AnnualCrop': '🌾',
            'Forest': '🌲',
            'HerbaceousVegetation': '🌿',
            'Highway': '🛣️',
            'Industrial': '🏭',
            'Pasture': '🐑',
            'PermanentCrop': '🍇',
            'Residential': '🏠',
            'River': '🌊',
            'SeaLake': '🌅'
        }
        return emoji_map.get(class_name, '📷')

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик фотографий"""
        try:
            # Отправляем сообщение о начале обработки
            message = await update.message.reply_text("⏳ Анализирую изображение...")

            # Получаем файл фото
            photo_file = await update.message.photo[-1].get_file()

            # Скачиваем фото в память
            photo_bytes = await photo_file.download_as_bytearray()

            # Конвертируем в PIL Image
            image = Image.open(io.BytesIO(photo_bytes)).convert('RGB')

            # Изменяем размер
            image = image.resize(self.config.IMG_SIZE)

            # Конвертируем в numpy array
            image_array = np.array(image)

            # Обновляем статистику
            self.stats['total_images'] += 1

            # Получаем предсказания от всех моделей
            results = {}

            # Traditional модель
            if self.traditional and hasattr(self.traditional, 'is_trained') and self.traditional.is_trained:
                try:
                    trad_pred, trad_probs = self.traditional.predict(image_array)
                    trad_confidence = float(np.max(trad_probs))
                    trad_class = self.config.CLASSES[trad_pred]

                    results['traditional'] = {
                        'class': trad_class,
                        'confidence': trad_confidence,
                        'emoji': self._get_emoji_for_class(trad_class)
                    }
                except Exception as e:
                    logger.error(f"Traditional prediction error: {e}")

            # Simple CNN модель
            if self.simple_cnn and hasattr(self.simple_cnn, 'is_trained') and self.simple_cnn.is_trained:
                try:
                    cnn_pred, cnn_probs = self.simple_cnn.predict(image_array)
                    cnn_confidence = float(np.max(cnn_probs))
                    cnn_class = self.config.CLASSES[cnn_pred]

                    results['simple_cnn'] = {
                        'class': cnn_class,
                        'confidence': cnn_confidence,
                        'emoji': self._get_emoji_for_class(cnn_class)
                    }
                except Exception as e:
                    logger.error(f"Simple CNN prediction error: {e}")

            # ResNet модель
            if self.resnet and hasattr(self.resnet, 'is_trained') and self.resnet.is_trained:
                try:
                    resnet_pred, resnet_probs = self.resnet.predict(image_array)
                    resnet_confidence = float(np.max(resnet_probs))
                    resnet_class = self.config.CLASSES[resnet_pred]

                    results['resnet'] = {
                        'class': resnet_class,
                        'confidence': resnet_confidence,
                        'emoji': self._get_emoji_for_class(resnet_class)
                    }

                    # Обновляем статистику по классам
                    if resnet_class not in self.stats['predictions']:
                        self.stats['predictions'][resnet_class] = 0
                    self.stats['predictions'][resnet_class] += 1
                except Exception as e:
                    logger.error(f"ResNet prediction error: {e}")

            # Формируем ответ
            response = self._format_response(results)

            # Отправляем результат
            await message.edit_text(response, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error processing photo: {e}")
            await update.message.reply_text(f"❌ Ошибка при обработке изображения: {str(e)}")

    def _format_response(self, results):
        """Форматирование ответа с результатами"""
        if not results:
            return "❌ Не удалось обработать изображение. Попробуйте другое."

        response = "🎯 *Результаты классификации:*\n\n"

        # Находим лучшую модель
        best_model = None
        best_confidence = 0

        for model_name, result in results.items():
            if result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_model = model_name

        # Формируем результаты по моделям
        model_names = {
            'traditional': 'Traditional (SIFT + SVM)',
            'simple_cnn': 'Simple CNN',
            'resnet': 'ResNet'
        }

        for model_key, model_name in model_names.items():
            if model_key in results:
                result = results[model_key]
                is_best = (model_key == best_model)
                medal = "🏆 " if is_best else "  "

                response += f"{medal}*{model_name}:*\n"
                response += f"  {result['emoji']} Класс: {result['class']}\n"
                response += f"  📊 Уверенность: {result['confidence'] * 100:.1f}%\n\n"

        # Добавляем рекомендацию
        if best_model == 'resnet':
            response += "💡 *Рекомендация:* Доверяйте результату ResNet - это самая точная модель!"
        elif best_model == 'simple_cnn':
            response += "💡 *Рекомендация:* Simple CNN показала хороший результат!"
        else:
            response += "💡 *Рекомендация:* Traditional модель может ошибаться, попробуйте другое изображение."

        return response

    async def run(self):
        """Запуск бота"""
        # Создаем приложение
        application = Application.builder().token(self.token).build()

        # Регистрируем обработчики
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("classes", self.classes_command))
        application.add_handler(CommandHandler("best", self.best_command))

        # Обработчик фотографий
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))

        # Запускаем бота
        print("🤖 Бот запущен! Нажмите Ctrl+C для остановки.")
        await application.initialize()
        await application.start()
        await application.updater.start_polling()

        # Бесконечный цикл
        await asyncio.Event().wait()

    def run_sync(self):
        """Синхронный запуск бота"""
        asyncio.run(self.run())


def get_bot_token():
    """Получить токен бота"""
    # Сначала пробуем получить из переменной окружения
    token = os.getenv('TELEGRAM_BOT_TOKEN')

    if not token:
        # Если нет в переменных окружения, запрашиваем у пользователя
        print("\n" + "=" * 50)
        print("🤖 НАСТРОЙКА TELEGRAM БОТА")
        print("=" * 50)
        print("\n1. Откройте Telegram и найдите @BotFather")
        print("2. Отправьте /newbot и следуйте инструкциям")
        print("3. Получите токен вашего бота")
        print("\n" + "=" * 50)

        token = input("\nВведите токен вашего Telegram бота: ").strip()

        # Предлагаем сохранить в .env файл
        save = input("\nСохранить токен в файл .env для будущего использования? (y/n): ")
        if save.lower() == 'y':
            with open('.env', 'w') as f:
                f.write(f"TELEGRAM_BOT_TOKEN={token}\n")
            print("✅ Токен сохранен в файл .env")

    return token


if __name__ == '__main__':
    token = get_bot_token()

    if not token:
        print("❌ Токен не получен. Бот не может быть запущен.")
        exit(1)

    bot = EuroSATTelegramBot(token)
    bot.run_sync()