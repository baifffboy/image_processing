import os
import sys
from data_analysis import DataAnalyzer
from traditional_model import TraditionalClassifier
from simple_cnn import SimpleCNNClassifier
from resnet_model import ResNetClassifier
from model_comparison import ModelComparator
from config import Config

def main():
    """Основная программа"""
    print("🌍 EuroSAT Classifier")
    print("=" * 50)
    
    os.makedirs(Config.MODELS_PATH, exist_ok=True)
    
    while True:
        print("\nВыберите действие:")
        print("1. Анализ данных")
        print("2. Обучить все модели")
        print("3. Сравнить модели")
        print("4. Запуск тг-бота")
        print("5. Выход")
        
        choice = input("\nВведите номер (1-5): ").strip()
        
        if choice == "1":
            analyzer = DataAnalyzer()
            analyzer.analyze()
            
        elif choice == "2":
            print("\n=== ОБУЧЕНИЕ МОДЕЛЕЙ ===")
            
            print("\n--- Traditional Classifier ---")
            traditional = TraditionalClassifier()
            traditional.train()
            traditional.save_model()
            4
            print("\n--- Simple CNN ---")
            simple_cnn = SimpleCNNClassifier()
            simple_cnn.train()
            simple_cnn.save_model()
            
            print("\n--- ResNet ---")
            resnet = ResNetClassifier()
            resnet.train()
            resnet.save_model()
            
            print("\n✅ Все модели обучены и сохранены!")
            
        elif choice == "3":
            comparator = ModelComparator()
            results, best_model = comparator.compare_models()

        elif choice == "4":
            print("\n🤖 Запуск Telegram бота...")
            print("=" * 50)

            try:
                # Импортируем модуль бота
                from telegram_bot import get_bot_token, EuroSATTelegramBot

                # Получаем токен
                token = get_bot_token()

                if not token:
                    print("❌ Токен не получен. Бот не может быть запущен.")
                    continue

                print(f"✅ Токен получен ({len(token)} символов)")
                print("\n🤖 Инициализация моделей...")

                # Создаем и запускаем бота
                bot = EuroSATTelegramBot(token)

                print("\n" + "=" * 50)
                print("🚀 Бот запущен!")
                print("📱 Откройте Telegram и найдите своего бота")
                print("📸 Отправьте боту спутниковое изображение")
                print("=" * 50)
                print("\nНажмите Ctrl+C для остановки бота и возврата в меню\n")

                # Запускаем бота
                bot.run_sync()

            except ImportError:
                print("❌ Модуль telegram не установлен.")
                print("Установите его: pip install python-telegram-bot")
            except KeyboardInterrupt:
                print("\n👋 Бот остановлен. Возвращаемся в меню...")
            except Exception as e:
                print(f"❌ Ошибка при запуске бота: {e}")

        elif choice == "5":
            print("Выход из программы")
            break
            
        else:
            print("Неверный выбор")

if __name__ == "__main__":
    main()