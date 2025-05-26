# MeowWoof-AI

**MeowWoof-AI** — это веб-сервис на базе FastAPI, позволяющий распознавать, изображена ли на фотографии кошка или собака. В основе лежит обученная модель от [Teachable Machine](https://teachablemachine.withgoogle.com/) и библиотека TensorFlow.

## 🚀 Функциональность

* Принимает изображение через POST-запрос.
* Определяет, кошка или собака на фото.

## 🧠 Используемая модель

Модель была обучена с помощью **Teachable Machine (Image Model)** и экспортирована в формате TensorFlow (`.h5`).
Её можно заменить своей, если вы хотите переобучить под свои данные.

## 📦 Зависимости

Убедитесь, что используете **Python 3.10**.
Установите зависимости:

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
fastapi==0.95.2
uvicorn[standard]==0.22.0
tensorflow==2.12.0
pillow==9.5.0
numpy==1.23.5
python-multipart==0.0.6
```

## 🛠️ Запуск

```bash
uvicorn main:app --reload
```

После запуска FastAPI будет доступен по адресу:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

Интерактивная документация (Swagger UI):
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
