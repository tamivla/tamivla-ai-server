# Паспорт проекта

**Проект:** Tamivla AI Server  
**Бренд:** Tamivla Industrial Group

**Дата создания документа:** 2024.09.15  
**Дата обновления документа:** 2025.10.15 10:40

## Обзор проекта

Tamivla AI Server - это OpenAI-совместимый сервер для управления AI-моделями, работающий под Windows. Сервер предоставляет стандартные OpenAI эндпоинты плюс кастомные функции для управления моделями и кешем.

Ключевые особенности:

✅ OpenAI-совместимый API

✅ Работает под Windows (в отличие от vLLM)

✅ Управление жизненным циклом моделей

✅ Сканирование локального кеша моделей

✅ Без автозагрузки из интернета (только локальные модели)

✅ Поддержка стандарта Hugging Face (формат models--author--name)

## Архитектура

```
API Request → FastAPI Route → Model Manager → Real AI Model → Response 
```

## Файловая структура

Основные файлы:

```text
src/
├── main.py                   # Точка входа FastAPI приложения
├── path_fix.py               # Фикс путей для Windows
│
├── api/
│   └── routes/
│       ├── embeddings.py     # OpenAI embeddings эндпоинты
│       ├── chat.py           # OpenAI chat эндпоинты
│       └── models.py         # Кастомные эндпоинты управления
│
└── services/
├── model_manager.py          # Менеджер жизненного цикла моделей
├── model_discovery.py        # Сканирование и анализ кеша
├── embedding_service.py      # Сервис эмбеддингов
└── llm_service.py            # Сервис языковых моделей
Вспомогательные файлы:
text
storage/
├── models/                   # Кеш моделей (автосоздается)
└── logs/
└── server.log                # Логи сервера
requirements.txt              # Зависимости Python
```

## Структура эндпоинтов

### OpenAI-совместимые:

```bash
GET /v1/models - список моделей
POST /v1/embeddings - эмбеддинги
POST /v1/chat/completions - чат
```

### Кастомные - управление моделями:

```bash
POST /v1/models/load - загрузка в память
POST /v1/models/unload - выгрузка из памяти
GET /v1/models/loaded - загруженные модели
POST /v1/models/download - скачивание моделей
```

### Кастомные - управление кешем:

```bash
GET /v1/cache/info - детальная информация о кеше
POST /v1/cache/analyze - анализ кеша на битые модели
DELETE /v1/cache/{model} - удаление моделей из кеша
```

### Системные:

```bash
GET / - информация о сервере
GET /health - здоровье сервера
GET /debug/routes - диагностика routes
```

## 1. OpenAI-совместимые эндпоинты

#### GET /v1/models - Получение списка доступных моделей

```bash
curl http://localhost:8000/v1/models
```

Ответ:

```json
{
	"object": "list",
	"data": [
		{
			"id": "multilingual-e5-large-instruct",
			"object": "model",
			"created": 1700000000,
			"owned_by": "tamivla"
		}
	]
}
```

#### POST /v1/embeddings - Получение векторных представлений текста

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["hello world", "привет мир"],
    "model": "multilingual-e5-large-instruct"
  }'
```

Ответ:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.023, -0.045, ...],
      "index": 0
    }
  ],
  "model": "multilingual-e5-large-instruct",
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 2
  }
}
```

#### POST /v1/chat/completions - Чат-комплишн (требует доработки для LLM моделей)

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct-GPTQ-Int4",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

## 2. Кастомные эндпоинты

### 2.1. Управление моделями

#### POST /v1/models/load - Загрузка модели в память

```bash
curl -X POST "http://localhost:8000/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "multilingual-e5-large-instruct",
    "model_type": "embedding"
  }'
```

Ответ:

```json
{
  "status": "success",
  "message": "Model multilingual-e5-large-instruct loaded",
  "model_name": "multilingual-e5-large-instruct"
}
```

#### POST /v1/models/unload - Выгрузка модели из памяти

```bash
curl -X POST "http://localhost:8000/v1/models/unload" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "multilingual-e5-large-instruct"}'
```

#### GET /v1/models/loaded - Список загруженных моделей

```bash
curl http://localhost:8000/v1/models/loaded
```

Ответ:

```json
{
  "loaded_models": {
    "multilingual-e5-large-instruct": "embedding"
  },
  "stats": {
    "total_loaded": 1,
    "embedding_models": 1,
    "llm_models": 0,
    "models": {
      "multilingual-e5-large-instruct": "embedding"
    }
  }
}
```

#### POST /v1/models/download - Скачивание модели из HuggingFace Hub

```bash
curl -X POST "http://localhost:8000/v1/models/download" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sentence-transformers/all-MiniLM-L6-v2"}'
```

### 2.2. Управление кешем

#### GET /v1/cache/info - Детальная информация о кеше

```bash
curl http://localhost:8000/v1/cache/info
```

Ответ (сокращенно):

```json
{
  "cache_path": "storage\\models",
  "total_models": 3,
  "models": [
    {
      "name": "multilingual-e5-large-instruct",
      "display_name": "multilingual-e5-large-instruct",
      "path": "storage\\models\\multilingual-e5-large-instruct",
      "size_mb": 3242.81,
      "type": "embedding",
      "files": [...],
      "is_usable": true,
      "architecture": "XLMRobertaModel",
      "model_type": "xlm-roberta",
      "vocab_size": 250002,
      "hidden_size": 1024
    }
  ]
}
```

#### POST /v1/cache/analyze - Анализ кеша на наличие битых моделей

```bash
curl -X POST http://localhost:8000/v1/cache/analyze
```

Ответ:

```json
{
  "total_models": 3,
  "broken_models": [],
  "usable_models": [
    "Qwen2.5-7B-Instruct-GPTQ-Int4",
    "models--Qwen--Qwen2.5-7B-Instruct", 
    "multilingual-e5-large-instruct"
  ]
}
```

#### DELETE /v1/cache/{model_name} - Удаление модели из кеша

```bash
curl -X DELETE "http://localhost:8000/v1/cache/all-mpnet-base-v2"
```

Ответ:

```json
{
  "status": "success",
  "message": "Модель all-mpnet-base-v2 удалена из кеша",
  "model_name": "all-mpnet-base-v2"
}
```

## 3. Системные эндпоинты

#### GET / - Информация о сервере

```bash
curl http://localhost:8000/
```

#### GET /health - Проверка здоровья сервера

```bash
curl http://localhost:8000/health
```

#### GET /debug/routes - Диагностика зарегистрированных routes

```bash
curl http://localhost:8000/debug/routes
```

## Практическое использование

### Старт сервера:

```bash
python src/main.py
```

### Базовый workflow:

#### Проверить доступные модели:

```bash
curl http://localhost:8000/v1/models
```

#### Загрузить модель:

```bash
curl -X POST "http://localhost:8000/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "multilingual-e5-large-instruct", "model_type": "embedding"}'
```

#### Использовать модель:

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": ["your text"], "model": "multilingual-e5-large-instruct"}'
```

### Управлять кешем:

```bash
# Посмотреть детальную информацию
curl http://localhost:8000/v1/cache/info

# Проанализировать кеш
curl -X POST http://localhost:8000/v1/cache/analyze

# Удалить ненужную модель
curl -X DELETE "http://localhost:8000/v1/cache/unwanted-model"
```

### Расширенное использование:

#### Скачивание новых моделей: 

```bash
# Скачать модель из HuggingFace
curl -X POST "http://localhost:8000/v1/models/download" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sentence-transformers/all-MiniLM-L6-v2"}'

# Проверить что модель появилась
curl http://localhost:8000/v1/models
```

#### Работа с разными форматами имен:

```bash
# Простое имя
curl -X POST "http://localhost:8000/v1/models/load" \
  -d '{"model_name": "multilingual-e5-large-instruct", "model_type": "embedding"}'

# Стандартное имя HuggingFace
curl -X POST "http://localhost:8000/v1/models/load" \
  -d '{"model_name": "models--sentence-transformers--all-MiniLM-L6-v2", "model_type": "embedding"}'
```

## Разработка и расширение

### Добавление новой модели:

#### Добавить специальный случай в model\_[manager.py]():

```bash
special_cases = {
    'multilingual-e5-large-instruct': 'intfloat/multilingual-e5-large-instruct',
    'new-model-name': 'author/new-model-name',
}
```

#### Проверить загрузку:

```bash
curl -X POST "http://localhost:8000/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "new-model-name", "model_type": "embedding"}'
```

### Мониторинг и логи: 

Логи: storage/logs/server.log

Метрики моделей: GET /v1/models/loaded

Метрики кеша: GET /v1/cache/info

## Рекомендации по развитию

### Приоритетные задачи:

- Доработать LLM сервис - полноценная работа chat/completions
- Реализовать эндпоинт скачивания моделей
- Добавить батчинг для эмбеддингов
- Реализовать квоты и лимиты
- Добавить аутентификацию

### Архитектурные улучшения:

- Кеширование эмбеддингов
- Фоновая загрузка моделей
- Health checks для моделей
- Метрики производительности

## 📊 Текущее состояние

✅ Реализовано:

- OpenAI-совместимые эндпоинты
- Управление жизненным циклом моделей
- Сканирование и анализ кеша
- Поддержка стандарта Hugging Face
- Работа под Windows

## Рабочие модели:

- Embedding: multilingual-e5-large-instruct (1024 размерность)
- LLM: Qwen2.5-7B-Instruct-GPTQ-Int4 (квантованная)
- LLM: models--Qwen--Qwen2.5-7B-Instruct (полная)

---

Tamivla AI Server готов к использованию в продакшене!

Документация актуальна на 24.10.2025
