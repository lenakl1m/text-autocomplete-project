# text-autocomplete-project
Нейросеть для автодополнения текстов (LSTM и distilgpt2)

# Проект: Нейросеть для автодополнения текстов

## Описание
Реализация модели автодополнения текста на основе LSTM и сравнение с предобученной моделью `distilgpt2`.

## Структура проекта

```
text-autocomplete-project/
├── data/                            # Датасеты
│   ├── raw_dataset.csv              # "сырой" скачанный датасет
│   └── dataset_processed.csv        # "очищенный" датасет
│   ├── train.csv                    # тренировочная выборка
│   ├── val.csv                      # валидационная выборка
│   └── test.csv                     # тестовая выборка
│
├── src/                             # Весь код проекта
│   ├── data_utils.py                # Обработка датасета
|   ├── next_token_dataset.py        # код с torch Dataset'ом 
│   ├── lstm_model.py                # код lstm модели
|   ├── eval_lstm.py                 # замер метрик lstm модели
|   ├── lstm_train.py                # код обучения модели
|   ├── eval_transformer_pipeline.py # код с запуском и замером качества трансформера
│
├── configs/                         # yaml-конфиги с настройками проекта
│
├── models/                          # веса обученных моделей
|
├── solution.ipynb                   # ноутбук с решением
└── requirements.txt                 # зависимости проекта
```

## Как запустить
1. Установить зависимости: `pip install -r requirements.txt`
2. Запустить `jupyter notebook solution.ipynb`

