from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import numpy as np
import io
import os
import logging

app = FastAPI()

# Завантаження HTML сторінки
@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("static/index.html", "r", encoding="utf-8") as file:
        return file.read()


# Попарні порівняння для однієї дівчини
def pairwise_comparison(ratings, data):
    n = len(ratings)
    comparison_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Порівняння оцінок квіток i та j
            if ratings[i] < ratings[j]:
                comparison_matrix[i, j] = 1  # i краще j
                comparison_matrix[j, i] = -1  # j гірше i
                logging.info(f'Порівняння: {data.iloc[i, 0]} краще {data.iloc[j, 0]}')
            elif ratings[i] > ratings[j]:
                comparison_matrix[i, j] = -1  # i гірше j
                comparison_matrix[j, i] = 1  # j краще i
                logging.info(f'Порівняння: {data.iloc[j, 0]} краще {data.iloc[i, 0]}')
            else:
                comparison_matrix[i, j] = 0  # i і j рівні
                comparison_matrix[j, i] = 0  # j і i рівні
                logging.info(f'Порівняння: {data.iloc[i, 0]} рівний {data.iloc[j, 0]}')

    return comparison_matrix


# Функція для обчислення ранжування на основі матриці попарних порівнянь
def rank_objects(matrix):
    n = matrix.shape[0]
    total_wins = np.sum(matrix, axis=1)
    return np.argsort(total_wins)[::-1]  # Сортуємо за спадаючим порядком


# Завантаження CSV файлу і додавання/видалення полів
@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...), new_flower: str = Form(None), remove_flower: str = Form(None)):
    # Читання CSV файлу
    contents = await file.read()
    flower_list = contents.decode('utf-8').split(',')  # Перетворюємо байти в список квіток

    # Створюємо DataFrame з індексами і квітками
    df = pd.DataFrame({'Індекс': range(len(flower_list)), 'Квітка': flower_list})
    df['Індекс'] = df['Індекс'].astype(float)  # Перетворення на int

    # Додавання нового об'єкта
    if new_flower:
        new_row = pd.DataFrame({'Індекс': [len(df)], 'Квітка': [new_flower]})
        df = pd.concat([df, new_row], ignore_index=True)

    # Видалення рядка
    if remove_flower and remove_flower in df['Квітка'].values:
        df = df[df['Квітка'] != remove_flower]

    # Виводимо або повертаємо результат
    print(df)

    # Рейтинги квіток однієї дівчини
    girl = 'Індекс'
    ratings = df[girl].values.tolist()

    # Генерація матриці попарних порівнянь
    pairwise_matrix = pairwise_comparison(ratings, df)

    ranking_indices = rank_objects(pairwise_matrix)

    # Виведення найкращого ранжування
    best_ranking = [df.iloc[i, 0] for i in ranking_indices]

    # Запис матриці попарних порівнянь у файл
    df_matrix = pd.DataFrame(pairwise_matrix, index=df['Квітка'].values, columns=df['Квітка'].values)
    file_path = f'{girl}_pairwise_comparison.csv'
    df_matrix.to_csv(file_path, index=True)

    # Повертаємо дані для відображення та можливість завантажити файл
    return {
        'flower_list': flower_list,
        "fields": list(df.columns),
        "best_ranking": best_ranking,
        "matrix": df.to_html(),  # Перетворюємо матрицю на HTML таблицю
        "comparison_matrix": df_matrix.to_html(),  # Перетворюємо матрицю на HTML таблицю
        "file_url": f"/download/{file_path}"  # Посилання на завантаження CSV файлу
    }


# Маршрут для завантаження файлу
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(os.getcwd(), filename)
    return FileResponse(path=file_path, media_type='text/csv', filename=filename)
