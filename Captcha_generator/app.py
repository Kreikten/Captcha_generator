import random
from fastapi import FastAPI, Request, Form

import requests
from fastapi.responses import FileResponse
import uuid
import os

import Captcha_generator.generate
application = FastAPI()

# словарь для хранения сгенерированных картинок и путей к ним в формате
# уникальный сгенерированный ключ: путь к картинке
files = []
flag_generate = False

@application.get('/generate')
def download_file(request: Request):
    global flag_generate
    ''' Обработчик get /generate запроса для генерации картинки.
    Output:
        {"code": None, "fileURL": None} если возникла ошибка,
        {"code": code, "fileURL": url} если всё прошло корректно
    '''
    flag_generate = True
    list_sym = ["0", "1", "2", "3", "4", "5","6", "7", "8","9",
                    "a", "b","c", "d", "e","f", "g", "h", "i", "j", "k",
                    "l", "m", "n","o" , "p", "q","r", "s", "t","u", "v",
                    "w","x", "y", "z"]

    code = ""
    #генерируем случайный код капчи
    #в котором есть от 2 до 4 цифр и соответственно от 4 до 2 букв
    num_count = random.randint(2,4)
    letter_count = 6-num_count
    for i in range(num_count):

        code += list_sym[random.randint(0, 9)]
    for i in range(letter_count):
        code += list_sym[random.randint(10, len(list_sym)-1)]
    #и перемешаем их в случайном порядке
    code_list = list(code)
    code_list = random.sample(code_list, len(code_list))
    code = ""
    for i in code_list:
        code += i
    #print(code)


    temp = Captcha_generator.generate.generate_captcha(code)

    filepath = "./"+temp

    #POSTим картинку на сервер
    req = requests.post("http://127.0.0.1/load", data = {"filepath":filepath})
    #если запощена успешно
    if req.status_code == 200:
        file_url = f'http://127.0.0.1'+req.json()["fileURL"]
        return{"code": code,

                "captcha": file_url}
    #иначе отправить пустоту
    else:
        return {"code": None, "captcha": None}

@application.get('/{file_id}')
def download_file(file_id):
   # global flag_generate
    ''' Обработчик get запроса для скачивания картинки.
    Input:
        fieldId : уникальный id файла.
    Output:
        None если файла не существует,
        Файл если файл существует.
    '''
  #  if not flag_generate:
    if file_id != "favicon.ico":
        filepath = "./"+files[files.index(file_id)]
        if filepath:
            filename = os.path.basename(filepath)
            headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
            return FileResponse(filepath, headers=headers, media_type='image/png')
        else:
            return None



@application.post('/load')
def convert(request: Request, filepath: str = Form(...)):
    ''' Обработчик POST /load запроса для загрузки картинки и регистрации
    Input:
        filepath : путь к картинке
    Output:
        {"fileURL": file_url} - ссылка на картинку на сервере

    '''
    #генерируем уникальный код для картинки
    file_id = str(uuid.uuid4())

    #сохраняем в словаре по сгенерированному коду
    files.append(filepath[2:])
    print(files)
    file_url = f'/{filepath[2:]}'
    return {"fileURL": file_url}
