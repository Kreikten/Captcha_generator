# neural_test
Данный проект реализует генерацию капчи, состоящей из 6 латинских букв и цифр с помощью модели нейронной сети
Пример результата:  
![image](https://user-images.githubusercontent.com/30326049/204719848-0a66acda-4c8b-42dc-b4d6-d244e297ee4d.png)

Для запуска докер контейнера использовать команду
```
sudo docker run -d --name generator -p 80:80 kreikten/captcha_generator
```

Для запуска необходимо ввести команде, находясь в корневом каталоге репозитория:
```
uvicorn Captcha_generator.app:application --host 0.0.0.0 --port 80
```
Модель нейронной сети
===========================

Описание модели
------------------------

Одним из наиболее эффективных генеративных алгоритмов для генерации изображений являются генеративные состязательные сети (или GAN).
![image](https://user-images.githubusercontent.com/30326049/204362212-f81b68fb-669c-401b-a019-e0c1862ca407.png)

В обычной структуре GAN есть две нейронных сети, конкурирующие друг с другом:  
*Генератор  
*Дискриминатор  
Они могут быть спроектированы с использованием различных сетей (например, сверточных нейронных сетей (CNN), рекуррентных нейронных сетей (RNN) или 
просто обычных нейронных сетей (ANN или RegularNets)). Поскольку мы будем генерировать изображения, CNN лучше подходят для этой задачи. 
Поэтому мы будем строить генератор и дискриминатор на основе сверточной нейронной сети

В двух словах, мы попросим генератор генерировать рукописные цифры, не предоставляя ему никаких дополнительных данных.
Одновременно мы отправим существующие рукописные цифры на дискриминатор и попросим его решить, являются ли изображения, генерируемые Генератором, подлинными или нет.
Сначала Генератор будет генерировать плохие изображения, которые сразу же будут помечены как поддельные Дискриминатором. Получив достаточную обратную связь от Дискриминатора, Генератор научится обманывать Дискриминатор. 
Следовательно, мы получим очень хорошую генеративную модель, которая может дать нам очень реалистичные результаты.


Обучение
------------------------

Используемая библиотека - TensorFlow, TensorFLow.Keras

В папке training_checkpoints находятся результаты обучения для каждой буквы и символа.
Для обучения использовался датасет EMNIST ByMerge, содержащий 47 несбалансированных классов

Результаты:  

0,1,2,3,4,5,6,7,8,9 - 100 эпох, датасет ~31000 изображений на символ  

b f g j k q r u - 1300 эпох, датасет ~4000 изображений на символ  

v w x y - 800  эпох, датасет ~4000 изображений на символ  

Остальные буквы - 200 эпох, датасет ~4000 изображений на символ  


FastAPI приложение
===========================

Для взаимодействия с моделью было разработано FastAPI приложение
Чтобы получить ссылку на изображение капчи и код, необходимо послать запрос GET /generate

Пример с помощью библиотеки requests:

```
import requests
req = requests.get("http://127.0.0.1/generate")
```
После чего в случае успеха(status_code == 200) будет возвращен словарь вида:
```
{"code": code, "fileURL":fileURL}
```


Библиотеки
=============================

Генерация капчи - *generate.py* 
---------------------------------

- библиотека, содержащая в себе функции, необходимые для генерации изображения:  

generate_captcha - создаёт труднораспознаваемую капчу из 6 символов, подгружая модели нейронной сети нужного символа и генерируя картинку  
add_noise - добавляет шум по гауссу для затруднения распознания капчи с помощью компьютерного зрения  
save_pic - сохраняет сгенерированное изображение  
concat_pics - объединяем сгенерированные картинки в одну большую  


Алгоритм работы функции generate_captcha:  
1. Повторить 6 раз:  
1.1 Подгрузить модель нейронной сети для i символа  
1.2 Сгенерировать изображение  
2. Объединить изображения в одно большое  
3. Добавить шум на изображение для затруднения распознавания компьютерным зрением  
4. Сохранить изображения  

Пример использования функции generate_captcha:  
```
filename = generate_captcha("123456")
```
Данный код создаст изображение с капчей 123456  

Модель нейронной сети - *neural_network.py*
---------------------------------

В данной библиотеке содержатся функции, необходимые предыдущей для подгрузки модели нейронной сети и генерации картинки(требуется создать пустую модель и в неё
через Checkpoint загрузить информацию из файлов)  

make_generator_model - возвращает пустой генератор  
make_discriminator_model - возвращает пустой дискриминатор  
generator_loss- возвращает функцию потерь генератора  
discriminator_loss - возвращает функцию потерь дискриминатора  


FastAPI приложение - *app.py*
---------------------------------

Главная программа - запускает сервер на localhost, позволяя посылать запрос на генерацию капчи  

Реализованные запросы:  

GET /{file_id} - позволяет скачать файл по указанному id  
GET /generate - генерирует капчу и возвращает словарь с кодом и ссылкой на изображение  
POST /load - загружает изображение на сервер и регистрирует его в программе  


