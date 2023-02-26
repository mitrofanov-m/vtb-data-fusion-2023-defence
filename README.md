<div align='center'><h1> 🛡 Data Fusion contest 2023: Defence 🛡 </h1></div>

Data Fusion contest 2023 - [онлайн-соревнование](https://ods.ai/tracks/data-fusion-2023-competitions) ВТБ на стыке Data Science и информационной безопасности.

В данном репозитории представлено открытое решение задачи в виде jupyter ноутбука и python библиотеки для задачи [`Защита`](https://ods.ai/competitions/data-fusion2023-defence).

## 🔎 Описание

> Machine learning models using transaction records as inputs are popular among financial institutions. The most efficient models use deep-learning architectures similar to those in the NLP community, posing a challenge due to their tremendous number of parameters and limited robustness. In particular, deep-learning models are vulnerable to adversarial attacks: a little change in the input harms the model's output.

> [[Arxiv, 15 Jun 2021]](https://arxiv.org/abs/2106.08361) Adversarial Attacks on Deep Models for Financial Transaction Records

В 2023 году Data Fusion contest - это турнир по Adversarial ML между командами атакующих и защищающих ML модели на транзакционных данных.
 
##  🎯 Задача
В задаче `Защита` участники будут создавать устойчивые к атакам модели дефолта клиента, обученные на данных транзакций.

## 💡 Решение
Данное решение опирается на baseline организаторов. Основной задачей является снижение порога входа для начинающих участников соревнования. Для этого дополнительно реализовано:
- **Каркас проекта для соревнования.** Тут постарался спроецировать личный опыт участия в соревнованиях, при такой структуре проекта работать было наиболее продуктивно.
- **Инструкция по установке необходимых библиотек.** Собрал минимально-необходимые [зависимости](research-requirements.txt) и написал инструкцию для максимально легкого и безболезненного входа в docker-соревнование (см. ниже). Также для информации добавил [requirements.txt](requirements.txt) образа организаторов.
- **Механизм локальной валидации.** Разработанные методы позволяют сравнивать свое решение как с атакой организаторов, так и с кастомными решениями [(тык)](notebooks/1.Local_validation.ipynb).
- **Минорное улучшение бейзлайна.** Изменил взвешивание результата с `mean` на `median`. Увеличил число подсемплов при оценке до 30. На public-лидерборде 0,709115 [(метод custom_predict)](fusionlib/predicts.py).


### Установка

1. Для того, чтобы скачать репозиторий, Скачайте архив репозитория или введите следующую команду:
```bash
$ git clone https://github.com/mitrofanov-m/vtb-data-fusion-2023-defence.git
```
2. Хорошим тоном считается использование отдельного окружения для работы над проектом. Организаторы предлагают готовый Docker-образ, но для локальной разработки может хватить и изолированного окружения, дублирующего зависимости:
```bash
$ python --version
Python 3.10.*
$ python -m venv fusion-env
$ source fusion-env/bin/activate
# окружение должно отобразиться в консоли, как в пункте ниже
```
3. Теперь можно установить необходимые библиотеки. Версии библиотек, необходимых для локального запуска сравнивал с requirements.txt, опубликованным организаторами в чате соревнования. Для установки выполните следующие команды:
```bash
(fusion-env) $ pip install --upgrade pip setuptools
(fusion-env) $ pip install -r research-requirements.txt
```
4. Для работы с jupyter ноутбуками также создадим ядро с текущим окружением:
```bash
(fusion-env) $ python -m ipykernel install --user --name=fusion-env
```
Теперь окружение должно отобразиться в вашей jupyter среде.
5. Также необходимо скачать данные с [сайта соревнования](https://ods.ai/competitions/data-fusion2023-defence/dataset).

### Пример
Процесс работы над улучшением этого решения можно представить следующим образом:
1. Улучшаем метод reliable_predict

## 🖊 Контакты
По всем вопросам:
- @m1trm - telegram