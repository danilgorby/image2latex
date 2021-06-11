# Image2LaTex

**Список участников:**

Горбенко Даниил, Евсикова Дарья, Праздничных Антон

**Краткое описание задачи**
![scheme](./imgs/scheme.jpg)

Цель данного проекта - обобщить знания о задаче преобразования изображения математической формулы в LaTex код. Это подразумевает выделение существующих архитектур моделей для данной задачи и их различных модификаций. Идея такая, что в разных статьях есть разные модификации, часто не зависящие от конкретной архитектуры, которые могут значительно влиять на перфоманс конкретной модели. Было бы интересно применить эти модификации к разным моделям, исследовать их влияние на результаты и, таким образом, определить лучшую комбинацию. Также мы хотим исследовать влияние различных параметров, связанных непосредственно с обучением (различные методы оптимизации, расписания learning rate, размеры батча, регуляризации, инициализации, разные модификации функции потерь). 

**Данные**

Набор данных IM2LATEX-100K содержит ~103 тыс. различных математических уравнений в LaTeX формате вместе с соответствующими изображениями формул. Формулы (количество символов в которых находится в диапазоне от 40 до 1024) были получены из LaTeX исходников различных статей. Файлы PDF с формулой были преобразованы в формат PNG. Итоговый набор данных содержит ~103k изображений с разрешением 1654×2339 и соответствующие формулы LaTeX. Набор данных разделен на обучающую выборку (~84 тыс. уравнений), валидационную выборку (~9 тыс. уравнений) и тестовую выборку (~10 тыс. уравнений).

## Промежуточные результаты

В ходе изучения литературы нам показались интересными следующие направления для экспериментов с архитектурой Image2Latex модели:

1. Вид CNN модели в encoder части:
    - архитектура сети из статьи [[Genthial, Sauvestre](http://cs231n.stanford.edu/reports/2017/pdfs/815.pdf)]

    ![net1](./imgs/harvard.jpg)
    
    - сеть из статьи [[Deng et al.](https://arxiv.org/pdf/1609.04938v1.pdf)]
    - DenseNet [[Wang et al.](https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050919X00034/1-s2.0-S1877050919302686/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAUaCXVzLWVhc3QtMSJHMEUCIQDlkyO6sZ6uJWozMXBMJvVVERvglN%2BKfHZ%2F7W6i5BMIzQIgFREQ0PDh%2FEljeyXWDsYlTRFytNim3VrnZ131I1s00cAqgwQIvv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDIaZOYY9Ffi%2BBH7SAirXAxiqW8XiYJ7zHYRsMIUCsNG4gwmd%2BdLWv%2FK24FCpSQuCvSd6G%2BmMeuhPaDu09dEiEoFdfGLCwxN%2FVAURRij%2BS7H8VfgGzD6vkXgEJXJPEtsPhaK6laXIzaHgpJSRSgP0iJEZvX1gpYK0rLRd7xqjN7g4JxEaYU2NiYYtiqLeZ95GERyaJJINlE%2F8f%2BF%2FYTQr1%2FwZz6tcaJh%2Bsx7HpjKFBFfPMLkDWMgFLLJwWk6by9TXRjTvvTBjO4gsUSyIAETyPu8pM7hWXkbX0HOnwz7aEV5uc0bFc8DMaUxwQrnDTGp6tKgCxQk3%2FyFduW4h044AFxj2kODAceyeal2y2lwNdONXFbz9d2U6f%2FMUcMFdxUZzM0pL2LlGjw7hbYqJzQiNGP57DWm7X4%2FXNNm1dD09LAQVIS8ojeF6%2BvKNnyMehkEXR0QpL%2FN2KsTLkB%2FSLPDHSvNUDorSWXEDxYyGd4NSAeRAG%2FpifqVjnMRFiY8V%2FoGx0edA8%2FbAUqqofimRlvhmaKkG2LR2pQmc%2BL02kQ34MEP7FMYqI35Z3%2BmkTfINrKvUa3n9sn0tEokIDvM%2BNUZjmBnX6Yfnfx1adga%2BJmJtnmqDvQyI505faJCkBoYqYfI3b77ANQ58xjDRsI2GBjqlAcfrrA65%2FMxGOeaRQrgD3gs%2BxdZWVv5gMSHbJjKDT4MXHgbFwTf9fk64xgr6tWolLQy16UpGOaI0eqzo0cqJYPI5tFgQz7ZqcKUlwYq1fAQa61cy4DzhdnQT%2BYvXmGyplR%2B40QJ5UAfj2%2F7iSKFpaD8PFxLOXfrMoA9jHrACSzgaggDnjmH34wsuTClxZ3v98Ca6DTh%2Fw8pBilsaKx89mwHxIOVdAQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210611T130718Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5GR4RU4V%2F20210611%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=4ced4119254503da9abe2467235b040fe8d08388ed62e49036a81da4b2034403&hash=528fb961803354177fdd295713429173ca5aa8a27f71a6f4ea080034984d34e6&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050919302686&tid=spdf-d6f28301-0cba-4280-9071-8c3cae19232f&sid=419612c63573184e018bd2d6e9ebb7b42741gxrqb&type=client)]

    ![net2](./imgs/densenet.jpg)

2. Positional encoding:
     - без него
     - с 2D positional encoding, как в [Wang, Liu]
     - добавление RNN Encoder, как в [Deng et al.]
     
3. Attention
     - без него
     - channel-wise  (модфикация из [Wang et al.])
     - spatial-wise (дефолтный)
     - оба
     
4. Инициализация скрытого состояния декодера (в случае отсутствия rnn в encoder):
     - нулевая
     - обучаемая

**Эксперименты**

Обучение 1 эпохи на GPU (Google Colab) занимает около 1 ч 30 мин на полном датасете. На данный момент мы обучили несколько моделей (две эпохи).

![loss](./imgs/train_loss.png)

<p float="left">
  <img src="./imgs/test_precision.png" alt="precision" width="300"/>
  <img src="./imgs/test_recall.png" alt="recall" width="300"/>
  <img src="./imgs/test_f1.png" alt="f1" width="300"/>
</p>

Графики:

1. harvard cnn + Adam + decoder initialization with 0

2. harvard cnn + Adam + learnable decoder initialization

3. stanford cnn + RAdam + positional encoding

4. harvard cnn + RAdam + positional encoding

**Метрики**

Пока что мы имплементировали самые базовые текстовые метрики для инференса: Precision, Recall, F1. Каждый предсказанный токен считается true positive -- если он есть в референсной формуле, false positive -- если его нет в референсной формуле, false negative -- если токен есть в референсной формуле, но нет в предсказанной, (остальные true negative). 


**Выводы**

Из графиков видно, что RAdam значительно превосходит Adam по сходимости функции потерь на обучающей выборке. Это может быть связано с тем, что на первых шагах обучения оценка адаптивного множителя в Adam’е имеет очень большую дисперсию, и метод может делать большие шаги в неоптимальных направлениях, см [Liu](https://arxiv.org/pdf/1908.03265.pdf) за подробностями. Кажется разумным в дальнейшем использовать RAdam в качестве дефолтного оптимизатора.

## TO DO

* Имплементировать стандартные для данной задачи метрики (текстовые: BLEU, Perplexity, Rouge, Levenshtein distance, картиночные: edit distance)

* Провести более последовательные эксперименты с существующими архитектурами - line search по пунктам 1-4 разделе “Промежуточные результаты” (меняем каждый из параметров при фиксированных остальных)

* Имплементировать собственные идеи по encoder-decoder архитектуре

* Модифицировать CNN-RNN архитектуру на основе результатов экспериментов с разными модификациями

* Реализовать Transformer-based encoder-decoder (идея такая, что сейчас есть cross attention между векторами представлений картинок и формул, а self attention нет)


## Links

* [What You Get Is What You See:A Visual Markup Decompiler](https://arxiv.org/pdf/1609.04938v1.pdf)

* [Image to Latex](http://cs231n.stanford.edu/reports/2017/pdfs/815.pdf)

* [Image To Latex with DenseNet Encoder and Joint Attention](https://www.sciencedirect.com/science/article/pii/S1877050919302686/pdf?md5=28f37e951b6f27a28b3ebb1879d5ed94&pid=1-s2.0-S1877050919302686-main.pdf)
