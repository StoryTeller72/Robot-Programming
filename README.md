# Обучение с подкреплением в задачах с разряженной наградой 

Reacher» — это двухзвенная роботизированная рука. 
Цель — переместить endeffector близко к цели (красная точка), которая появляется в случайном месте.
![Alt Text](https://gymnasium.farama.org/_images/reacher.gif)
Цели проекта:
- Изучить алгоритм Deep Deterministic Policy Gradients (DDPG)
- Реализовать алгоритм DDPG на языке Python
- Реализовать алгоритм DDPG на языке Python
- Обучить агента в среде Gymnasium Reacher с разряженной наградой
- Сравнить полученные результаты

# Установка

```shell
git clone https://github.com/StoryTeller72/Robot-Programming.git
cd reacher
conda create --name reacher --file requirements.txt
```
# Структура проекта
- checpoints: содержит обученные модели 
- runs: tensorboard логи
- sucsess_rate: содержит данные о результатах обучения
- train: код для обучения агента в разных средах
- utils: вспомогательный код, в том числе модель DDPG

