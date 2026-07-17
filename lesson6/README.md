# Результат обучения

Модель обучалась на тексте "Моби Дика" Германа Мелвилла.

### 20 эпох:

Вы: When the entire ship’s company were
Бот: When the entire ship’s company were assembled upon the sailors of
the whole, he made in the
pokovator, and had been found at the ship. It was
others, and more significantly the contrary, had been once been the ship had been
carries, were repeatedly way distinguished degree of the Pequod to those
Cle to, but the loomed round in the Pequod’s business.

Вы: What do ye do
Бот: What do ye do that thought, ye, that way brand us, sir, thou must needs
he’t know it.”

Вы: But, they were all eagerness again
Бот: But, they were all eagerness again and standing in a high,
when, so vainly, seemed to pointing a rather papered way.

Вы: While the mate was
Бот: While the mate was nearly three of the sea, this boat was
resolved was at the boat, and the mate, the compasses was intent
cely carrying on that hand to raise the stern. Elijah had so that his
been nearly, he had a little space, in the bows at the boat; and
other the Pequod was noged; for, with a boat was basket upon the
boat, and soft time to the ship on. Allong all the mate had

### 30 эпох:

Вы: When the entire ship’s company were
Бот: When the entire ship’s company were at last came from outset; and as
the massive, the boat was no spare boats swung over by the wind, with one hand
somehow furiously, and the scuttle, and with the rail, by
with the steady, and the crew to them; but so the boats were
wrist; though it seemed, to their unconsciously, for their eyes, while
when lowered every instant, whatever they had turned out of them. But all sailorars

Вы: What do ye do
Бот: What do ye do ye, ye are just enough? You are not the
anchor with sharp knife of clamped of them.

Вы: But, they were all eagerness again
Бот: But, they were all eagerness again, so stranger at these very
dreamed, seemed to the agitated waters caused by the bows to
him at the mast-head like a flash of the chase; and, in
the Parsee’s stern. A line was heard, the mates and
the smiting the port-head of the sea, he cried out of his
peculiarity, till a certain strange ship was at that steakened fast
with a gate. But a moment

Вы: While the mate was
Бот: While the mate was swung upon the whale with both hands with his
burnt of the crew, he was now, his-by his blubber
andish, and a moment vibrating his ivory leg had been near the ship
was, and at the same time, and so at the time, and the
ship was made so excited by the custom, that the enormous cutting was
he was therefore, as he had best made great stress his history of his own
evolence and his

### 23 эпохи

Вы: When the entire ship’s company were
Бот: When the entire ship’s company were engaged with the black-oil at the deck
in cold deck, and the severest of the ship, was now to be of the
came very carefully, Ahab advanced, and Ahab stood upright, Starbuck was
“heard thither!”

Вы: What do ye do
Бот: What do ye do? Don? Yes, Flask, sir. Don’t sell? Why’t ye?
standing ye know a fool”—litning-room. Behold me? Ahab
me, I must be thee; but I won’t ye see, my Captain Ahab
chinkers. Thou art thyself included on deck. Oh!—that’s a
names!—boats off! here goes ye?—pitch—pull—

Вы: But, they were all eagerness again
Бот: But, they were all eagerness again, all three days and almost
danced by the same time to the amazement of the most valuable
dragseemed that they were the best ground with this delirious
from the threateningest secretane perils of the irregularly
Nantom of these two ships were on the marks of
the poor leviathan.

Вы: While the mate was
Бот: While the mate was few inches of the native coasts; and the two
rigged and the very marked, it was given to blow the
buls, the crew almost as he had been, and a knocking
the mate, as a vessel to his hotness.


| Критерий                                 | 20 эпох                                                                                                  | 23 эпохи                                                                                                                                                     | 30 эпох                                                                                                                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Связность текста                  | Низкая. Частые обрывы предложений и потеря смысла.                 | Высокая. Большинство предложений выглядит естественно.                                                            | Средняя. Предложения длиннее, но встречаются смысловые повторы.                                         |
| Грамматика                             | Много грамматических ошибок и неестественных конструкций. | В целом корректнее, встречаются отдельные ошибки.                                                                       | Грамматика лучше, чем на 20 эпохах, но присутствуют длинные повторяющиеся конструкции. |
| Лексика                                   | Есть случайные бессмысленные слова (pokovator, Cle).                          | Лексика соответствует стилю романа, появляются характерные выражения.                                | Богатая морская лексика, но заметны частые повторы слов (boat, ship).                                          |
| Соответствие стилю Moby-Dick | Низкое. Стиль угадывается лишь частично.                                   | Наиболее близко к оригиналу: присутствуют Ahab, Starbuck, Flask, характерные обращения и диалоги. | Стиль сохранен, но текст становится более шаблонным.                                                              |
| Повторения                             | Средние.                                                                                              | Небольшие.                                                                                                                                               | Выраженные повторы слов и конструкций.                                                                                       |
| Осмысленность ответов        | Низкая. Большинство ответов быстро теряют смысл.                    | Высокая. Ответы выглядят наиболее логичными и литературными.                                                  | Средняя. Есть удачные фрагменты, но наблюдается деградация связности из-за повторов.   |
| Общая оценка                          | 2/5                                                                                                          | 4/5                                                                                                                                                               | 3/5                                                                                                                                                           |

## Вывод
По результатам экспериментов можно сделать вывод, что за 20 эпох модель не успела изучить все паттерны романа, на 30 эпохах модель начинает переобучаться (учащаются повторы слов и отдельных фраз), оптимальный результат достигнут на 23 эпохе.

Для дальнейшего повышения качества генерации можно рассмотреть следующие направления:
- **Увеличение размера модели.** Повышение числа слоев (num_layers), размерности скрытого пространства (d_model) или размера полносвязного слоя (d_ff) позволит модели изучать более сложные языковые зависимости.
- **Обучение на большем объеме данных.** Использование нескольких произведений одного автора или более крупного корпуса англоязычной художественной литературы позволит модели лучше усвоить грамматику и лексику.
- **Подбор гиперпараметров.** Изменение скорости обучения, размера батча, вероятности dropout, длины последовательности и других параметров может положительно сказаться на качестве обучения.
- **Использование Learning Rate Scheduler.** Постепенное уменьшение скорости обучения позволяет модели стабильнее сходиться на последних эпохах.
- **Применение ранней остановки (Early Stopping).** Остановка обучения при отсутствии улучшения на валидационной выборке помогает избежать переобучения.
- **Использование предварительно обученной модели (Fine-tuning).** Дообучение существующего языкового трансформера (например, GPT-подобной архитектуры) на тексте *Moby-Dick* позволит получить значительно более качественную генерацию по сравнению с обучением модели с нуля.

Следует отметить, что заранее невозможно определить, какое именно изменение окажет наибольшее влияние на качество генерации. На практике улучшение языковых моделей достигается путем последовательного проведения экспериментов с архитектурой, гиперпараметрами и обучающими данными с последующей оценкой полученных результатов.