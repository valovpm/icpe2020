# Transferring Pareto Frontiers across Heterogeneous Hardware Environments

# Description
This repository contains code for a study on transferring Pareto frontiers of configurable software systems across heterogeneous hardware platforms, that was published in the International Conference on Performance Engineering 2020 ([ICPE 2020](https://icpe2020.spec.org/)) proceedings.
Below you will find the abstract of the [published paper](https://bit.ly/3oPyUxk).

# Abstract
Software systems provide user-relevant configuration options called features.
Features affect functional and non-functional system properties, whereas selections of features represent system configurations.
A subset of configuration space forms a Pareto frontier of optimal configurations in terms of multiple properties, from which a user can choose the best configuration for a particular scenario.
However, when a well-studied system is redeployed on a different hardware, information about property value and the Pareto frontier might not apply.
We investigate whether it is possible to transfer this information across heterogeneous hardware environments.

We propose a methodology for approximating and transferring Pareto frontiers of configurable systems across different hardware environments.
We approximate a Pareto frontier by training an individual predictor model for each system property, and by aggregating predictions of each property into an approximated frontier.
We transfer the approximated frontier across hardware by training a transfer model for each property, by applying it to a respective predictor, and by combining transferred properties into a frontier.

We evaluate our approach by modeling Pareto frontiers as binary classifiers that separate all system configurations into optimal and non-optimal ones. Thus we can assess quality of approximated and transferred frontiers using common statistical measures like sensitivity and specificity. We test our approach using five real-world software systems from the compression domain, while paying special attention to their performance. Evaluation results demonstrate that accuracy of approximated frontiers depends linearly on predictors' training sample sizes, whereas transferring introduces only minor additional error to a frontier even for small training sizes.



# Перенос Парето-фронтов в гетерогенных аппаратных средах

# Описание
Этот репозиторий содержит код исследования трансфера Парето-фронтов конфигурируемых программных систем между гетерогенными аппаратными платформами, которое было опубликовано в материалах International Conference on Performance Engineering 2020 ([ICPE 2020](https://icpe2020.spec.org/)).
Ниже вы найдете аннотацию [опубликованной статьи](https://bit.ly/3oPyUxk).

# Аннотация
Программные системы предоставляют значимые для пользователя конфигурационные параметры, называемые функциями (features).
Функции (features) влияют на функциональные и нефункциональные свойства системы, тогда как выбор функций представляет собой конфигурацию системы.
Подмножество конфигурационного пространства образует Парето-фронт оптимальных конфигураций с точки зрения заданного множества свойств, из которых пользователь может выбрать наилучшую конфигурацию для конкретного сценария.
Однако, когда хорошо изученная система развертывается на другом оборудовании, информация о её свойствах и Парето-фронте может оказаться неприменимой.
Мы исследуем возможность трансфера этой информации между разнородными аппаратными средами.

Мы предлагаем методологию аппроксимации и трансфера Парето-фронтов для конфигурируемых систем в различных аппаратных средах. 
Мы аппроксимируем Парето-фронт, обучая индивидуальную модель предиктора для каждого свойства системы и объединяя предсказания каждого свойства в аппроксимированный Парето-фронт.
Мы переносим аппроксимированный фронт между аппаратными средами, обучая модель трансфера для каждого свойства, применяя ее к соответствующему предиктору и объединяя полученные значения в Парето-фронт.

Мы оцениваем наш подход, моделируя Парето-фронты как бинарные классификаторы, разделяющие все конфигурации системы на оптимальные и неоптимальные.
Таким образом, мы можем оценить качество аппроксимированных Парето-фронтов и Парето-фронтов перенесенных в другую аппаратную среду, используя общие статистические меры, такие как чувствительность (sensitivity) и специфичность (specificity).
Мы тестируем наш подход, используя пять реальных программных систем из области сжатия данных, уделяя особое внимание их производительности.
Результаты оценки показывают, что точность аппроксимированных Парето-фронтов линейно зависит от размеров обучающей выборки предикторов, в то время как перенос вносит лишь незначительную дополнительную ошибку в Парето-фронт даже при небольших размерах обучения.
