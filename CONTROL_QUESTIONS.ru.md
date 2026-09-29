# Контрольные вопросы

## Q1
Question: Какие архитектурные слои использует Now in Android?

Expectation: В ответе должны быть упомянуты слой данных, доменный слой и UI-слой, а также реактивная или однонаправленная модель потока данных.

Sources:
- docs/ArchitectureLearningJourney.md

Checks:
- data layer | слой данных
- domain layer | доменный слой
- ui layer | ui-слой | слой интерфейса
- reactive | unidirectional | реактивн | однонаправлен
- data flows up | events flow down | данные поднимаются вверх | события идут вниз

## Q2
Question: Как данные попадают на экран For You при первом запуске приложения?

Expectation: В ответе должны быть описаны стартовая синхронизация через WorkManager, синхронизация репозиториев, обновления локального Room, а также роль GetUserNewsResourcesUseCase и ForYouViewModel в формировании UI state.

Sources:
- docs/ArchitectureLearningJourney.md

Checks:
- workmanager
- foryouviewmodel
- getusernewsresourcesusecase
- room | newsresourcedao
- newsfeeduistate | success | ui state

## Q3
Question: Почему data layer описывается как offline-first?

Expectation: В ответе должно быть сказано, что локальное хранилище является source of truth, чтение идет из локальных потоков данных, а удаленные данные согласуются с локальным хранилищем.

Sources:
- docs/ArchitectureLearningJourney.md
- core/data/src/main/kotlin/com/google/samples/apps/nowinandroid/core/data/repository/OfflineFirstNewsRepository.kt

Checks:
- source of truth | источник истины
- local storage | room | локальное хранилище
- reads | data streams | flow | чтение | поток данных
- remote | reconcile | sync | удаленн | синхрон

## Q4
Question: Как в проекте реализована синхронизация данных?

Expectation: В ответе должны быть упомянуты WorkManager, SyncWorker, синхронизация репозиториев, retry или exponential backoff, а также обновление локальных данных после получения удаленных.

Sources:
- docs/ArchitectureLearningJourney.md
- sync/work/src/main/kotlin/com/google/samples/apps/nowinandroid/sync/workers/SyncWorker.kt
- core/data/src/main/kotlin/com/google/samples/apps/nowinandroid/core/data/SyncUtilities.kt

Checks:
- workmanager
- syncworker
- repository | repositories | репозитор
- retry | backoff | exponential backoff | повтор | экспоненциальн
- local storage | room | database | локальн | база данных

## Q5
Question: Какова роль domain layer в Now in Android?

Expectation: В ответе должно быть сказано, что domain layer содержит use cases с operator invoke, которые упрощают или убирают дублирование логики во ViewModel за счет комбинирования или преобразования данных репозиториев.

Sources:
- docs/ArchitectureLearningJourney.md

Checks:
- use case | use cases | сценар | вариант использования
- operator fun invoke | single invocable method | invoke
- viewmodel | viewmodels
- combine | transform | комбинир | преобраз

## Q6
Question: Как ViewModel преобразуют данные репозиториев в UI state?

Expectation: В ответе должны быть упомянуты flows, combine или map, преобразование в StateFlow через stateIn, а также sealed immutable UI state.

Sources:
- docs/ArchitectureLearningJourney.md

Checks:
- flow | flows | поток
- combine | map | комбинир | отображ
- statein | state flow | stateflow
- sealed | immutable | неизменяем
- ui state | состояние интерфейса

## Q7
Question: Какие build variants и product flavors используются в Now in Android?

Expectation: В ответе должны быть упомянуты debug, release и benchmark варианты, flavors demo и prod, demoDebug для обычной разработки, локальные данные в demo и backend-вызовы в prod.

Sources:
- README.md

Checks:
- debug
- release
- benchmark
- demo
- prod
- demodebug
- local data | static local data | локальные данные
- backend | network calls | real network | сервер | сетевые вызовы

## Q8
Question: Как в документации описаны screenshot tests?

Expectation: В ответе должны быть упомянуты Roborazzi, known correct screenshots, а также gradle-задачи verify, record и compare.

Sources:
- README.md

Checks:
- screenshot | скриншот
- roborazzi
- known correct | эталонн
- verifyroborazzidemodebug
- recordroborazzidemodebug
- compareroborazzidemodebug

## Q9
Question: Как устроена модульная структура проекта?

Expectation: В ответе должны быть описаны модули app, feature, core и miscellaneous, а также правило, что feature разделены на api и impl и имеют ограничения по зависимостям.

Sources:
- docs/ModularizationLearningJourney.md

Checks:
- app
- feature
- core
- miscellaneous | sync | benchmark | test modules | вспомогательн
- api
- impl
- should not depend on feature | should not depend on app | не должны зависеть от feature | не должны зависеть от app

## Q10
Question: Что именно делает SyncWorker во время синхронизации?

Expectation: В ответе должны быть упомянуты старт аналитики, подписка на sync status, параллельная синхронизация topic и news repositories, заполнение FTS при успехе, а также возврат success или retry.

Sources:
- sync/work/src/main/kotlin/com/google/samples/apps/nowinandroid/sync/workers/SyncWorker.kt

Checks:
- analytics | logsyncstarted | sync started | аналитик
- subscribe | syncsubscriber | подписк
- parallel | awaitall | параллельн
- topicrepository | topics | topic
- newsrepository | news
- populateftsdata | fts
- result.success | success | успех
- result.retry | retry | повтор
