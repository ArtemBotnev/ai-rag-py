# Control Questions

## Q1
Question: What architecture layers does Now in Android use?

Expectation: The answer should mention the data layer, domain layer and UI layer, and note the reactive or unidirectional data flow model.

Sources:
- docs/ArchitectureLearningJourney.md

Checks:
- data layer
- domain layer
- ui layer
- reactive | unidirectional
- data flows up | events flow down | data flows up and events flow down

## Q2
Question: How does data reach the For You screen on first app launch?

Expectation: The answer should describe startup sync through WorkManager, repository sync, local Room updates, GetUserNewsResourcesUseCase and ForYouViewModel producing UI state.

Sources:
- docs/ArchitectureLearningJourney.md

Checks:
- workmanager
- foryouviewmodel
- getusernewsresourcesusecase
- room | newsresourcedao
- newsfeeduistate | success

## Q3
Question: Why is the data layer described as offline-first?

Expectation: The answer should explain that local storage is the source of truth, reads happen from local data streams, and remote data is reconciled into local storage.

Sources:
- docs/ArchitectureLearningJourney.md
- core/data/src/main/kotlin/com/google/samples/apps/nowinandroid/core/data/repository/OfflineFirstNewsRepository.kt

Checks:
- source of truth
- local storage | room
- reads | data streams | flow
- remote | reconcile | sync

## Q4
Question: How is data synchronization implemented in the project?

Expectation: The answer should mention WorkManager, SyncWorker, repository synchronization, retry or exponential backoff, and updating local data after remote fetches.

Sources:
- docs/ArchitectureLearningJourney.md
- sync/work/src/main/kotlin/com/google/samples/apps/nowinandroid/sync/workers/SyncWorker.kt
- core/data/src/main/kotlin/com/google/samples/apps/nowinandroid/core/data/SyncUtilities.kt

Checks:
- workmanager
- syncworker
- repository | repositories
- retry | backoff | exponential backoff
- local storage | room | database

## Q5
Question: What is the purpose of the domain layer in Now in Android?

Expectation: The answer should describe use cases with operator invoke that simplify or deduplicate logic in ViewModels by combining or transforming repository data.

Sources:
- docs/ArchitectureLearningJourney.md

Checks:
- use case | use cases
- operator fun invoke | single invocable method
- viewmodel | viewmodels
- combine | transform

## Q6
Question: How do ViewModels turn repository data into UI state?

Expectation: The answer should mention flows, combine or map, conversion to StateFlow with stateIn, and sealed immutable UI state.

Sources:
- docs/ArchitectureLearningJourney.md

Checks:
- flow | flows
- combine | map
- statein | state flow | stateflow
- sealed | immutable
- ui state

## Q7
Question: Which build variants and product flavors are used in Now in Android?

Expectation: The answer should mention debug, release and benchmark variants, the demo and prod flavors, demoDebug for regular development, demo local data and prod backend calls.

Sources:
- README.md

Checks:
- debug
- release
- benchmark
- demo
- prod
- demodebug
- local data | static local data
- backend | network calls | real network

## Q8
Question: How are screenshot tests described in the documentation?

Expectation: The answer should mention Roborazzi, known correct screenshots and the verify, record and compare gradle tasks.

Sources:
- README.md

Checks:
- screenshot
- roborazzi
- known correct
- verifyroborazzidemodebug
- recordroborazzidemodebug
- compareroborazzidemodebug

## Q9
Question: How is the project modularized?

Expectation: The answer should describe app, feature, core and miscellaneous modules, and explain that features are split into api and impl with dependency rules around them.

Sources:
- docs/ModularizationLearningJourney.md

Checks:
- app
- feature
- core
- miscellaneous | sync | benchmark | test modules
- api
- impl
- should not depend on feature | shouldn't depend on feature | should not depend on app

## Q10
Question: What exactly does SyncWorker do during synchronization?

Expectation: The answer should mention analytics start, sync subscription, parallel sync of topic and news repositories, FTS population on success and success or retry result handling.

Sources:
- sync/work/src/main/kotlin/com/google/samples/apps/nowinandroid/sync/workers/SyncWorker.kt

Checks:
- analytics | logsyncstarted | sync started
- subscribe | syncsubscriber
- parallel | awaitall
- topicrepository | topics
- newsrepository | news
- populateftsdata | fts
- result.success | success
- result.retry | retry
