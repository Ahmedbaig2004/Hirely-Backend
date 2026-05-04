import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { embedText, batchEmbedTexts } from "../config/gemini.js";
import { prisma } from "../config/db.js";
import * as cheerio from "cheerio";
import dotenv from "dotenv";
dotenv.config();

// ─────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────
const PAGE_BATCH_SIZE = 5; // pages fetched in parallel
const DELAY_MS = 1000; // ms between page batches
const EMBED_BATCH_SIZE = 50; // chunks per single batchEmbedTexts() call
const EMBED_TIMEOUT_MS = 30000; // 30s timeout per embed API call
const MAX_URLS_PER_CATEGORY = 30; // hard cap — prevents any sitemap explosion

// ─────────────────────────────────────────
// FALLBACK URL LIBRARY
// ─────────────────────────────────────────
const FALLBACKS = {
  javascript: [
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Closures",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Event_loop",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Destructuring_assignment",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/Arrow_functions",
    "https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model/Introduction",
    "https://javascript.info/async-await",
    "https://javascript.info/generators",
    "https://javascript.info/classes",
    "https://javascript.info/prototypes",
    "https://javascript.info/modules-intro",
    "https://javascript.info/error-handling",
    "https://javascript.info/map-set",
    "https://javascript.info/weakmap-weakset",
    "https://javascript.info/proxy",
    "https://javascript.info/regexp-introduction",
  ],

  typescript: [
    "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html",
    "https://www.typescriptlang.org/docs/handbook/2/narrowing.html",
    "https://www.typescriptlang.org/docs/handbook/2/functions.html",
    "https://www.typescriptlang.org/docs/handbook/2/objects.html",
    "https://www.typescriptlang.org/docs/handbook/2/generics.html",
    "https://www.typescriptlang.org/docs/handbook/2/keyof-types.html",
    "https://www.typescriptlang.org/docs/handbook/utility-types.html",
    "https://www.typescriptlang.org/docs/handbook/2/types-from-types.html",
    "https://www.typescriptlang.org/docs/handbook/2/conditional-types.html",
    "https://www.typescriptlang.org/docs/handbook/2/mapped-types.html",
    "https://www.typescriptlang.org/docs/handbook/2/template-literal-types.html",
    "https://www.typescriptlang.org/docs/handbook/declaration-files/introduction.html",
    "https://www.typescriptlang.org/docs/handbook/decorators.html",
    "https://www.typescriptlang.org/docs/handbook/2/classes.html",
    "https://www.typescriptlang.org/docs/handbook/module-resolution.html",
  ],

  react: [
    "https://react.dev/learn/thinking-in-react",
    "https://react.dev/learn/passing-props-to-a-component",
    "https://react.dev/learn/state-a-components-memory",
    "https://react.dev/learn/render-and-commit",
    "https://react.dev/learn/queueing-a-series-of-state-updates",
    "https://react.dev/learn/synchronizing-with-effects",
    "https://react.dev/learn/you-might-not-need-an-effect",
    "https://react.dev/learn/managing-state",
    "https://react.dev/learn/scaling-up-with-reducer-and-context",
    "https://react.dev/reference/react/useState",
    "https://react.dev/reference/react/useEffect",
    "https://react.dev/reference/react/useContext",
    "https://react.dev/reference/react/useReducer",
    "https://react.dev/reference/react/useMemo",
    "https://react.dev/reference/react/useCallback",
    "https://react.dev/reference/react/useRef",
    "https://react.dev/reference/react/memo",
    "https://react.dev/learn/reusing-logic-with-custom-hooks",
    "https://react.dev/learn/keeping-components-pure",
    "https://react.dev/reference/react/Suspense",
  ],

  nextjs: [
    "https://nextjs.org/docs/app/building-your-application/routing/defining-routes",
    "https://nextjs.org/docs/app/building-your-application/routing/layouts-and-templates",
    "https://nextjs.org/docs/app/building-your-application/routing/linking-and-navigating",
    "https://nextjs.org/docs/app/building-your-application/routing/loading-ui-and-streaming",
    "https://nextjs.org/docs/app/building-your-application/routing/error-handling",
    "https://nextjs.org/docs/app/building-your-application/rendering/server-components",
    "https://nextjs.org/docs/app/building-your-application/rendering/client-components",
    "https://nextjs.org/docs/app/building-your-application/data-fetching/fetching-caching-and-revalidating",
    "https://nextjs.org/docs/app/api-reference/functions/server-actions",
    "https://nextjs.org/docs/app/building-your-application/optimizing/images",
    "https://nextjs.org/docs/app/building-your-application/optimizing/fonts",
    "https://nextjs.org/docs/app/building-your-application/optimizing/metadata",
    "https://nextjs.org/docs/app/building-your-application/deploying",
    "https://nextjs.org/docs/app/api-reference/file-conventions/middleware",
  ],

  nodejs: [
    "https://nodejs.org/en/learn/getting-started/introduction-to-nodejs",
    "https://nodejs.org/en/learn/asynchronous-work/asynchronous-flow-control",
    "https://nodejs.org/en/learn/asynchronous-work/overview-of-blocking-vs-non-blocking",
    "https://nodejs.org/en/learn/asynchronous-work/javascript-asynchronous-programming-and-callbacks",
    "https://nodejs.org/en/learn/asynchronous-work/the-nodejs-event-emitter",
    "https://nodejs.org/en/learn/modules/how-to-use-streams",
    "https://nodejs.org/en/learn/manipulating-files/reading-files-with-nodejs",
    "https://nodejs.org/en/learn/manipulating-files/writing-files-with-nodejs",
    "https://nodejs.org/en/learn/command-line/how-to-work-with-environment-variables-in-nodejs",
    "https://nodejs.org/en/learn/getting-started/nodejs-with-typescript",
    "https://nodejs.org/en/learn/modules/the-module-scope",
    "https://nodejs.org/en/learn/getting-started/differences-between-nodejs-and-the-browser",
  ],

  sql: [
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-joins/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-inner-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-left-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-self-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-full-outer-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-cross-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-indexes/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-transaction/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-primary-key/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-foreign-key/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-subquery/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-cte/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-window-function/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-aggregate-functions/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-group-by/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-having/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-stored-procedures/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-triggers/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-views/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-materialized-views/",
  ],

  aws: [
    "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/lambda-invocation.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html",
    "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
    "https://docs.aws.amazon.com/AmazonS3/latest/userguide/storage-class-intro.html",
    "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html",
    "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
    "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
    "https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Subnets.html",
    "https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html",
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html",
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html",
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html",
    "https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html",
    "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html",
    "https://docs.aws.amazon.com/AmazonDynamoDB/latest/developerguide/Introduction.html",
    "https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html",
    "https://docs.aws.amazon.com/sns/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/welcome.html",
    "https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html",
  ],

  docker: [
    "https://docs.docker.com/get-started/docker-overview/",
    "https://docs.docker.com/get-started/",
    "https://docs.docker.com/build/concepts/dockerfile/",
    "https://docs.docker.com/compose/intro/features-uses/",
    "https://docs.docker.com/compose/compose-file/",
    "https://docs.docker.com/storage/volumes/",
    "https://docs.docker.com/network/",
    "https://docs.docker.com/engine/security/",
    "https://docs.docker.com/build/building/multi-stage/",
    "https://docs.docker.com/build/building/best-practices/",
    "https://docs.docker.com/engine/swarm/",
    "https://docs.docker.com/registry/",
  ],

  kubernetes: [
    "https://kubernetes.io/docs/concepts/overview/",
    "https://kubernetes.io/docs/concepts/workloads/pods/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/",
    "https://kubernetes.io/docs/concepts/services-networking/service/",
    "https://kubernetes.io/docs/concepts/services-networking/ingress/",
    "https://kubernetes.io/docs/concepts/configuration/configmap/",
    "https://kubernetes.io/docs/concepts/configuration/secret/",
    "https://kubernetes.io/docs/concepts/storage/persistent-volumes/",
    "https://kubernetes.io/docs/concepts/storage/volumes/",
    "https://kubernetes.io/docs/concepts/scheduling-eviction/",
    "https://kubernetes.io/docs/concepts/security/rbac-good-practices/",
    "https://kubernetes.io/docs/concepts/cluster-administration/networking/",
    "https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/",
    "https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/",
  ],

  git: [
    "https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F",
    "https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository",
    "https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell",
    "https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging",
    "https://git-scm.com/book/en/v2/Git-Branching-Rebasing",
    "https://git-scm.com/book/en/v2/Git-Tools-Stashing-and-Cleaning",
    "https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History",
    "https://git-scm.com/book/en/v2/Git-Tools-Submodules",
    "https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project",
    "https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project",
    "https://git-scm.com/book/en/v2/Git-Internals-Git-Objects",
    "https://git-scm.com/book/en/v2/Git-Internals-Git-References",
  ],

  python: [
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://docs.python.org/3/tutorial/errors.html",
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/tutorial/stdlib.html",
    "https://docs.python.org/3/tutorial/stdlib2.html",
    "https://docs.python.org/3/library/functools.html",
    "https://docs.python.org/3/library/itertools.html",
    "https://docs.python.org/3/library/asyncio.html",
    "https://docs.python.org/3/library/typing.html",
    "https://docs.python.org/3/library/dataclasses.html",
    "https://docs.python.org/3/library/contextlib.html",
    "https://docs.python.org/3/library/collections.html",
    "https://docs.python.org/3/howto/descriptor.html",
    "https://docs.python.org/3/howto/logging.html",
    "https://docs.python.org/3/reference/datamodel.html",
  ],

  dart: [
    "https://dart.dev/language",
    "https://dart.dev/language/variables",
    "https://dart.dev/language/built-in-types",
    "https://dart.dev/language/functions",
    "https://dart.dev/language/loops",
    "https://dart.dev/language/branches",
    "https://dart.dev/language/error-handling",
    "https://dart.dev/language/classes",
    "https://dart.dev/language/constructors",
    "https://dart.dev/language/mixins",
    "https://dart.dev/language/generics",
    "https://dart.dev/language/async",
    "https://dart.dev/language/isolates",
    "https://dart.dev/null-safety",
    "https://dart.dev/effective-dart/style",
    "https://dart.dev/effective-dart/design",
  ],

  flutter: [
    "https://docs.flutter.dev/get-started/flutter-for/uikit-devs",
    "https://docs.flutter.dev/ui",
    "https://docs.flutter.dev/ui/widgets-intro",
    "https://docs.flutter.dev/ui/layout",
    "https://docs.flutter.dev/ui/adaptive-responsive",
    "https://docs.flutter.dev/data-and-backend/state-mgmt/intro",
    "https://docs.flutter.dev/data-and-backend/state-mgmt/options",
    "https://docs.flutter.dev/data-and-backend/networking",
    "https://docs.flutter.dev/data-and-backend/serialization/json",
    "https://docs.flutter.dev/cookbook/navigation/navigation-basics",
    "https://docs.flutter.dev/cookbook/navigation/named-routes",
    "https://docs.flutter.dev/cookbook/persistence/key-value",
    "https://docs.flutter.dev/cookbook/persistence/sqlite",
    "https://docs.flutter.dev/ui/animations",
    "https://docs.flutter.dev/testing/overview",
    "https://docs.flutter.dev/deployment/android",
  ],

  golang: [
    "https://go.dev/tour/welcome/1",
    "https://go.dev/doc/effective_go",
    "https://go.dev/blog/goroutines",
    "https://go.dev/blog/pipelines",
    "https://go.dev/blog/context",
    "https://go.dev/blog/error-handling-and-go",
    "https://go.dev/blog/go-maps-in-action",
    "https://go.dev/blog/slices-intro",
    "https://go.dev/blog/defer-panic-and-recover",
    "https://go.dev/doc/faq",
    "https://go.dev/blog/using-go-modules",
    "https://go.dev/blog/race-detector",
    "https://go.dev/blog/pprof",
    "https://go.dev/blog/http-tracing",
  ],

  rust: [
    "https://doc.rust-lang.org/book/ch01-00-getting-started.html",
    "https://doc.rust-lang.org/book/ch03-00-common-programming-concepts.html",
    "https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html",
    "https://doc.rust-lang.org/book/ch05-00-structs.html",
    "https://doc.rust-lang.org/book/ch06-00-enums.html",
    "https://doc.rust-lang.org/book/ch08-00-common-collections.html",
    "https://doc.rust-lang.org/book/ch09-00-error-handling.html",
    "https://doc.rust-lang.org/book/ch10-00-generics.html",
    "https://doc.rust-lang.org/book/ch13-00-functional-features.html",
    "https://doc.rust-lang.org/book/ch15-00-smart-pointers.html",
    "https://doc.rust-lang.org/book/ch16-00-concurrency.html",
    "https://doc.rust-lang.org/book/ch17-00-oop.html",
    "https://doc.rust-lang.org/book/ch20-00-final-project-a-web-server.html",
  ],

  java: [
    "https://dev.java/learn/getting-started/",
    "https://dev.java/learn/oop/",
    "https://dev.java/learn/lambdas/",
    "https://dev.java/learn/streams/",
    "https://dev.java/learn/generics/",
    "https://dev.java/learn/exceptions/",
    "https://dev.java/learn/collections/",
    "https://dev.java/learn/concurrency/",
    "https://www.baeldung.com/java-memory-management-interview-questions",
    "https://www.baeldung.com/java-8-streams",
    "https://www.baeldung.com/java-optional",
    "https://www.baeldung.com/java-design-patterns-series",
  ],

  spring: [
    "https://docs.spring.io/spring-boot/docs/current/reference/html/getting-started.html",
    "https://docs.spring.io/spring-boot/docs/current/reference/html/features.html",
    "https://docs.spring.io/spring-framework/reference/web/webmvc.html",
    "https://docs.spring.io/spring-framework/reference/data-access.html",
    "https://docs.spring.io/spring-security/reference/index.html",
    "https://www.baeldung.com/spring-boot-annotations",
    "https://www.baeldung.com/spring-mvc-annotations",
    "https://www.baeldung.com/transaction-configuration-with-jpa-and-spring",
  ],

  system_design: [
    "https://www.techinterviewhandbook.org/system-design/",
    "https://www.designgurus.io/blog/system-design-interview-fundamentals",
    "https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction",
    "https://highscalability.com/blog/2016/1/11/a-beginners-guide-to-scaling-to-11-million-users-on-amazons.html",
    "https://www.educative.io/blog/complete-guide-to-system-design",
  ],

  design_patterns: [
    "https://refactoring.guru/design-patterns/creational-patterns",
    "https://refactoring.guru/design-patterns/singleton",
    "https://refactoring.guru/design-patterns/factory-method",
    "https://refactoring.guru/design-patterns/abstract-factory",
    "https://refactoring.guru/design-patterns/builder",
    "https://refactoring.guru/design-patterns/prototype",
    "https://refactoring.guru/design-patterns/structural-patterns",
    "https://refactoring.guru/design-patterns/adapter",
    "https://refactoring.guru/design-patterns/decorator",
    "https://refactoring.guru/design-patterns/facade",
    "https://refactoring.guru/design-patterns/proxy",
    "https://refactoring.guru/design-patterns/behavioral-patterns",
    "https://refactoring.guru/design-patterns/observer",
    "https://refactoring.guru/design-patterns/strategy",
    "https://refactoring.guru/design-patterns/command",
    "https://refactoring.guru/design-patterns/iterator",
    "https://refactoring.guru/design-patterns/chain-of-responsibility",
    "https://www.patterns.dev/vanilla/singleton-pattern/",
    "https://www.patterns.dev/vanilla/observer-pattern/",
    "https://www.patterns.dev/react/hooks-pattern/",
  ],

  dsa: [
    "https://www.techinterviewhandbook.org/algorithms/array/",
    "https://www.techinterviewhandbook.org/algorithms/string/",
    "https://www.techinterviewhandbook.org/algorithms/hash-table/",
    "https://www.techinterviewhandbook.org/algorithms/recursion/",
    "https://www.techinterviewhandbook.org/algorithms/sorting-searching/",
    "https://www.techinterviewhandbook.org/algorithms/tree/",
    "https://www.techinterviewhandbook.org/algorithms/graph/",
    "https://www.techinterviewhandbook.org/algorithms/dynamic-programming/",
    "https://www.techinterviewhandbook.org/algorithms/binary-search/",
    "https://www.techinterviewhandbook.org/algorithms/heap/",
    "https://www.techinterviewhandbook.org/algorithms/queue/",
    "https://www.techinterviewhandbook.org/algorithms/stack/",
    "https://www.techinterviewhandbook.org/algorithms/linked-list/",
    "https://www.techinterviewhandbook.org/algorithms/matrix/",
    "https://www.techinterviewhandbook.org/algorithms/interval/",
    "https://www.techinterviewhandbook.org/algorithms/trie/",
    "https://www.techinterviewhandbook.org/coding-interview-cheatsheet/",
    "https://www.techinterviewhandbook.org/best-practice-questions/",
  ],

  behavioral: [
    "https://www.techinterviewhandbook.org/behavioral-interview-questions/",
    "https://www.techinterviewhandbook.org/star-method/",
    "https://www.techinterviewhandbook.org/coding-interview-mistakes/",
    "https://www.techinterviewhandbook.org/self-introduction/",
    "https://www.techinterviewhandbook.org/final-questions/",
    "https://www.techinterviewhandbook.org/negotiation/",
    "https://www.techinterviewhandbook.org/preparing-for-a-system-design-interview/",
  ],

  graphql: [
    "https://graphql.org/learn/",
    "https://graphql.org/learn/queries/",
    "https://graphql.org/learn/mutations/",
    "https://graphql.org/learn/subscriptions/",
    "https://graphql.org/learn/schema/",
    "https://graphql.org/learn/execution/",
    "https://graphql.org/learn/introspection/",
    "https://graphql.org/learn/best-practices/",
    "https://graphql.org/learn/thinking-in-graphs/",
    "https://graphql.org/learn/authorization/",
    "https://graphql.org/learn/pagination/",
    "https://graphql.org/learn/caching/",
  ],

  rest_api: [
    "https://restfulapi.net/",
    "https://restfulapi.net/rest-architectural-constraints/",
    "https://restfulapi.net/resource-naming/",
    "https://restfulapi.net/http-methods/",
    "https://restfulapi.net/http-status-codes/",
    "https://restfulapi.net/rest-api-design-tutorial-with-example/",
    "https://restfulapi.net/caching/",
    "https://restfulapi.net/statelessness/",
    "https://restfulapi.net/hateoas/",
    "https://restfulapi.net/rest-api-versioning/",
    "https://restfulapi.net/security-essentials/",
  ],

  mongodb: [
    "https://www.mongodb.com/docs/manual/introduction/",
    "https://www.mongodb.com/docs/manual/core/document/",
    "https://www.mongodb.com/docs/manual/crud/",
    "https://www.mongodb.com/docs/manual/aggregation/",
    "https://www.mongodb.com/docs/manual/indexes/",
    "https://www.mongodb.com/docs/manual/core/transactions/",
    "https://www.mongodb.com/docs/manual/replication/",
    "https://www.mongodb.com/docs/manual/sharding/",
    "https://www.mongodb.com/docs/manual/security/",
    "https://www.mongodb.com/docs/manual/data-modeling/",
    "https://www.mongodb.com/docs/manual/changeStreams/",
    "https://www.mongodb.com/docs/manual/core/timeseries-collections/",
  ],

  redis: [
    "https://redis.io/docs/latest/develop/get-started/",
    "https://redis.io/docs/latest/develop/data-types/",
    "https://redis.io/docs/latest/develop/data-types/strings/",
    "https://redis.io/docs/latest/develop/data-types/lists/",
    "https://redis.io/docs/latest/develop/data-types/sets/",
    "https://redis.io/docs/latest/develop/data-types/sorted-sets/",
    "https://redis.io/docs/latest/develop/data-types/hashes/",
    "https://redis.io/docs/latest/develop/data-types/streams/",
    "https://redis.io/docs/latest/develop/interact/transactions/",
    "https://redis.io/docs/latest/develop/interact/pubsub/",
    "https://redis.io/docs/latest/operate/rs/clusters/",
    "https://redis.io/docs/latest/develop/interact/search-and-query/",
  ],

  linux: [
    "https://linuxcommand.org/lc3_learning_the_shell.php",
    "https://linuxcommand.org/lc3_writing_shell_scripts.php",
    "https://linuxize.com/post/understanding-linux-file-permissions/",
    "https://linuxize.com/post/how-to-use-grep-command-in-linux/",
    "https://linuxize.com/post/how-to-use-sed-command-in-linux/",
    "https://linuxize.com/post/how-to-use-awk/",
    "https://linuxize.com/post/how-to-set-and-list-environment-variables-in-linux/",
    "https://linuxize.com/post/ssh-command-in-linux/",
    "https://linuxize.com/post/linux-cron-jobs/",
    "https://linuxize.com/post/how-to-use-linux-crontab/",
  ],

  networking: [
    "https://www.cloudflare.com/learning/dns/what-is-dns/",
    "https://www.cloudflare.com/learning/dns/dns-records/",
    "https://www.cloudflare.com/learning/network-layer/what-is-a-subnet/",
    "https://www.cloudflare.com/learning/network-layer/what-is-a-packet/",
    "https://www.cloudflare.com/learning/network-layer/internet-protocol/",
    "https://www.cloudflare.com/learning/ssl/what-is-ssl/",
    "https://www.cloudflare.com/learning/ssl/what-is-https/",
    "https://www.cloudflare.com/learning/ssl/what-is-a-tls-handshake/",
    "https://www.cloudflare.com/learning/access-management/what-is-oauth/",
    "https://www.cloudflare.com/learning/performance/what-is-http2/",
    "https://www.cloudflare.com/learning/performance/http3-vs-http2/",
    "https://www.cloudflare.com/learning/cdn/what-is-a-cdn/",
  ],

  security: [
    "https://owasp.org/www-project-top-ten/",
    "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/JWT_Security_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/CORS_Security_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Security_Response_Headers_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html",
  ],

  cicd: [
    "https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions",
    "https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions",
    "https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-nodejs",
    "https://docs.github.com/en/actions/deployment/about-deployments/deploying-with-github-actions",
    "https://docs.github.com/en/actions/security-guides/encrypted-secrets",
    "https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs",
    "https://docs.github.com/en/actions/using-containerized-services/about-service-containers",
    "https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows",
    "https://docs.github.com/en/actions/creating-actions/creating-a-docker-container-action",
  ],

  testing: [
    "https://jestjs.io/docs/getting-started",
    "https://jestjs.io/docs/using-matchers",
    "https://jestjs.io/docs/asynchronous",
    "https://jestjs.io/docs/mock-functions",
    "https://jestjs.io/docs/timer-mocks",
    "https://jestjs.io/docs/snapshot-testing",
    "https://testing-library.com/docs/react-testing-library/intro",
    "https://testing-library.com/docs/queries/about",
    "https://testing-library.com/docs/user-event/intro",
    "https://playwright.dev/docs/intro",
    "https://playwright.dev/docs/writing-tests",
    "https://playwright.dev/docs/test-assertions",
    "https://playwright.dev/docs/api-testing",
    "https://vitest.dev/guide/",
    "https://vitest.dev/guide/mocking",
  ],

  prisma: [
    "https://www.prisma.io/docs/getting-started",
    "https://www.prisma.io/docs/concepts/components/prisma-schema",
    "https://www.prisma.io/docs/concepts/components/prisma-client",
    "https://www.prisma.io/docs/concepts/components/prisma-migrate",
    "https://www.prisma.io/docs/concepts/components/prisma-client/crud",
    "https://www.prisma.io/docs/concepts/components/prisma-client/filtering-and-sorting",
    "https://www.prisma.io/docs/concepts/components/prisma-client/aggregation-grouping-summarizing",
    "https://www.prisma.io/docs/concepts/components/prisma-client/transactions",
    "https://www.prisma.io/docs/concepts/components/prisma-client/middleware",
    "https://www.prisma.io/docs/concepts/components/prisma-client/relation-queries",
    "https://www.prisma.io/docs/guides/performance-and-optimization",
  ],

  css: [
    "https://developer.mozilla.org/en-US/docs/Learn/CSS/First_steps/What_is_CSS",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Flexible_Box_Layout/Basic_Concepts_of_Flexbox",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Grid_Layout/Basic_Concepts_of_Grid_Layout",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Animations/Using_CSS_animations",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Transitions/Using_CSS_transitions",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/@media",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Variables",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/Specificity",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/position",
    "https://tailwindcss.com/docs/utility-first",
    "https://tailwindcss.com/docs/responsive-design",
    "https://tailwindcss.com/docs/dark-mode",
    "https://tailwindcss.com/docs/customizing-config",
    "https://tailwindcss.com/docs/adding-custom-styles",
    "https://tailwindcss.com/docs/reusing-styles",
  ],

  terraform: [
    "https://developer.hashicorp.com/terraform/intro",
    "https://developer.hashicorp.com/terraform/language",
    "https://developer.hashicorp.com/terraform/language/resources",
    "https://developer.hashicorp.com/terraform/language/data-sources",
    "https://developer.hashicorp.com/terraform/language/variables",
    "https://developer.hashicorp.com/terraform/language/values/outputs",
    "https://developer.hashicorp.com/terraform/language/modules",
    "https://developer.hashicorp.com/terraform/language/state",
    "https://developer.hashicorp.com/terraform/cli/commands",
    "https://developer.hashicorp.com/terraform/language/providers",
    "https://developer.hashicorp.com/terraform/language/expressions",
    "https://developer.hashicorp.com/terraform/language/functions",
    "https://developer.hashicorp.com/terraform/language/meta-arguments/lifecycle",
  ],

  kafka: [
    "https://kafka.apache.org/documentation/",
    "https://kafka.apache.org/documentation/#gettingStarted",
    "https://kafka.apache.org/documentation/#design",
    "https://kafka.apache.org/documentation/#producerapi",
    "https://kafka.apache.org/documentation/#consumerapi",
    "https://kafka.apache.org/documentation/#streams",
    "https://kafka.apache.org/documentation/#connect",
    "https://kafka.apache.org/documentation/#replication",
    "https://kafka.apache.org/documentation/#operations",
    "https://kafka.apache.org/documentation/#security",
  ],

  ai_ml: [
    "https://developers.google.com/machine-learning/crash-course/ml-intro",
    "https://developers.google.com/machine-learning/crash-course/framing/ml-terminology",
    "https://developers.google.com/machine-learning/crash-course/descending-into-ml/linear-regression",
    "https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent",
    "https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc",
    "https://developers.google.com/machine-learning/crash-course/neural-networks/nodes-and-hidden-layers",
    "https://huggingface.co/docs/transformers/index",
    "https://huggingface.co/learn/nlp-course/chapter1/1",
    "https://huggingface.co/learn/nlp-course/chapter2/1",
    "https://huggingface.co/learn/nlp-course/chapter3/1",
  ],

  web_performance: [
    "https://web.dev/articles/vitals",
    "https://web.dev/articles/lcp",
    "https://web.dev/articles/fid",
    "https://web.dev/articles/cls",
    "https://web.dev/articles/ttfb",
    "https://web.dev/articles/inp",
    "https://web.dev/articles/fast",
    "https://web.dev/articles/critical-rendering-path",
    "https://web.dev/articles/image-cdns",
    "https://web.dev/articles/caching-case-study",
    "https://web.dev/articles/service-workers-cache-storage",
    "https://web.dev/articles/performance-budgets-101",
  ],
};

// ─────────────────────────────────────────
// TARGETS — all use hand-picked fallback URLs
// NO sitemaps → no explosion risk
// ─────────────────────────────────────────
const targets = [
  { category: "react", fallback: FALLBACKS.react },
  { category: "nodejs", fallback: FALLBACKS.nodejs },
  { category: "docker", fallback: FALLBACKS.docker },
  { category: "kubernetes", fallback: FALLBACKS.kubernetes },
  { category: "javascript", fallback: FALLBACKS.javascript },
  { category: "typescript", fallback: FALLBACKS.typescript },
  { category: "nextjs", fallback: FALLBACKS.nextjs },
  { category: "python", fallback: FALLBACKS.python },
  { category: "dart", fallback: FALLBACKS.dart },
  { category: "flutter", fallback: FALLBACKS.flutter },
  { category: "golang", fallback: FALLBACKS.golang },
  { category: "rust", fallback: FALLBACKS.rust },
  { category: "java", fallback: FALLBACKS.java },
  { category: "spring", fallback: FALLBACKS.spring },
  { category: "sql", fallback: FALLBACKS.sql },
  { category: "mongodb", fallback: FALLBACKS.mongodb },
  { category: "redis", fallback: FALLBACKS.redis },
  { category: "prisma", fallback: FALLBACKS.prisma },
  { category: "aws", fallback: FALLBACKS.aws },
  { category: "terraform", fallback: FALLBACKS.terraform },
  { category: "kafka", fallback: FALLBACKS.kafka },
  { category: "git", fallback: FALLBACKS.git },
  { category: "linux", fallback: FALLBACKS.linux },
  { category: "networking", fallback: FALLBACKS.networking },
  { category: "security", fallback: FALLBACKS.security },
  { category: "cicd", fallback: FALLBACKS.cicd },
  { category: "testing", fallback: FALLBACKS.testing },
  { category: "graphql", fallback: FALLBACKS.graphql },
  { category: "rest_api", fallback: FALLBACKS.rest_api },
  { category: "css", fallback: FALLBACKS.css },
  { category: "design_patterns", fallback: FALLBACKS.design_patterns },
  { category: "dsa", fallback: FALLBACKS.dsa },
  { category: "system_design", fallback: FALLBACKS.system_design },
  { category: "ai_ml", fallback: FALLBACKS.ai_ml },
  { category: "web_performance", fallback: FALLBACKS.web_performance },
  { category: "behavioral", fallback: FALLBACKS.behavioral },
];

// ─────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────

function isRateLimitError(err) {
  const msg = String(err?.message ?? err ?? "");
  return (
    msg.includes("429") ||
    msg.includes("RESOURCE_EXHAUSTED") ||
    msg.includes("Quota exceeded")
  );
}

/**
 * Embed with timeout + retry.
 * Uses batchEmbedTexts → 1 API call for ALL texts in the array.
 * Before: 50 chunks = 50 API calls
 * After:  50 chunks = 1 API call
 */
async function embedWithRetry(texts, retries = 5) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const vectors = await Promise.race([
        batchEmbedTexts(texts), // 👈 1 call for all texts
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("Embed timed out after 30s")),
            EMBED_TIMEOUT_MS,
          ),
        ),
      ]);

      const emptyCount = vectors.filter((v) => !v || v.length === 0).length;
      if (emptyCount > 0) throw new Error(`${emptyCount} empty vectors`);
      return vectors;
    } catch (err) {
      if (attempt === retries) {
        console.error(`  ❌ Failed after ${retries} attempts: ${err.message}`);
        throw err;
      }
      const waitMs = isRateLimitError(err) ? 15000 * attempt : 3000 * attempt;
      console.warn(
        `  ⚠️  Attempt ${attempt}/${retries}${isRateLimitError(err) ? " [RATE LIMIT]" : ""} — retrying in ${waitMs / 1000}s`,
      );
      await new Promise((r) => setTimeout(r, waitMs));
    }
  }
}

/**
 * Split chunk array into EMBED_BATCH_SIZE groups.
 * Each group = 1 API call (not 1 call per chunk).
 */
async function embedInBatches(chunks) {
  const texts = chunks.map((c) => c.pageContent);
  let allVectors = [];
  const totalBatches = Math.ceil(texts.length / EMBED_BATCH_SIZE);

  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
    const batchNum = Math.floor(i / EMBED_BATCH_SIZE) + 1;
    console.log(
      `  🔢 Batch ${batchNum}/${totalBatches} — ${batch.length} chunks → 1 API call`,
    );

    const vectors = await embedWithRetry(batch);
    allVectors = [...allVectors, ...vectors];

    if (i + EMBED_BATCH_SIZE < texts.length) {
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  return allVectors;
}

/** Fetch page, strip noise, return cleaned text */
async function fetchAndCleanPage(url) {
  try {
    const res = await fetch(url, {
      signal: AbortSignal.timeout(12000),
      headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
    });
    if (!res.ok) return null;
    const html = await res.text();
    const $ = cheerio.load(html);

    $(
      "nav, footer, aside, script, style, noscript, header, button, svg, img",
    ).remove();
    $(".ad, .on-this-page, .cookie-banner, .search-bar, #docsearch").remove();
    $(
      "[class*='sidebar'], [class*='nav'], [class*='menu'], [class*='toc']",
    ).remove();

    let content =
      $("article").text() ||
      $("main").text() ||
      $(".content").text() ||
      $("body").text();

    content = content.replace(/\s+/g, " ").trim();
    if (content.length < 300) return null;

    return { url, title: $("h1").first().text().trim() || url, content };
  } catch {
    return null;
  }
}

// ─────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────
async function main() {
  const totalTopics = targets.length;
  const totalUrls = targets.reduce((sum, t) => sum + t.fallback.length, 0);

  console.log("╔══════════════════════════════════════════════════╗");
  console.log(`║  🚀 Ingestion — ${totalTopics} topics, ${totalUrls} URLs`);
  console.log(`║  ⚡ ${EMBED_BATCH_SIZE} chunks per API call (batch mode)`);
  console.log(`║  🌍 Multi-region rotation active in gemini.js`);
  console.log("╚══════════════════════════════════════════════════╝\n");

  // Sanity check before wiping DB
  console.log("🔬 Testing embedding API...");
  try {
    const testVec = await embedText("hello world");
    if (!testVec || testVec.length === 0) throw new Error("Empty vector");
    console.log(`✅ Embedding OK — ${testVec.length} dimensions\n`);
  } catch (err) {
    console.error(`❌ Embedding FAILED: ${err.message}`);
    process.exit(1);
  }

  console.log("🧹 Clearing database...");
  await prisma.$executeRaw`TRUNCATE TABLE "Document" RESTART IDENTITY CASCADE`;
  console.log("✅ Cleared.\n");

  const startTime = Date.now();
  let grandTotal = 0;

  for (const [idx, target] of targets.entries()) {
    console.log(`\n${"─".repeat(55)}`);
    console.log(
      `📚 [${idx + 1}/${totalTopics}] ${target.category.toUpperCase()}`,
    );
    console.log("─".repeat(55));

    const urls = target.fallback.slice(0, MAX_URLS_PER_CATEGORY);
    console.log(`  📋 ${urls.length} URLs to process`);

    let categoryTotal = 0;

    for (let i = 0; i < urls.length; i += PAGE_BATCH_SIZE) {
      const batchUrls = urls.slice(i, i + PAGE_BATCH_SIZE);
      const rawPages = await Promise.all(
        batchUrls.map((url) => fetchAndCleanPage(url)),
      );
      const validPages = rawPages.filter((p) => p && p.content.length > 300);

      console.log(
        `\n  📄 Page batch ${Math.floor(i / PAGE_BATCH_SIZE) + 1}: ${validPages.length}/${batchUrls.length} valid`,
      );

      if (validPages.length === 0) {
        await new Promise((r) => setTimeout(r, DELAY_MS));
        continue;
      }

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 100,
      });

      let allChunks = [];
      for (const page of validPages) {
        const chunks = await splitter.createDocuments(
          [page.content],
          [{ source: page.url, title: page.title, category: target.category }],
        );
        allChunks = [...allChunks, ...chunks];
      }

      const sanitized = allChunks.filter(
        (c) => c.pageContent && c.pageContent.trim().length > 5,
      );

      console.log(`  🧩 ${sanitized.length} chunks to embed`);

      if (sanitized.length > 0) {
        try {
          const vectors = await embedInBatches(sanitized);

          let saved = 0;
          for (let j = 0; j < sanitized.length; j++) {
            const chunk = sanitized[j];
            const vector = vectors[j];
            if (!vector || vector.length === 0) continue;

            await prisma.$executeRaw`
              INSERT INTO "Document" (content, embedding, metadata, source)
              VALUES (
                ${chunk.pageContent},
                ${`[${vector.join(",")}]`}::vector,
                ${JSON.stringify(chunk.metadata)}::jsonb,
                ${chunk.metadata.source}
              )
            `;
            saved++;
          }

          categoryTotal += saved;
          grandTotal += saved;
          console.log(
            `  ✅ Saved ${saved} chunks (category total: ${categoryTotal})`,
          );
        } catch (err) {
          console.error(`  ❌ Error: ${err.message}`);
        }
      }

      await new Promise((r) => setTimeout(r, DELAY_MS));
    }

    const elapsed = Math.round((Date.now() - startTime) / 1000);
    console.log(
      `  🏁 [${target.category}] — ${categoryTotal} chunks | ${Math.floor(elapsed / 60)}m ${elapsed % 60}s total`,
    );
  }

  const totalSecs = Math.round((Date.now() - startTime) / 1000);
  console.log("\n╔══════════════════════════════════════════════════╗");
  console.log(
    `║  🎉 Done! ${grandTotal} chunks in ${Math.floor(totalSecs / 60)}m ${totalSecs % 60}s`,
  );
  console.log("╚══════════════════════════════════════════════════╝\n");

  await prisma.$disconnect();
}

main().catch((e) => {
  console.error("Fatal Error:", e);
  process.exit(1);
});
