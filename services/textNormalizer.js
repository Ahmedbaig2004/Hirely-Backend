// ─────────────────────────────────────────────────────────────────────────────
// TECHNICAL INTERVIEW TTS NORMALIZER – Optimized for Deepgram Aura-2 (Luna)
// Works perfectly with: aura-2-luna-en, aura-2-asteria-en, aura-2-hermes-en
// ─────────────────────────────────────────────────────────────────────────────

const PRONUNCIATION_MAP = {
  // ── Languages & Frameworks ──────────────────────────────────────────────
  js: "J S",
  ts: "TypeScript",
  jsx: "J S X.",
  tsx: "T S X.",
  "c#": "C Sharp",
  "c++": "C Plus Plus",
  "f#": "F Sharp",
  ".net": "Dot Net",
  php: "P H P.",
  go: "Go",
  rust: "Rust",
  kotlin: "Kotlin",
  swift: "Swift",
  dart: "Dart",
  flutter: "Flutter",
  react: "React",
  "react.js": "React J S.",
  "next.js": "Next J S.",
  nuxt: "Nuked",
  svelte: "Svelt",
  angular: "Angular",
  vue: "View",
  "vue.js": "View J S ",
  "node.js": "Node J S ",
  "express.js": "Express J S ",
  nestjs: "Nest J S ",
  fastify: "Fastify",
  "socket.io": "Socket I O", // <--- ADD THIS LINE
  socketio: "Socket I O",

  // ── Databases & Storage ─────────────────────────────────────────────────
  sql: "S Q L.",
  nosql: "No S Q L ",
  postgresql: "Postgres Q L.",
  postgres: "Postgres",
  mysql: "My S Q L.",
  mariadb: "Maria D B.",
  mongodb: "Mongo D B.",
  redis: "Red Iss",
  dynamodb: "Dynamo D B.",
  firestore: "Fire Store",
  supabase: "Soup a base",
  prisma: "Prisma",

  // ── Cloud & DevOps ──────────────────────────────────────────────────────
  aws: "A W S ",
  gcp: "G C P.",
  azure: "Azure", // Luna says this correctly now
  ec2: "E C Two",
  s3: "S Three",
  rds: "R D S ",
  lambda: "Lambda",
  cloudfront: "Cloud Front",
  docker: "Docker",
  kubernetes: "Kubernetes",
  k8s: "Kubernates",
  terraform: "Terra Form",
  nginx: "Engine X",
  apache: "Apache",
  github: "Git Hub",
  gitlab: "Git Lab",
  vercel: "Ver sel",
  netlify: "Net li fy",

  // ── Core Acronyms & Jargon ──────────────────────────────────────────────
  api: " A.P.I ",
  apis: " A.P.I.s ",
  rpc: "R P C ",
  rest: "Rest",
  graphql: "Graph Q L.",
  grpc: "G R P C ",
  http: "H T T P ",
  https: "H T T P S ",
  jwt: "J W T ",
  oauth: "Oh Auth",
  sso: "S S O ",
  csrf: "C S R F ",
  xss: "X S S ",
  cors: "Cors",
  dom: "D O M ",
  html: "H T M L ",
  css: "C S S ",
  json: "Jason",
  xml: "X M L ",
  yaml: "Yammel",
  toml: "Tommel",
  cli: "C L I ",
  gui: "G U I ",
  ui: "U I ",
  ux: "U x ",
  spa: "S P A ",
  ssr: "S S R ",
  csr: "C S R ",
  ssg: "S S G ",
  isr: "I S R ",
  seo: "S E O ",
  jamstack: "Jam Stack",
  crud: "Crud",
  orm: "O R M ",
  sdk: "S D K ",
  npm: "N P M ",
  yarn: "Yarn",
  pnpm: "P N P M ",
  dns: "D N S ",
  ssl: "S S L ",
  tls: "T L S ",
  ssh: "S S H ",
  url: "U R L ",
  uri: "U R I ",
  uuid: "Universally Unique Identifier",
  regex: "Rej ex",
  regexp: "Rej exp",

  // ── Architecture & Patterns ─────────────────────────────────────────────
  mvc: "M V C ",
  mvvm: "M V V M ",
  solid: "Solid",
  dry: "D R Y ",
  kiss: "Kiss",
  yagni: "Yag knee",
  oop: "O O P.",
  fp: "F P.",
  tdd: "T D D ",
  bdd: "B D D ",
  ci: "C I ",
  cd: "C D ",
  "ci/cd": "C I  C D ",

  // ── AI / ML ───────────────────────────────────────────────────────────
  llm: "L L M ",
  ai: "A I ",
  ml: "M L ",
  nlp: "N L P ",
  rag: "Rag",
  gpt: "G P T ",
  bert: "Bert",
  transformer: "Transformer",

  // ── Accessibility & Internationalization ───────────────────────────────
  a11y: "Accessibility",
  i18n: "Internationalization",
  l10n: "Localization",

  // ── Code Symbols (for reading snippets aloud) ───────────────────────────
  "===": "triple equals",
  "==": "double equals",
  "!==": "not triple equals",
  "!=": "not equals",
  "=>": "arrow function",
  "&&": "and",
  "||": "or",
  "??": "nullish coalescing",
  "?.": "optional chaining",
  "{": "open curly brace",
  "}": "close curly brace",
  "[": "open bracket",
  "]": "close bracket",
  "(": "open parenthesis",
  ")": "close parenthesis",
  "<": "less than",
  ">": "greater than",
  "&": "ampersand",
  "|": "pipe",
  "@": "at symbol",
  "#": "hash",
  $: "dollar sign",
};

// ── Build Regex Once (sorted longest → shortest) ─────────────────────────────
const ESCAPED_KEYS = Object.keys(PRONUNCIATION_MAP)
  .sort((a, b) => b.length - a.length)
  .map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
  .join("|");

// Match dictionary terms bounded by non-word chars (so "aws" matches but "jaws" doesn't)
const MASTER_REGEX = new RegExp(
  `(?<=\\b|\\W|^)(${ESCAPED_KEYS})(?=\\b|\\W|$)`,
  "gi"
);

/**
 * Splits CamelCase and PascalCase into separate words.
 * e.g., "useEffect" -> "use Effect", "XMLHttpRequest" -> "XML Http Request"
 */
function splitCamelCase(text) {
  return (
    text
      // 1. Break consecutive capitals followed by lowercase (XMLHttp -> XML Http)
      .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
      // 2. Break lowercase followed by capital (camelCase -> camel Case)
      .replace(/([a-z])([A-Z])/g, "$1 $2")
  );
}

export function normalizeForSpeech(text) {
  if (!text) return "";

  let clean = text.trim();

  // 1. DICTIONARY REPLACEMENT (Case-insensitive)
  clean = clean.replace(MASTER_REGEX, (match) => {
    return PRONUNCIATION_MAP[match.toLowerCase()] || match;
  });

  // 2. VERSION NUMBER HANDLING
  // Converts "v18.2.0" or "Node 18.2" to "18 dot 2".
  // (Prevents "18 point 2" which sounds like math, not versioning)
  clean = clean.replace(
    /\b(\d+)\.(\d+)(\.(\d+))?\b/g,
    (match, p1, p2, p3, p4) => {
      // If it looks like a version (e.g. 2.0.1 or v2.0), say "dot"
      return `${p1} dot ${p2}${p4 ? " dot " + p4 : ""}`;
    }
  );

  // 3. CAMELCASE SPLITTER
  // We run this AFTER dictionary (so "XMLHttpRequest" isn't split if it's in the dict,
  // but "getStaticProps" will become "get Static Props")
  clean = splitCamelCase(clean);

  // 4. CLEANUP SYMBOLS FOR AURA-2
  clean = clean
    .replace(/\*\*/g, "") // Remove bold markdown
    .replace(/`/g, "") // Remove code ticks
    .replace(/\[(.*?)\]\(.*?\)/g, "$1") // Remove links: [Title](url) -> Title
    .replace(/[:;]/g, ",") // Deepgram pauses better on commas than semi-colons
    .replace(/\s+/g, " "); // Collapse whitespace

  // 5. AURA-2 PROSODY HACK
  // Aura-2 reads lists better if they end with periods.
  // If a line is short and doesn't end in punctuation, add a period.
  if (!/[.?!]$/.test(clean)) {
    clean += ".";
  }

  return clean;
}
