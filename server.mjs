import 'dotenv/config';
import express from 'express';
import Papa from 'papaparse';
import fetch from 'node-fetch';
import path from 'path';
import { fileURLToPath } from 'url';
import OpenAI from "openai";
import dotenv from "dotenv";

// ------------------------------------
// Key Config
// ------------------------------------
const openaiKeyNODE = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ------------------------------------
// Config
// ------------------------------------
dotenv.config();

const VOCAB = [
  "population", "pop", "people", "residents",
  "hispanic", "spanish",
  "projected", "proj",
  "total", "sum", "overall", "combined",
  "county",
  "region", "area", "east", "west", "central",
  "percent", "percentage", "ratio", "portion", "share"
];

// lower = more sensitive to vague questions
const TOKEN_MATCH_THRESHOLD = 0.1;

const { PORT = 3000, SHEET1_CSV_URL } = process.env;
if (!SHEET1_CSV_URL) throw new Error('Missing SHEET1_CSV_URL in .env');

const app = express();
app.use(express.json());

// Resolve directory (ESM safe)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.static(__dirname));
app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// ------------------------------------
// Helpers
// ------------------------------------
const clean = (s) =>
  String(s ?? '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const tokenize = (s) => clean(s).split(' ').filter(Boolean);

function toNumber(v) {
  if (v == null) return NaN;
  let s = String(v).replace(/\u00a0/g, ' ');
  s = s.replace(/[^0-9.\-]/g, '');
  if (!s.trim()) return NaN;
  const num = Number(s);
  return Number.isFinite(num) ? num : NaN;
}

function formatNumber(metric, value) {
  if (value == null || Number.isNaN(value)) return 'unknown';
  if (metric === 'projected') {
    return Number(value).toLocaleString('en-US', {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    });
  }
  return Math.round(value).toLocaleString('en-US');
}

// ------------------------------------
// Data stores
// ------------------------------------
let rows = [];
let totals = {
  population: 0,
  hispanic: 0,
  projected: 0
};

let regionTotals = {
  east:    { population: 0, hispanic: 0, projected: 0 },
  west:    { population: 0, hispanic: 0, projected: 0 },
  central: { population: 0, hispanic: 0, projected: 0 }
};

// ------------------------------------
// Load CSV
// ------------------------------------
async function loadSheet() {
  const csv = await fetch(SHEET1_CSV_URL).then((r) => {
    if (!r.ok) throw new Error(`Failed to fetch CSV: ${r.status}`);
    return r.text();
  });

  const parsed = Papa.parse(csv, { header: false });
  const data = parsed.data;

  let headerRowIndex = data.findIndex(
    (row) =>
      row &&
      String(row[0] || '').trim() === 'County' &&
      String(row[1] || '').trim() === 'Population'
  );
  if (headerRowIndex === -1) headerRowIndex = 2;

  const headerRow = data[headerRowIndex];

  // Build headers (handle duplicates)
  const headers = [];
  const seen = {};
  for (let i = 0; i < headerRow.length; i++) {
    let base = headerRow[i] ? String(headerRow[i]).trim() : `col_${i}`;
    let name = base;
    let suffix = 2;
    while (seen[name]) name = `${base}_${suffix++}`;
    seen[name] = true;
    headers.push(name);
  }

  const rawRows = [];
  for (let r = headerRowIndex + 1; r < data.length; r++) {
    const arr = data[r];
    if (!arr || arr.every((c) => c == null || c === '')) continue;

    const obj = {};
    for (let c = 0; c < headers.length; c++) obj[headers[c]] = arr[c];
    rawRows.push(obj);
  }

  rows = [];
  totals = { population: 0, hispanic: 0, projected: 0 };
  regionTotals = {
    east:    { population: 0, hispanic: 0, projected: 0 },
    west:    { population: 0, hispanic: 0, projected: 0 },
    central: { population: 0, hispanic: 0, projected: 0 }
  };

  for (const r of rawRows) {
    const c1 = r["County"];
    const c2 = r["County_2"];

    if (!c1 || /^\s*$/.test(c1)) continue;
    if (/total/i.test(c1)) continue;

    const countyLeft = String(c1).trim();
    const countyRight = c2 ? String(c2).trim() : countyLeft;

    const pop = toNumber(r["Population"]);
    const hisPop = toNumber(r["Population - H"]);
    const projPop = toNumber(r["Projected Population - H"]);

    const regionLeft = normalizeRegion(r["Region"]);
    const regionRight = normalizeRegion(r["Region_2"]) || regionLeft;

    // General population
    if (!Number.isNaN(pop)) {
      rows.push({
        county: countyLeft,
        kind: "population",
        population: pop,
        hispanicPopulation: null,
        projectedPopulation: null,
        region: regionLeft
      });

      totals.population += pop;
      if (regionLeft && regionTotals[regionLeft])
        regionTotals[regionLeft].population += pop;
    }

    // Hispanic row
    if (!Number.isNaN(hisPop) || !Number.isNaN(projPop)) {
      rows.push({
        county: countyRight,
        kind: "hispanic",
        population: null,
        hispanicPopulation: !Number.isNaN(hisPop) ? hisPop : null,
        projectedPopulation: !Number.isNaN(projPop) ? projPop : null,
        region: regionRight
      });

      if (!Number.isNaN(hisPop)) totals.hispanic += hisPop;
      if (!Number.isNaN(projPop)) totals.projected += projPop;

      if (regionRight && regionTotals[regionRight]) {
        if (!Number.isNaN(hisPop)) regionTotals[regionRight].hispanic += hisPop;
        if (!Number.isNaN(projPop)) regionTotals[regionRight].projected += projPop;
      }
    }
  }

  console.log(`Loaded ${rows.length} rows.`);
}

// ------------------------------------
// Matching utilities
// ------------------------------------
function extractCounty(q) {
  const cq = clean(q);
  const tokens = cq.split(" ");

  const counties = [...new Set(rows.map(r => clean(r.county)))];

  // Exact match first (strong signal)
  for (const c of counties) {
    if (tokens.includes(c)) {
      return rows.find(r => clean(r.county) === c).county;
    }
  }

  // Fallback: "X county"
  for (const c of counties) {
    if (cq.includes(`${c} county`)) {
      return rows.find(r => clean(r.county) === c).county;
    }
  }

  return null;
}

// guess what metric is being calculated/ouputt to the screen
function guessMetric(q) {
  q = clean(q);

  if (/\b(projected|proj)\b/.test(q) && /\b(hispanic|spanish)\b/.test(q))
    return "projected";

  if (/\b(hispanic|spanish)\b/.test(q))
    return "hispanic";

  if (/\b(projected|proj)\b/.test(q))
    return "projected";

  if (/\b(population|pop|people|residents)\b/.test(q))
    return "population";

  return "population";
}


function wantsTotal(q, countyFound) {
  q = clean(q);
  if (countyFound) return false;
  return /\b(total|overall|combined|all counties|sum)\b/.test(q);
}

function normalizeRegion(raw) {
  if (!raw) return null;
  const r = clean(raw);

  if (r.includes("central")) return "central";
  if (r.includes("east")) return "east";
  if (r.includes("west")) return "west";

  // Support single-letter region codes too
  if (r === "c") return "central";
  if (r === "e") return "east";
  if (r === "w") return "west";

  return null;
}

function extractRegion(q) {
  q = clean(q);

  if (q.includes("central")) return "central";
  if (q.includes("east")) return "east";
  if (q.includes("west")) return "west";

  return null;
}

function asksForRegion(q) {
  q = clean(q);
  return (
    q.startsWith("which region") ||
    q.includes("what region") ||
    q.includes("which area")
  );
}


function wantsPercentage(q) {
  q = clean(q);
  return (
    q.includes("percent") ||
    q.includes("percentage") ||
    q.includes("ratio") ||
    q.includes("portion") ||
    q.includes("share")
  );
}

function findRow(county, metric) {
  const c = clean(county);
  const matches = rows.filter((r) => clean(r.county) === c);

  if (metric === "population") return matches.find((r) => r.population != null);
  if (metric === "hispanic")
    return matches.find((r) => r.hispanicPopulation != null);
  if (metric === "region") return matches.find((r) => r.region != null);
  if (metric === "projected")
    return (
      matches.find((r) => r.projectedPopulation != null) ||
      matches.find((r) => r.hispanicPopulation != null)
    );

  return null;
}

function questionClarityScore(q) {
  const tokens = tokenize(q);
  if (!tokens.length) return 0;

  let matches = 0;
  for (const t of tokens) if (VOCAB.includes(t)) matches++;

  return matches / tokens.length;
}

// ------------------------------------
// Core QA Logic
// ------------------------------------
function answerQuestion(query) {
  const clarity = questionClarityScore(query);
  if (clarity < TOKEN_MATCH_THRESHOLD) {
    return {
      answer:
        "I'm not fully sure what you mean. Try asking about population, Hispanic population, projected population, or regions.",
      meta: { type: "unclear", clarity }
    };
  }

  const county = extractCounty(query);
  const metric = guessMetric(query);
  const region = extractRegion(query);
  const percentFlag = wantsPercentage(query);
  const totalFlag = !percentFlag && wantsTotal(query, county);

  // ------------------------------------
  // COUNTY → REGION QUESTIONS (MUST COME FIRST)
  // e.g. "Which region is Colbert County in?"
  // ------------------------------------
  if (county && asksForRegion(query)) {
    const row = findRow(county, "population"); // any row with region is fine
    if (row && row.region) {
      return {
        answer: `${row.county} County is in the ${row.region} region.`,
        meta: { type: "county", county: row.county, metric: "region" }
      };
    }
  }

  // ------------------------------------
  // REGION % CALCULATIONS
  // ------------------------------------
  if (percentFlag && region) {
    const k = region.toLowerCase();

    if (metric === "population") {
      const pct = (regionTotals[k].population / totals.population) * 100;
      return {
        answer: `${pct.toFixed(1)}% of the total population lives in the ${region} region.`,
        meta: { type: "percent", region, metric }
      };
    }

    if (metric === "hispanic") {
      const pct = (regionTotals[k].hispanic / totals.hispanic) * 100;
      return {
        answer: `${pct.toFixed(1)}% of the Hispanic population lives in the ${region} region.`,
        meta: { type: "percent", region, metric }
      };
    }

    if (metric === "projected") {
      const pct = (regionTotals[k].projected / totals.projected) * 100;
      return {
        answer: `${pct.toFixed(1)}% of the projected Hispanic population is in the ${region} region.`,
        meta: { type: "percent", region, metric }
      };
    }

    return {
      answer: `I can identify the ${region} region, but I can't calculate a percentage for ${metric}.`,
      meta: { type: "error", scope: "region-percent", region, metric }
    };
  }

  // ------------------------------------
  // REGION TOTALS
  // ------------------------------------
  if (totalFlag && region) {
    const k = region.toLowerCase();

    if (metric === "population") {
      return {
        answer: `The total population of the ${region} region is ${formatNumber(
          "population",
          regionTotals[k].population
        )}.`,
        meta: { type: "total", scope: "region", region, metric }
      };
    }

    if (metric === "hispanic") {
      return {
        answer: `The total Hispanic population of the ${region} region is ${formatNumber(
          "hispanic",
          regionTotals[k].hispanic
        )}.`,
        meta: { type: "total", scope: "region", region, metric }
      };
    }

    if (metric === "projected") {
      return {
        answer: `The total projected Hispanic population of the ${region} region is ${formatNumber(
          "projected",
          regionTotals[k].projected
        )}.`,
        meta: { type: "total", scope: "region", region, metric }
      };
    }

    return {
      answer: `I found the ${region} region, but I don't have ${metric} totals for it.`,
      meta: { type: "error", region, metric }
    };
  }

  // ------------------------------------
  // ALL-COUNTIES TOTALS
  // ------------------------------------
  if (totalFlag && !county) {
    if (metric === "population") {
      return {
        answer: `The total population across all counties is ${formatNumber(
          "population",
          totals.population
        )}.`,
        meta: { type: "total", metric }
      };
    }

    if (metric === "hispanic") {
      return {
        answer: `The total Hispanic population across all counties is ${formatNumber(
          "hispanic",
          totals.hispanic
        )}.`,
        meta: { type: "total", metric }
      };
    }

    if (metric === "projected") {
      return {
        answer: `The total projected Hispanic population across all counties is ${formatNumber(
          "projected",
          totals.projected
        )}.`,
        meta: { type: "total", metric }
      };
    }
  }

  // ------------------------------------
  // NO COUNTY FOUND (AFTER REGION LOGIC)
  // ------------------------------------
  if (!county) {
    return {
      answer: "I'm not sure which county you're asking about.",
      meta: { type: "error", reason: "no_county" }
    };
  }

  // ------------------------------------
  // PER-COUNTY DATA
  // ------------------------------------
  const row = findRow(county, metric);
  if (!row) {
    return {
      answer: `I couldn't find ${metric} data for ${county} County.`,
      meta: { type: "error", reason: "no_row", county, metric }
    };
  }

  if (metric === "population" && row.population != null) {
    return {
      answer: `${row.county} County has a population of ${formatNumber(
        "population",
        row.population
      )}.`,
      meta: { type: "county", county: row.county, metric }
    };
  }

  if (metric === "hispanic" && row.hispanicPopulation != null) {
    return {
      answer: `${row.county} County has a Hispanic population of ${formatNumber(
        "hispanic",
        row.hispanicPopulation
      )}.`,
      meta: { type: "county", county: row.county, metric }
    };
  }

  if (metric === "projected" && row.projectedPopulation != null) {
    return {
      answer: `The projected Hispanic population of ${row.county} County is ${formatNumber(
        "projected",
        row.projectedPopulation
      )}.`,
      meta: { type: "county", county: row.county, metric }
    };
  }

  return {
    answer: `Here is what I found for ${row.county} County: ${JSON.stringify(row)}.`,
    meta: { type: "county", county: row.county, metric }
  };
}


// ------------------------------------
// Routes
// ------------------------------------
app.get('/health', (_req, res) => {
  res.json({ ok: true, rows: rows.length, totals, regionTotals });
});

app.post("/ask", async (req, res) => {
  try {
    const question = req.body.question;
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Missing question" });
    }

    const tools = [
      {
        type: "function",
        function: {
          name: "compute_answer",
          description: "Compute exact answers using verified county data",
          parameters: {
            type: "object",
            properties: {
              question: {
                type: "string",
                description: "The user's original question"
              }
            },
            required: ["question"]
          }
        }
      }
    ];

    // 🔹 FIRST MODEL CALL (intent + optional tool call)
    const completion = await openaiKeyNODE.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content:
              "You are RANBot. If the user mentions a name that matches a known Alabama county, you MUST call the compute_answer tool. Do NOT answer from general knowledge."
        },
        { role: "user", content: question }
      ],
      tools,
      tool_choice: "auto"
    });

    // 🔹 ADD THIS BLOCK RIGHT HERE ⬇️
    const msg = completion.choices[0].message;

    // ---- Conversational response (no math needed) ----
    if (!msg.tool_calls) {
      return res.json({ answer: msg.content });
    }

    // 🔹 TOOL PATH CONTINUES BELOW
    const toolCall = msg.tool_calls[0];
    if (toolCall.function.name !== "compute_answer") {
      return res.status(500).json({ error: "Unexpected tool call" });
    }

    const args = JSON.parse(toolCall.function.arguments);

    // ---- Deterministic math ----
    const result = answerQuestion(args.question);

    res.json({
      answer: result.answer,
      meta: result.meta
    });


  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Server error" });
  }
});

/*
app.get('/ask', (req, res) => {
  const q = req.query.q;
  if (!q || typeof q !== "string") {
    return res.status(400).json({ error: "Please provide ?q=your+question" });
  }
  const result = answerQuestion(q);
  res.json({ answer: result.answer, text: result.answer, meta: result.meta });
});
*/

app.post('/reload', async (_req, res) => {
  try {
    await loadSheet();
    res.json({ ok: true, rows: rows.length, totals, regionTotals });
  } catch (e) {
    res.status(500).json({ error: e.message || "Reload failed" });
  }
});

// ------------------------------------
// Start Server
// ------------------------------------
const start = async () => {
  await loadSheet();
  app.listen(PORT, () => {
    console.log(`County bot running at http://localhost:${PORT}`);
    console.log("API KEY LOADED?", !!process.env.OPENAI_API_KEY);
  });
};
start();
