import express from "express";
import fetch from "node-fetch";
import multer from "multer";
import pdfParse from "pdf-parse";
import path from "path";
import sre from "speech-rule-engine";
import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { CHTML } from "mathjax-full/js/output/chtml.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { HTMLMathItem } from "mathjax-full/js/handlers/html/HTMLMathItem.js";
import { SerializedMmlVisitor } from "mathjax-full/js/core/MmlTree/SerializedMmlVisitor.js";

await sre.setupEngine({ locale: "en", domain: "mathspeak", style: "default" });
await sre.engineReady();

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const texInput = new TeX({ packages: ["base", "ams", "newcommand", "noundefined"] });
const chtmlOutput = new CHTML({});
const mathDocument = mathjax.document("", { InputJax: texInput, OutputJax: chtmlOutput });
const mmlVisitor = new SerializedMmlVisitor();

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(express.static("."));

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const DEMO_MODE = process.env.DEMO_MODE === "true" || !OPENAI_API_KEY;
const MAX_PAGES = 15;
const APPROX_WORDS_PER_PAGE = 450;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 25 * 1024 * 1024
  }
});

function cleanupSpeechEngineOutput(text) {
  return text
    .replace(/\bupper\s+([A-Z])\b/g, "$1")
    .replace(/\blower\s+([a-z])\b/g, "$1")
    .replace(/StartFraction\s+/g, "")
    .replace(/\s+Over\s+/g, " over ")
    .replace(/\s+EndFraction/g, "")
    .replace(/StartRoot\s+/g, "square root of ")
    .replace(/\s+EndRoot/g, "")
    .replace(/Subscript\s+([A-Za-z])\s+0/g, "$1 nought")
    .replace(/\bepsilon\s+0\b/g, "epsilon nought")
    .replace(/\bnabla\b/g, "del")
    .replace(/\bnegative\b/g, "minus")
    .replace(/\s+/g, " ")
    .trim();
}

function latexToSpeech(latex) {
  try {
    const item = new HTMLMathItem(latex.trim(), texInput, true);
    item.compile(mathDocument);
    const mathml = mmlVisitor.visitTree(item.root);
    return cleanupSpeechEngineOutput(sre.toSpeech(mathml));
  } catch (err) {
    return null;
  }
}

function replaceDelimitedLatex(text) {
  const patterns = [
    /\$\$([\s\S]+?)\$\$/g,
    /\\\[([\s\S]+?)\\\]/g,
    /\\\(([\s\S]+?)\\\)/g
  ];

  let result = text;

  for (const pattern of patterns) {
    result = result.replace(pattern, (match, latex) => latexToSpeech(latex) || match);
  }

  return result;
}

function replaceRawLatexEquations(text) {
  const command = "(?:frac|sqrt|hbar|nabla|partial|alpha|beta|gamma|delta|epsilon|varepsilon|mu|pi|theta|lambda|omega|int|sum|lim)";
  const equationPattern = new RegExp(`([A-Za-z0-9_{}\\\\^+\\-*/=(),.\\s]{0,80}\\\\${command}\\b[A-Za-z0-9_{}\\\\^+\\-*/=(),.\\s]{0,180})`, "g");

  return text.replace(equationPattern, (match) => {
    const trimmed = match.trim();
    if (!/[=^_]|\\frac|\\sqrt|\\int|\\sum/.test(trimmed)) {
      return match;
    }

    return latexToSpeech(trimmed) || match;
  });
}

function repairPdfMathArtifacts(text) {
  return text
    .replace(/\u0000/g, " minus ")
    .replace(/([A-Za-z])↵([A-Za-z])/g, "$1ff$2")
    .replace(/⇡/g, " pi ")
    .replace(/✏/g, " epsilon ")
    .replace(/↵\s*⌘/g, " alpha is defined as ")
    .replace(/↵\s*is\b/g, " alpha is")
    .replace(/↵/g, " alpha ")
    .replace(/⇣/g, " (")
    .replace(/⌘/g, ") ")
    .replace(/⇠\s*=/g, " approximately equals ")
    .replace(/⇠/g, " approximately ")
    .replace(/⇥/g, " times ")
    .replace(//g, " less than or equal to ")
    .replace(/≥/g, " greater than or equal to ")
    .replace(/≤/g, " less than or equal to ")
    .replace(/≠/g, " not equal to ")
    .replace(/≈/g, " approximately equals ")
    .replace(/⌦/g, " expectation value of ")
    .replace(/µ/g, " mu ")
    .replace(/μ/g, " mu ")
    .replace(/ˆ\s*([A-Za-z])/g, "$1 hat")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeMathForSpeech(text) {
  let spoken = replaceRawLatexEquations(replaceDelimitedLatex(repairPdfMathArtifacts(text)));

  const greekLetters = {
    alpha: "alpha",
    beta: "beta",
    gamma: "gamma",
    delta: "delta",
    epsilon: "epsilon",
    varepsilon: "epsilon",
    zeta: "zeta",
    eta: "eta",
    theta: "theta",
    vartheta: "theta",
    lambda: "lambda",
    mu: "mu",
    nu: "nu",
    xi: "xi",
    pi: "pi",
    rho: "rho",
    sigma: "sigma",
    tau: "tau",
    phi: "phi",
    varphi: "phi",
    chi: "chi",
    psi: "psi",
    omega: "omega",
    Gamma: "capital gamma",
    Delta: "capital delta",
    Theta: "capital theta",
    Lambda: "capital lambda",
    Xi: "capital xi",
    Pi: "capital pi",
    Sigma: "capital sigma",
    Phi: "capital phi",
    Psi: "capital psi",
    Omega: "capital omega"
  };

  const latexCommands = {
    hbar: "h bar",
    nabla: "del",
    partial: "partial",
    infty: "infinity",
    infinity: "infinity",
    cdot: "times",
    times: "times",
    div: "divided by",
    pm: "plus or minus",
    mp: "minus or plus",
    leq: "less than or equal to",
    geq: "greater than or equal to",
    neq: "not equal to",
    approx: "approximately equal to",
    sim: "similar to",
    proportional: "proportional to",
    propto: "proportional to",
    int: "integral of",
    sum: "sum of",
    prod: "product of",
    lim: "limit of",
    sin: "sine",
    cos: "cosine",
    tan: "tangent",
    exp: "exponential",
    ln: "natural log",
    log: "log"
  };

  const superscripts = {
    "⁰": " to the power of zero",
    "¹": " to the power of one",
    "²": " squared",
    "³": " cubed",
    "⁴": " to the power of four",
    "⁵": " to the power of five",
    "⁶": " to the power of six",
    "⁷": " to the power of seven",
    "⁸": " to the power of eight",
    "⁹": " to the power of nine",
    "⁻": " minus "
  };

  const subscripts = {
    "₀": " nought",
    "₁": " sub one",
    "₂": " sub two",
    "₃": " sub three",
    "₄": " sub four",
    "₅": " sub five",
    "₆": " sub six",
    "₇": " sub seven",
    "₈": " sub eight",
    "₉": " sub nine"
  };

  function cleanMathGroup(value) {
    return normalizeMathForSpeech(value)
      .replace(/\s+/g, " ")
      .trim();
  }

  function replaceLatexFractions(value) {
    let previous;
    let current = value;

    do {
      previous = current;
      current = current.replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, (_match, numerator, denominator) => {
        return `${cleanMathGroup(numerator)} over ${cleanMathGroup(denominator)}`;
      });
    } while (current !== previous);

    return current;
  }

  spoken = replaceLatexFractions(spoken);

  spoken = spoken.replace(/\\sqrt\s*\{([^{}]+)\}/g, (_match, value) => {
    return `square root of ${cleanMathGroup(value)}`;
  });

  spoken = spoken.replace(/\\(?:left|right|big|Big|bigg|Bigg)\b/g, "");
  spoken = spoken.replace(/\\(?:mathbf|mathrm|mathit|text|vec|hat|bar|tilde)\s*\{([^{}]+)\}/g, "$1");
  spoken = spoken.replace(/\\dot\s*\{([^{}]+)\}/g, "$1 dot");
  spoken = spoken.replace(/\\ddot\s*\{([^{}]+)\}/g, "$1 double dot");

  for (const [command, name] of Object.entries({ ...greekLetters, ...latexCommands })) {
    spoken = spoken.replace(new RegExp(`\\\\${command}\\b`, "g"), name);
  }

const symbolReplacements = [
    [/ℏ|ħ|~(?=\s*(?:\^?2|²|2|c\b))/g, ", h bar, "], // Added commas
    [/∇\s*(?:\^?2|²|2)/g, ", del squared, "],
    [/▽\s*(?:\^?2|²|2)/g, ", del squared, "],
    [/∇|▽/g, ", del, "],
    [/∂/g, ", partial, "],
    [/∞/g, ", infinity, "],
    [/∫/g, ", the integral of, "], // Added 'the' and commas
    [/∑/g, ", the sum of, "],
    [/∏/g, ", the product of, "],
    [/√\s*([^\s+\-=,.;]+)/g, ", the square root of $1, "],
    [/π/g, " pi "],
    [/ε/g, " epsilon "],
    [/λ/g, " lambda "],
    [/θ/g, " theta "],
    [/φ/g, " phi "],
    [/ω/g, " omega "],
    [/∝/g, ", is proportional to, "],
    [/×|·|⋅/g, ", times, "],
    [/÷/g, ", divided by, "],
    [/→|⇒/g, ", implies, "],
    [/←/g, ", is implied by, "],
    [/↔/g, ", if and only if, "],
    [/\|([^|]+)\|/g, ", the absolute value of $1, "],
    [/\s*=\s*/g, ", equals, "], // Added commas around equals
    [/\s*\+\s*/g, ", plus, "],   // Added commas around plus
    [/\s+[−–—-]\s+/g, ", minus, "], // Added commas around minus
    [/\s*\/\s*/g, ", over, "]    // Added commas around over
  ];
  for (const [pattern, replacement] of symbolReplacements) {
    spoken = spoken.replace(pattern, replacement);
  }

  spoken = spoken.replace(/\^\s*\{([^{}]+)\}/g, " to the power of $1");
  spoken = spoken.replace(/\^\s*2\b/g, " squared");
  spoken = spoken.replace(/\^\s*3\b/g, " cubed");
  spoken = spoken.replace(/\^\s*([A-Za-z0-9]+)/g, " to the power of $1");

  for (const [symbol, phrase] of Object.entries(superscripts)) {
    spoken = spoken.replaceAll(symbol, phrase);
  }

  spoken = spoken.replace(/_\s*\{([^{}]+)\}/g, (_match, value) => {
    const cleaned = cleanMathGroup(value);
    return cleaned === "0" ? " nought" : ` sub ${cleaned}`;
  });
  spoken = spoken.replace(/_\s*0\b/g, " nought");
  spoken = spoken.replace(/_\s*([A-Za-z0-9]+)/g, " sub $1");

  for (const [symbol, phrase] of Object.entries(subscripts)) {
    spoken = spoken.replaceAll(symbol, phrase);
  }

  spoken = spoken.replace(/\b(E|H|V|T|L|S|r|n|l|m|a|x|y|z)\s+1\b/g, "$1 sub one");
  spoken = spoken.replace(/\b(e|p|r|v|c|m|n|l|a|x|y|z|E|T|H|V|L|S)\s+2\b/g, "$1 squared");
  spoken = spoken.replace(/\b(e|p|r|v|c|m|n|l|a|x|y|z|E|T|H|V|L|S)\s+3\b/g, "$1 cubed");
  spoken = spoken.replace(/\b(e|p|r|v|c|m|n|l|a|x|y|z|E|T|H|V|L|S)\s+4\b/g, "$1 to the fourth");
  spoken = spoken.replace(/\b(E|H|V|T|L|S|r|n|l|m|a|x|y|z)\s+0\b/g, "$1 nought");
  spoken = spoken.replace(/\b(h bar)\s+2\b/g, "$1 squared");
  spoken = spoken.replace(/\b(alpha|epsilon|mu|pi)\s+2\b/g, "$1 squared");
  spoken = spoken.replace(/\b1\s+(r|x|y|z|n|a)\s+squared\b/g, "one over $1 squared");
  spoken = spoken.replace(/\b1\s+(r|x|y|z|n|a)\b/g, "one over $1");
  spoken = spoken.replace(/\b4\s+pi\s+epsilon\s+0\b/g, "four pi epsilon nought");
  spoken = spoken.replace(/\bepsilon\s+0\b/g, "epsilon nought");
  spoken = spoken.replace(/\bh bar\s+c\b/g, "h bar c");
  spoken = spoken.replace(/\b2\s*m\s+r\s+squared\b/g, "two m times del squared");
  spoken = spoken.replace(/\b2m\s+r\s+squared\b/g, "two m times del squared");
  spoken = spoken.replace(/\bminus\s+h bar\s+squared\s+two m times del squared\b/g, "minus h bar squared over two m times del squared");
  spoken = spoken.replace(/\be\s+squared\s+four pi epsilon nought\s+one over r\b/g, "e squared over four pi epsilon nought r");
  spoken = spoken.replace(/\balpha\s+is defined as\s+e\s+squared\s+four pi epsilon nought\s+h bar c\b/g, "alpha is defined as e squared over four pi epsilon nought h bar c");
  spoken = spoken.replace(/\b(approximately equals|equals)\s+1\s+(\d+(?:\.\d+)?)/g, "$1 one over $2");
  spoken = spoken.replace(/\bcorrectionand\b/g, "correction and");
  spoken = spoken.replace(/\bthespin\b/g, "the spin");
  spoken = spoken.replace(/\barelativistic\b/g, "a relativistic");
  spoken = spoken.replace(/[{}\[\]]/g, " ");
  spoken = spoken.replace(/\\[A-Za-z]+/g, " ");
  spoken = spoken.replace(/\s+/g, " ").trim();

  return spoken;
}

function estimateTextPages(text) {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(words / APPROX_WORDS_PER_PAGE));
}

async function extractTextFromUpload(file) {
  const extension = path.extname(file.originalname).toLowerCase();
  const mimeType = file.mimetype;

  if (extension === ".pdf" || mimeType === "application/pdf") {
    const parsed = await pdfParse(file.buffer);

    if (parsed.numpages > MAX_PAGES) {
      throw new Error(`This PDF has ${parsed.numpages} pages. Please upload ${MAX_PAGES} pages or fewer.`);
    }

    if (!parsed.text || !parsed.text.trim()) {
      throw new Error("I could not find readable text in this PDF. It may be scanned images instead of selectable text.");
    }

    return {
      text: normalizeMathForSpeech(parsed.text),
      pageCount: parsed.numpages,
      fileType: "PDF"
    };
  }

  if ([".txt", ".md"].includes(extension) || mimeType.startsWith("text/")) {
    const text = normalizeMathForSpeech(file.buffer.toString("utf8"));
    const pageCount = estimateTextPages(text);

    if (pageCount > MAX_PAGES) {
      throw new Error(`This text file is about ${pageCount} pages. Please upload about ${MAX_PAGES} pages or fewer.`);
    }

    if (!text) {
      throw new Error("This file looks empty.");
    }

    return {
      text,
      pageCount,
      fileType: extension === ".md" ? "Markdown" : "Text"
    };
  }

  throw new Error("Please upload a PDF, TXT, or MD file.");
}

function createDemoRecap(text) {
  const cleanText = normalizeMathForSpeech(text).replace(/\s+/g, " ").trim();
  const sentences = cleanText
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  const keyPoints = sentences.slice(0, 4);
  const mainIdea = keyPoints[0] || cleanText.slice(0, 220);

  return `Before your next class, here's what you need to remember from this lecture. The main idea is: ${mainIdea}

In simpler terms, focus on what the equation is saying, not just the symbols. The symbols describe how the important quantities depend on each other. ${keyPoints.slice(1).join(" ")}

If you only remember one thing, remember the central relationship or rule from the lecture, and try to explain it back to yourself in one sentence. That is the part most likely to help you when you come back to study later.`;
}

function makeSpeechPayload(summary) {
  return {
    summary,
    speechText: normalizeMathForSpeech(summary)
  };
}

app.post("/upload", upload.single("lectureFile"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Please choose a file to upload." });
    }

    const result = await extractTextFromUpload(req.file);

    res.json({
      fileName: req.file.originalname,
      fileType: result.fileType,
      pageCount: result.pageCount,
      text: result.text
    });
  } catch (err) {
    res.status(400).json({ error: err.message || "Could not read this upload." });
  }
});

app.post("/summarize", async (req, res) => {
  const { text } = req.body;

  if (!text || !text.trim()) {
    return res.status(400).json({
      error: "Please paste lecture notes or upload a file before generating a recap."
    });
  }

  if (DEMO_MODE) {
    return res.json({
      ...makeSpeechPayload(createDemoRecap(text)),
      demo: true
    });
  }

  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: "You are an expert tutor creating a 3-minute spoken audio recap for a university student walking between classes.\n\nCRITICAL RULES FOR EQUATIONS:\n- Do NOT read equations symbol-by-symbol (e.g., never say 'integral from zero to infinity of x squared dx').\n- Instead, summarize the CONCEPT the formula represents.\n- Focus on proportionality and physical/mathematical meaning (e.g., 'The equation shows that the energy scales with the square of the amplitude...').\n- If a famous equation appears, call it by its name (e.g., 'Schrödinger\\'s equation', 'Gauss\\'s Law')."
          },
          {
            role: "user",
            content: `Make it conversational, clear, and useful. Start with: "Before your next class, here's what you need to remember..." \n\nLecture notes:\n${text}`
          }
        ]
      })
    });

    const data = await response.json();

    if (!response.ok) {
      console.error(data);
      return res.status(response.status).json({
        error: data.error?.message || "OpenAI could not summarize the text."
      });
    }

    const summary = data.choices?.[0]?.message?.content;

    if (!summary) {
      console.error(data);
      return res.status(500).json({
        error: "OpenAI returned a response, but no summary text was found."
      });
    }

    res.json(makeSpeechPayload(summary));
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error summarizing text" });
  }
});

app.post("/tts", async (req, res) => {
  const { text } = req.body;

  if (!text || !text.trim()) {
    return res.status(400).json({
      error: "Please generate a recap before creating audio."
    });
  }

  const speechText = normalizeMathForSpeech(text);

  if (DEMO_MODE) {
    return res.status(503).json({
      error: "Demo mode uses your browser's built-in speech instead of OpenAI audio.",
      demo: true
    });
  }

  try {
    const response = await fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini-tts",
        voice: "alloy",
        input: speechText
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(errorText);
      return res.status(response.status).json({
        error: "OpenAI could not generate audio."
      });
    }

    const audioBuffer = await response.arrayBuffer();

    res.set({
      "Content-Type": "audio/mpeg"
    });

    res.send(Buffer.from(audioBuffer));
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error generating audio" });
  }
});

app.listen(3000, () => {
  const mode = DEMO_MODE ? "demo mode" : "OpenAI mode";
  console.log(`Server running on http://localhost:3000 (${mode})`);
});