const fs = require("fs");
const fetch = require("node-fetch");
const path = require("path");

// List of model files to fetch
const files = [
  {
    url: "https://huggingface.co/DS-Capstone/personalityclusterpredictionmodel/resolve/main/random_forest_model.joblib",
    dest: path.join("models", "random_forest_model.joblib"),
  },
  {
    url: "https://huggingface.co/DS-Capstone/personalityclusterpredictionmodel/resolve/main/scaler.joblib",
    dest: path.join("models", "scaler.joblib"),
  }
];

// Utility: Download file from URL
async function downloadFile(url, dest) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download ${url}`);
  const fileStream = fs.createWriteStream(dest);
  return new Promise((resolve, reject) => {
    res.body.pipe(fileStream);
    res.body.on("error", reject);
    fileStream.on("finish", resolve);
  });
}

(async () => {
  if (!fs.existsSync("models")) {
    fs.mkdirSync("models");
  }

  for (const file of files) {
    if (fs.existsSync(file.dest)) {
      console.log(`Already exists: ${file.dest}`);
    } else {
      console.log(`Downloading: ${file.url}`);
      try {
        await downloadFile(file.url, file.dest);
        console.log(`Saved: ${file.dest}`);
      } catch (err) {
        console.error(`Failed to download ${file.url}:`, err.message);
      }
    }
  }
})();
