/**
 * screenshot.js — Puppeteer renderer para ig-posts.html
 *
 * Uso:
 *   node screenshot.js              → exporta todos los posts/stories
 *   node screenshot.js post-01      → exporta solo ese elemento
 *
 * Requisitos:
 *   npm install puppeteer
 *
 * Output: social/exports/{id}.png
 */

const puppeteer = require('puppeteer');
const path      = require('path');
const fs        = require('fs');

const HTML_PATH    = path.resolve(__dirname, 'ig-posts.html');
const EXPORTS_DIR  = path.resolve(__dirname, 'exports');
const SCALE_FACTOR = 2; // @2x → posts 540px CSS = 1080px reales

async function main() {
  const targetId = process.argv[2] || null;

  if (!fs.existsSync(EXPORTS_DIR)) fs.mkdirSync(EXPORTS_DIR, { recursive: true });

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();

  // deviceScaleFactor: 2 → duplica la resolución de cada pixel CSS
  await page.setViewport({ width: 1600, height: 4000, deviceScaleFactor: SCALE_FACTOR });

  await page.goto(`file://${HTML_PATH}`, {
    waitUntil: 'networkidle0', // espera fuentes e imágenes de red
    timeout: 30000,
  });

  // Espera extra para Google Fonts (carga asíncrona)
  await new Promise(r => setTimeout(r, 1500));

  // Obtener todos los .post y .story del HTML
  const elements = await page.$$('.post, .story');

  let exported = 0;

  for (const el of elements) {
    const id = await el.evaluate(node => node.id);
    if (!id) continue;
    if (targetId && id !== targetId) continue;

    const outPath = path.join(EXPORTS_DIR, `${id}.png`);
    await el.screenshot({ path: outPath, type: 'png' });

    const box = await el.boundingBox();
    const w   = Math.round(box.width  * SCALE_FACTOR);
    const h   = Math.round(box.height * SCALE_FACTOR);
    console.log(`✓ ${id}.png  [${w}×${h}px]  →  ${outPath}`);
    exported++;
  }

  await browser.close();

  if (exported === 0) {
    console.error(`✗ No se encontró el elemento: ${targetId}`);
    process.exit(1);
  }

  console.log(`\n${exported} archivo(s) exportado(s) → ${EXPORTS_DIR}`);
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
