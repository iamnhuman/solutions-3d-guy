import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = new URL('../', import.meta.url);
const source = (path, encoding) => readFile(new URL(path, root), encoding);
const [template, css, model, threeLicense] = await Promise.all([
  source('src/page.html', 'utf8'),
  source('src/style.css', 'utf8'),
  source('assets/character.glb'),
  source('node_modules/three/LICENSE', 'utf8'),
]);
if (model.toString('ascii', 0, 4) !== 'glTF' || model.readUInt32LE(8) !== model.length) {
  throw new Error('assets/character.glb is not a complete GLB file');
}

const result = await build({
  absWorkingDir: fileURLToPath(root),
  entryPoints: ['src/main.js'],
  bundle: true,
  format: 'iife',
  target: ['es2021'],
  platform: 'browser',
  minify: true,
  legalComments: 'inline',
  write: false,
});

const js = result.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const html = template
  .replace('<!-- STUDIO_STYLE -->', () => `<style>\n${css}</style>`)
  .replace('<!-- CHARACTER_DATA -->', () =>
    `<script id="character-data" type="application/octet-stream">${model.toString('base64')}</script>`)
  .replace('<!-- STUDIO_SCRIPT -->', () => `<!-- Three.js\n${threeLicense}-->\n<script>\n${js}</script>`);
await writeFile(new URL('index.html', root), html);
console.log(`Built index.html (${(Buffer.byteLength(html) / 1024 / 1024).toFixed(2)} MiB, self-contained)`);
