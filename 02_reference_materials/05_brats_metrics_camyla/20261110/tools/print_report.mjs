import path from 'node:path';
import os from 'node:os';
import { pathToFileURL } from 'node:url';

const modulePath = path.join(os.homedir(), '.local/share/codex-doc-tools/node/node_modules/playwright/index.js');
const { chromium } = (await import(pathToFileURL(modulePath).href)).default;
const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
});
try {
  const page = await browser.newPage();
  await page.route('**/*', route => {
    const protocol = new URL(route.request().url()).protocol;
    return ['file:', 'data:'].includes(protocol) ? route.continue() : route.abort();
  });
  await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
  await page.evaluate(async () => {
    await document.fonts.ready;
    if (![...document.fonts].every(font => font.status === 'loaded')) {
      throw new Error('Report font loading failed');
    }
    const widths = {
      2: [30, 70], 3: [18, 41, 41], 4: [16, 28, 28, 28],
    };
    for (const table of document.querySelectorAll('table')) {
      const row = table.rows[0];
      [...row.cells].forEach((cell, index) => {
        cell.style.width = widths[row.cells.length][index] + '%';
      });
    }
    if (document.querySelectorAll('math[display="block"]').length !== 3) {
      throw new Error('Expected three professionally rendered equations');
    }
  });
  await page.pdf({
    path: path.resolve(process.argv[3]), preferCSSPageSize: true, printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font:9px Arial;width:100%;text-align:center;color:black"><span class="pageNumber"></span></div>',
  });
  console.log('Printed PDF using scoped embedded fonts and native MathML');
} finally {
  await browser.close();
}
