/**
 * Test all display modes + capture screenshots for visual inspection.
 */

import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import puppeteer from 'puppeteer';

const __dirname = dirname(fileURLToPath(import.meta.url));

const MIME = {
    '.html': 'text/html', '.js': 'application/javascript',
    '.mjs': 'application/javascript', '.css': 'text/css',
};

function startServer(port) {
    return new Promise((resolve) => {
        const server = createServer((req, res) => {
            const url = req.url.split('?')[0];
            const filePath = join(__dirname, url === '/' ? 'index.html' : url);
            if (!existsSync(filePath)) { res.writeHead(404); res.end('Not found'); return; }
            const ext = extname(filePath);
            res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
            res.end(readFileSync(filePath));
        });
        server.listen(port, () => resolve(server));
    });
}

async function main() {
    const PORT = 8097;
    const server = await startServer(PORT);
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--enable-webgl', '--use-gl=angle', '--no-sandbox'],
    });

    const page = await browser.newPage();
    const errors = [];
    page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`http://localhost:${PORT}`, { waitUntil: 'load', timeout: 30000 });

    // Let it render a few frames first
    await new Promise(r => setTimeout(r, 2000));

    // Capture each display mode
    const modeNames = ['noise', 'color', 'motion', 'sidebyside', 'raw'];
    for (let mode = 0; mode < 5; mode++) {
        // Press digit key to change mode
        await page.keyboard.press(`Digit${mode + 1}`);
        await new Promise(r => setTimeout(r, 500));
        await page.screenshot({ path: join(__dirname, `test_mode_${mode}_${modeNames[mode]}.png`) });
        console.log(`Mode ${mode} (${modeNames[mode]}): screenshot saved`);
    }

    // Get final stats
    const stats = await page.$eval('#stats', e => e.textContent);
    console.log('Stats:', stats);

    if (errors.filter(e => !e.includes('favicon')).length > 0) {
        console.error('Errors:', errors);
    }

    await browser.close();
    server.close();
}

main();
