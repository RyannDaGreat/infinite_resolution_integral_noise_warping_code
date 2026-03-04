/**
 * Headless test: launches the web demo in Puppeteer, captures console logs,
 * screenshots, and validates noise stats.
 *
 * Usage: node test_headless.mjs
 */

import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import puppeteer from 'puppeteer';

const __dirname = dirname(fileURLToPath(import.meta.url));

const MIME = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
};

function startServer(port) {
    return new Promise((resolve) => {
        const server = createServer((req, res) => {
            const url = req.url.split('?')[0];
            const filePath = join(__dirname, url === '/' ? 'index.html' : url);
            if (!existsSync(filePath)) {
                console.log('[404]', req.url);
                res.writeHead(404);
                res.end('Not found');
                return;
            }
            const ext = extname(filePath);
            const mime = MIME[ext] || 'application/octet-stream';
            res.writeHead(200, { 'Content-Type': mime });
            res.end(readFileSync(filePath));
        });
        server.listen(port, () => resolve(server));
    });
}

async function main() {
    const PORT = 8098;
    const server = await startServer(PORT);
    console.log(`Server on http://localhost:${PORT}`);

    const browser = await puppeteer.launch({
        headless: 'new',
        args: [
            '--enable-webgl',
            '--use-gl=angle',
            '--no-sandbox',
            '--disable-setuid-sandbox',
        ],
    });

    const page = await browser.newPage();
    page.on('console', (msg) => console.log(`[${msg.type()}] ${msg.text()}`));
    page.on('pageerror', (err) => console.error(`[PAGE ERR] ${err.message}`));
    page.on('requestfailed', (req) => console.log(`[REQ FAIL] ${req.url()} ${req.failure().errorText}`));

    console.log('Navigating...');
    await page.goto(`http://localhost:${PORT}`, { waitUntil: 'load', timeout: 30000 });

    console.log('Waiting 5s for frames...');
    await new Promise(r => setTimeout(r, 5000));

    await page.screenshot({ path: join(__dirname, 'test_screenshot.png') });
    console.log('Screenshot saved');

    const stats = await page.$eval('#stats', e => e.textContent);
    const error = await page.$eval('#error', e => e.textContent);
    console.log(`Stats: ${stats}`);
    if (error) console.error(`Error element: ${error}`);

    // Analyze
    let ok = true;
    if (error) ok = false;

    const meanMatch = stats.match(/mean:\s*(-?[\d.]+)/);
    const stdMatch = stats.match(/std:\s*([\d.]+)/);
    if (meanMatch && stdMatch) {
        const mean = parseFloat(meanMatch[1]);
        const std = parseFloat(stdMatch[1]);
        if (Math.abs(mean) > 0.15) { console.error(`FAIL mean=${mean}`); ok = false; }
        if (Math.abs(std - 1.0) > 0.3) { console.error(`FAIL std=${std}`); ok = false; }
    } else if (stats === 'Loading...') {
        console.error('FAIL: still loading — demo never started');
        ok = false;
    }

    console.log(ok ? '\nPASSED' : '\nFAILED');
    await browser.close();
    server.close();
    process.exit(ok ? 0 : 1);
}

main();
