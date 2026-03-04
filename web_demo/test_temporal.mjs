/**
 * Test temporal coherence: static areas should have stable noise values.
 * Read noise from a pixel in the static floor region across frames.
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
    '.mjs': 'application/javascript',
};

function startServer(port) {
    return new Promise((resolve) => {
        const server = createServer((req, res) => {
            const url = req.url.split('?')[0];
            const filePath = join(__dirname, url === '/' ? 'index.html' : url);
            if (!existsSync(filePath)) { res.writeHead(404); res.end('Not found'); return; }
            res.writeHead(200, { 'Content-Type': MIME[extname(filePath)] || 'application/octet-stream' });
            res.end(readFileSync(filePath));
        });
        server.listen(port, () => resolve(server));
    });
}

async function main() {
    const PORT = 8096;
    const server = await startServer(PORT);
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--enable-webgl', '--use-gl=angle', '--no-sandbox'],
    });
    const page = await browser.newPage();

    // Inject a function to read pixel color from the canvas
    await page.goto(`http://localhost:${PORT}`, { waitUntil: 'load', timeout: 30000 });

    // Wait for some frames
    await new Promise(r => setTimeout(r, 2000));

    // Read canvas pixel at a static floor region (bottom-center of canvas)
    // In noise mode (default), read the displayed pixel color to check temporal stability
    const samples = [];
    for (let i = 0; i < 5; i++) {
        const pixel = await page.evaluate(() => {
            const canvas = document.getElementById('canvas');
            const gl = canvas.getContext('webgl2');
            // Read from the canvas backbuffer
            // We need to read right after a frame is drawn
            const buf = new Uint8Array(4);
            gl.readPixels(256, 50, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
            return Array.from(buf);
        });
        samples.push(pixel);
        await new Promise(r => setTimeout(r, 200));
    }

    console.log('Pixel samples at (256, 50) — floor region:');
    for (const s of samples) {
        console.log(`  RGBA: [${s.join(', ')}]`);
    }

    // Check if static pixels are stable
    let stable = true;
    for (let i = 1; i < samples.length; i++) {
        for (let c = 0; c < 3; c++) {
            if (Math.abs(samples[i][c] - samples[0][c]) > 5) {
                stable = false;
            }
        }
    }
    console.log(stable ? 'STABLE: static region has consistent noise' : 'UNSTABLE: static region changes (may be OK if camera moves)');

    // Final stats
    const stats = await page.$eval('#stats', e => e.textContent);
    console.log('Stats:', stats);

    await browser.close();
    server.close();
}

main();
