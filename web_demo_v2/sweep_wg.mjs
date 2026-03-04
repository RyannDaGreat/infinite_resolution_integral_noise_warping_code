/**
 * Workgroup size sweep for brownian shader.
 * Tests different sizes and reports per-phase GPU timings.
 *
 * Usage: node sweep_wg.mjs
 * Requires the renderer to support URL parameter: ?brownian_wg=N
 */
import puppeteer from 'puppeteer';

const SIZES = [32, 64, 128, 256, 512];
const FRAMES = 150;
const WARMUP = 50;
const URL_BASE = 'http://localhost:8082/web_demo_v2/';

function stats(arr) {
    const n = arr.length;
    const mean = arr.reduce((a, b) => a + b, 0) / n;
    const std = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
    return { mean, std, n };
}

async function testSize(size) {
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--no-sandbox', '--disable-gpu-sandbox'],
    });
    const page = await browser.newPage();
    await page.goto(`${URL_BASE}?brownian_wg=${size}`, { waitUntil: 'domcontentloaded' });

    const totalWait = Math.ceil((WARMUP + FRAMES) / 60) * 1000 + 3000;
    await new Promise(r => setTimeout(r, totalWait));

    const data = await page.evaluate((warmup, frames) => {
        const h = window.__gpuTimingHistory || [];
        return h.slice(warmup, warmup + frames);
    }, WARMUP, FRAMES);

    await browser.close();

    if (data.length === 0) return null;

    const result = {};
    for (const phase of Object.keys(data[0])) {
        result[phase] = stats(data.map(t => t[phase]));
    }
    return result;
}

async function main() {
    console.log('Brownian Workgroup Size Sweep');
    console.log('='.repeat(80));
    console.log(`${'WG'.padEnd(6)} ${'brownian'.padStart(12)} ${'backwardMap'.padStart(12)} ${'total'.padStart(12)} ${'noise ok'.padStart(10)}`);
    console.log('-'.repeat(60));

    for (const size of SIZES) {
        process.stdout.write(`${String(size).padEnd(6)} `);
        const r = await testSize(size);
        if (!r) {
            console.log('FAILED (no data)');
            continue;
        }
        const fmt = (s) => `${s.mean.toFixed(3)}±${s.std.toFixed(3)}`;
        console.log(
            `${fmt(r.brownian).padStart(12)} ` +
            `${fmt(r.backwardMap).padStart(12)} ` +
            `${fmt(r.total).padStart(12)}`
        );
    }
}

main().catch(console.error);
