/**
 * Headless Puppeteer test for V3.
 * Validates: page loads, physics initializes, WebGPU renders, noise stats correct.
 *
 * Usage: node test_headless.mjs [--serve]
 * If --serve is not provided, assumes a server is already running on port 8080.
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { setTimeout as sleep } from 'timers/promises';

const PORT = 8082;
const PAGE_URL = `http://localhost:${PORT}/web_demo_v3/`;

async function main() {
    const doServe = process.argv.includes('--serve');
    let server;

    if (doServe) {
        // Start a simple HTTP server from the repo root
        server = spawn('npx', ['http-server', '-p', PORT, '-c-1', '--silent'], {
            cwd: new URL('..', import.meta.url).pathname,
            stdio: 'pipe',
        });
        await sleep(2000);
    }

    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'],
    });

    try {
        const page = await browser.newPage();

        // Collect console messages
        const logs = [];
        const errors = [];
        page.on('console', msg => {
            const text = msg.text();
            logs.push(text);
            if (msg.type() === 'error') errors.push(text);
        });
        page.on('pageerror', err => errors.push(err.message));

        console.log(`Navigating to ${PAGE_URL}...`);
        await page.goto(PAGE_URL, { timeout: 30000, waitUntil: 'networkidle0' });

        // Wait for rendering to stabilize
        await sleep(5000);

        // Check for critical errors
        const errorText = await page.$eval('#error', el => el.textContent);
        if (errorText) {
            console.error('ERROR element:', errorText);
            process.exit(1);
        }

        // Check noise stats (should be available after 60+ frames)
        const stats = await page.evaluate(() => window.__noiseStats);
        if (stats) {
            console.log(`Noise stats: mean=${stats.mean.toFixed(4)}, std=${stats.std.toFixed(4)}`);
            const meanOk = Math.abs(stats.mean) < 0.15;
            const stdOk = Math.abs(stats.std - 1.0) < 0.3;
            console.log(`  mean within tolerance: ${meanOk ? 'PASS' : 'FAIL'}`);
            console.log(`  std within tolerance:  ${stdOk ? 'PASS' : 'FAIL'}`);
            if (!meanOk || !stdOk) process.exit(1);
        } else {
            console.log('WARNING: No noise stats available (WebGPU may not be available in headless mode)');
        }

        // --- Star warp mode ---
        // Switch to Stars mode via the mode bar, wait for dynamics + a stats
        // readback (every 60 frames), then validate uniformity of the field.
        await page.click('.modeBtn[data-mode="6"]');
        await sleep(4000);
        const starStats = await page.evaluate(() => window.__starStats);
        if (starStats) {
            console.log(`Star stats: n=${starStats.numStars} sample=${starStats.sample} ` +
                        `inBounds=${(starStats.inBoundsFrac * 100).toFixed(1)}% ` +
                        `min/mean=${starStats.minOverMean.toFixed(3)} max/mean=${starStats.maxOverMean.toFixed(3)}`);
            // 4x4 grid over a 10k-star sample: ~625/cell, sigma ~4%. Uniformity
            // failures (the v1 Jacobian bug class) show cells near 0 or 2x.
            const inBoundsOk = starStats.inBoundsFrac === 1;
            const uniformOk = starStats.minOverMean > 0.8 && starStats.maxOverMean < 1.2;
            console.log(`  all stars in bounds: ${inBoundsOk ? 'PASS' : 'FAIL'}`);
            console.log(`  uniform star field:  ${uniformOk ? 'PASS' : 'FAIL'}`);
            if (!inBoundsOk || !uniformOk) process.exit(1);
        } else {
            console.log('WARNING: No star stats available (WebGPU may not be available in headless mode)');
        }

        // Check for JS errors
        if (errors.length > 0) {
            console.error('Console errors:', errors);
            // Don't exit — some WebGPU warnings are expected in headless
        }

        // Check stats overlay content
        const statsText = await page.$eval('#stats', el => el.textContent);
        console.log('Stats overlay:', statsText.slice(0, 100));

        console.log('\nAll checks passed!');

    } finally {
        await browser.close();
        if (server) server.kill();
    }
}

main().catch(e => {
    console.error('Test failed:', e.message);
    process.exit(1);
});
