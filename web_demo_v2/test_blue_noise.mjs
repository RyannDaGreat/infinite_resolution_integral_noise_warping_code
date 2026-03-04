/**
 * Quick test: enable blue noise and check for GPU errors.
 */
import puppeteer from 'puppeteer';

const URL = 'http://localhost:8082/web_demo_v2/';

async function main() {
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--no-sandbox', '--disable-gpu-sandbox'],
    });
    const page = await browser.newPage();

    const errors = [];
    page.on('console', msg => {
        if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', err => errors.push(err.message));

    await page.goto(URL, { waitUntil: 'domcontentloaded' });

    // Wait for initial render
    await new Promise(r => setTimeout(r, 3000));

    // Enable blue noise via button click
    await page.click('#blueNoiseBtn');
    console.log('Blue noise enabled');

    // Wait for a few frames with blue noise active
    await new Promise(r => setTimeout(r, 5000));

    // Check noise stats
    const stats = await page.evaluate(() => window.__noiseStats);
    console.log('Noise stats:', stats);

    // Check for errors
    if (errors.length > 0) {
        console.log('ERRORS:');
        errors.forEach(e => console.log('  ', e));
    } else {
        console.log('No errors');
    }

    // Get FPS
    const fps = await page.evaluate(() => window.__stats?.fps);
    console.log('FPS:', fps);

    await browser.close();
    process.exit(errors.length > 0 ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
