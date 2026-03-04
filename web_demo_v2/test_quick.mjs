/**
 * Quick smoke test: launch Chrome, load V2 page, check for errors.
 * Reports shader compilation or WebGPU init failures.
 */
import puppeteer from 'puppeteer';

const URL = 'http://localhost:8082/web_demo_v2/';

async function main() {
    const browser = await puppeteer.launch({
        headless: true,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-webgpu-adapter=swiftshader',
            '--no-sandbox',
        ],
    });

    const page = await browser.newPage();

    // Collect console errors
    const errors = [];
    page.on('console', msg => {
        const text = msg.text();
        if (msg.type() === 'error') errors.push(text);
        console.log(`[${msg.type()}] ${text}`);
    });
    page.on('pageerror', err => {
        errors.push(err.message);
        console.error('[pageerror]', err.message);
    });

    console.log('Navigating to', URL);
    await page.goto(URL, { waitUntil: 'domcontentloaded' });

    // Wait for init
    await new Promise(r => setTimeout(r, 5000));

    // Check stats element
    const stats = await page.$eval('#stats', el => el.textContent);
    console.log('Stats:', stats);

    // Check error element
    const error = await page.$eval('#error', el => el.textContent);
    if (error) console.error('Error element:', error);

    // Check if frames are being rendered
    const hasGpuTimings = await page.evaluate(() => typeof window.__gpuTimingSample !== 'undefined');
    console.log('GPU timings available:', hasGpuTimings);

    // Check noise stats
    const noiseStats = await page.evaluate(() => window.__noiseStats);
    console.log('Noise stats:', noiseStats);

    // Take screenshot
    await page.screenshot({ path: 'web_demo_v2/test_screenshot.png' });
    console.log('Screenshot saved');

    if (errors.length > 0) {
        console.error('\n=== ERRORS ===');
        errors.forEach(e => console.error(e));
        process.exit(1);
    }

    console.log('\n=== PASS ===');
    await browser.close();
}

main().catch(e => { console.error(e); process.exit(1); });
