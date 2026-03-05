/**
 * Quick headless test: verify fluid noise warp renders and noise stats are sane.
 * Not pure: launches browser, takes screenshots, writes files.
 *
 * Usage: node test_quick.mjs
 */

import puppeteer from 'puppeteer';

async function main() {
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--use-gl=angle', '--enable-webgl2-compute-context'],
    });
    const page = await browser.newPage();
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

    await page.goto('http://localhost:8765', { waitUntil: 'networkidle0' });
    // Wait for a few frames
    await page.waitForTimeout(3000);

    // Read stats from the overlay
    const statsText = await page.$eval('#stats', el => el.textContent);
    console.log('Stats:', statsText);

    // Parse mean and std
    const meanMatch = statsText.match(/mean:\s*([-\d.]+)/);
    const stdMatch = statsText.match(/std:\s*([-\d.]+)/);
    const fpsMatch = statsText.match(/FPS:\s*(\d+)/);

    if (meanMatch && stdMatch) {
        const mean = parseFloat(meanMatch[1]);
        const std = parseFloat(stdMatch[1]);
        console.log(`Noise: mean=${mean.toFixed(3)} std=${std.toFixed(3)}`);

        if (Math.abs(mean) > 0.5) console.warn('WARNING: mean too far from 0');
        if (std < 0.1 || std > 5.0) console.warn(`WARNING: std ${std} outside expected range [0.1, 5.0]`);
        else console.log('Noise stats OK');
    }

    if (fpsMatch) {
        const fps = parseInt(fpsMatch[1]);
        console.log(`FPS: ${fps}`);
        if (fps < 30) console.warn('WARNING: FPS below 30');
    }

    // Screenshot each mode
    for (let mode = 0; mode < 6; mode++) {
        await page.keyboard.press(`Digit${mode + 1}`);
        await page.waitForTimeout(200);
        await page.screenshot({ path: `test_mode_${mode}.png` });
        console.log(`Screenshot: mode ${mode} saved`);
    }

    await browser.close();
    console.log('Test complete');
}

main().catch(e => { console.error(e); process.exit(1); });
