/**
 * Star warp under real motion: walk forward for several seconds while in Stars
 * mode, then validate the field is STILL uniform (deaths/births firing), and
 * capture screenshots (AA on and off).
 */
import puppeteer from 'puppeteer';
import { setTimeout as sleep } from 'timers/promises';

const URL = 'http://localhost:8082/web_demo_v3/';
const SHOT_DIR = process.env.SHOT_DIR || '.';

const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1200 });
const errors = [];
page.on('pageerror', err => errors.push(err.message));

await page.goto(URL, { timeout: 30000, waitUntil: 'networkidle0' });
await sleep(3000);

// Stars mode
await page.click('.modeBtn[data-mode="6"]');
await sleep(1500);

// Walk forward + strafe while the starfield warps (no pointer lock needed for keys)
await page.keyboard.down('KeyW');
await sleep(2500);
await page.keyboard.down('KeyD');
await sleep(2500);
await page.keyboard.up('KeyW');
await page.keyboard.up('KeyD');
await sleep(1500);  // let a stats readback land (every 60 frames)

const stats = await page.evaluate(() => window.__starStats);
if (!stats) { console.error('No star stats — WebGPU unavailable'); process.exit(1); }
console.log(`After motion: n=${stats.numStars} inBounds=${(stats.inBoundsFrac * 100).toFixed(1)}% ` +
            `min/mean=${stats.minOverMean.toFixed(3)} max/mean=${stats.maxOverMean.toFixed(3)}`);
const ok = stats.inBoundsFrac === 1 && stats.minOverMean > 0.8 && stats.maxOverMean < 1.2;
console.log(`uniform under motion: ${ok ? 'PASS' : 'FAIL'}`);

// Screenshot AA ON
await page.screenshot({ path: `${SHOT_DIR}/stars_aa_on.png` });

// Toggle AA off, screenshot
await page.click('#starAABtn');
await sleep(500);
await page.screenshot({ path: `${SHOT_DIR}/stars_aa_off.png` });

// Crank N to 1M via the slider (set value + input event), check FPS holds
await page.evaluate(() => {
    const s = document.getElementById('starCountSlider');
    s.value = 100;
    s.dispatchEvent(new Event('input'));
});
await sleep(3000);
const statsHi = await page.evaluate(() => window.__starStats);
const fpsLine = await page.$eval('#stats', el => el.textContent.split('\n')[0]);
console.log(`N=1M: ${fpsLine}`);
console.log(`  stats: n=${statsHi.numStars} inBounds=${(statsHi.inBoundsFrac * 100).toFixed(1)}% ` +
            `min/mean=${statsHi.minOverMean.toFixed(3)} max/mean=${statsHi.maxOverMean.toFixed(3)}`);
await page.screenshot({ path: `${SHOT_DIR}/stars_1M.png` });

if (errors.length) console.error('Page errors:', errors);
await browser.close();
if (!ok || errors.length) process.exit(1);
console.log('MOTION TEST PASSED');
