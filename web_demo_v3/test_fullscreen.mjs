/**
 * Headless fullscreen + anaglyph-options test. Validates: [F] fullscreen makes
 * the render resolution track the (headless) screen size — including a
 * NON-SQUARE renderer with corrected projection aspect — stars mode stays
 * uniform there, the depth-shift/color/swap controls render without GPU
 * issues, and exiting restores the preset resolution.
 *
 * Usage: node test_fullscreen.mjs   (server on :8082)
 */
import puppeteer from 'puppeteer';
import { setTimeout as sleep } from 'timers/promises';

const PAGE_URL = 'http://localhost:8082/web_demo_v3/';
const VIEW_W = 1600, VIEW_H = 1000;   // headless "monitor"

const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox',
           `--window-size=${VIEW_W},${VIEW_H}`],
});
const page = await browser.newPage();
await page.setViewport({ width: VIEW_W, height: VIEW_H });
const issues = [];
page.on('console', m => {
    const t = m.text();
    if (m.type() === 'error' && !t.includes('404')) issues.push(t.slice(0, 200));
    if (m.type() === 'warning' && /invalid|error while|exceeds the maximum|validating/i.test(t)) {
        issues.push(t.slice(0, 200));
    }
});
page.on('pageerror', e => issues.push(String(e).slice(0, 200)));

await page.goto(PAGE_URL, { waitUntil: 'networkidle0', timeout: 30000 });
await sleep(2500);
await page.click('[data-mode="6"]');
await sleep(1200);

const dims = () => page.evaluate(() => {
    const c = document.getElementById('canvas');
    return { w: c.width, h: c.height, fs: !!document.fullscreenElement };
});
const stats = () => page.evaluate(() => window.__starStats);
let failures = 0;
const expect = (label, cond) => {
    if (!cond) failures++;
    console.log(`${label}: ${cond ? 'PASS' : 'FAIL'}`);
};

expect('windowed 1024', (await dims()).w === 1024);
// headless Chrome's screen object is independent of the viewport — assert
// against whatever the browser reports as the monitor (that's the contract:
// fullscreen tracks screen.width/height, re-read at entry).
const screenDims = await page.evaluate(() => ({ w: window.screen.width, h: window.screen.height }));
await page.keyboard.press('KeyF');
await sleep(3500);                                   // renderer recreation
const fsDims = await dims();
console.log('fullscreen dims:', JSON.stringify(fsDims), `(screen ${screenDims.w}x${screenDims.h})`);
expect('fullscreen active', fsDims.fs);
expect('render res = monitor res', fsDims.w === screenDims.w && fsDims.h === screenDims.h);
await sleep(2400);
const s1 = await stats();
expect('non-square stars uniform', s1.inBoundsFrac === 1 && s1.minOverMean > 0.7 && s1.maxOverMean < 1.3);

// stereo red-blue with depth shift + custom tints, in fullscreen.
// NOTE: the fullscreened canvas covers the toolbar, so coordinate clicks hit
// the canvas — drive buttons via DOM el.click() (users use keyboard shortcuts).
for (let i = 0; i < 4; i++) await page.$eval('#stereoBtn', el => el.click());
await page.evaluate(() => {
    const d = document.getElementById('depthShiftSlider');
    d.value = 120; d.dispatchEvent(new Event('input'));
    const l = document.getElementById('anaglyphLInput');
    l.value = '#ff2200'; l.dispatchEvent(new Event('input'));
});
await sleep(2400);
const s2 = await stats();
console.log('stereo fullscreen:', `mergedL/N=${(s2.mergedL / s2.numStars).toFixed(3)}`,
    `shared=${(s2.shared / Math.max(s2.mergedL, 1) * 100).toFixed(1)}%`);
expect('stereo merge healthy in fullscreen', Math.abs(s2.mergedL / s2.numStars - 1) < 0.06);
await page.$eval('#stereoBtn', el => el.click());            // back to OFF

await page.keyboard.press('KeyF');                   // exit fullscreen
await sleep(3500);
const back = await dims();
expect('exit restores preset', !back.fs && back.w === 1024 && back.h === 1024);

console.log(issues.length ? `GPU/JS ISSUES:\n${issues.join('\n')}` : 'zero GPU/JS issues');
const pass = failures === 0 && issues.length === 0;
console.log(pass ? 'FULLSCREEN TEST PASSED' : `FULLSCREEN TEST FAILED (${failures})`);
await browser.close();
process.exit(pass ? 0 : 1);
