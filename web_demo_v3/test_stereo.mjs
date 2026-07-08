/**
 * Headless stereo-stars test. Validates, for each stereo mode (SBS / blend /
 * red-blue): zero GPU validation issues, and the merged-view survivor counts —
 * the uniformity invariant of the render-time merge predicts E[merged] = N per
 * eye (report §13) — stay within tolerance of N, static AND under motion.
 *
 * Usage: node test_stereo.mjs   (server on :8082, like test_headless.mjs)
 */
import puppeteer from 'puppeteer';
import { setTimeout as sleep } from 'timers/promises';

const PAGE_URL = 'http://localhost:8082/web_demo_v3/';
const STATS_TICK_MS = 1100;
const MERGE_TOL = 0.06;        // |mergedCount/N - 1| bound (deterministic thinning + MC)
const SHOT_DIR = process.env.STEREO_SHOT_DIR || null;

const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 1300 });   // show the WHOLE canvas in screenshots
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
await sleep(STATS_TICK_MS);

const stats = () => page.evaluate(() => window.__starStats);
let failures = 0;
const check = (label, s) => {
    const n = s.numStars;
    const okL = Math.abs(s.mergedL / n - 1) < MERGE_TOL;
    const okR = Math.abs(s.mergedR / n - 1) < MERGE_TOL;
    const okStream = s.inBoundsFrac === 1 && s.minOverMean > 0.7 && s.maxOverMean < 1.3;
    // shared-identity coupling: both eyes should agree on well over half the
    // stars (Python reference: 93.7% at its settings; loose bound for 3D play)
    const sharedFrac = s.shared / Math.max(s.mergedL, 1);
    const okShared = sharedFrac > 0.5;
    if (!(okL && okR && okStream && okShared)) failures++;
    console.log(`${label}: mergedL/N=${(s.mergedL / n).toFixed(3)} mergedR/N=${(s.mergedR / n).toFixed(3)}`
        + ` shared=${(sharedFrac * 100).toFixed(1)}%`
        + ` streamL ${s.minOverMean.toFixed(2)}/${s.maxOverMean.toFixed(2)}`
        + `  ${okL && okR && okStream && okShared ? 'PASS' : 'FAIL'}`);
};

for (const [mode, name] of [[1, 'sbs-crossed'], [2, 'sbs-parallel'], [3, 'blend'], [4, 'red-blue']]) {
    await page.click('#stereoBtn');                 // cycles OFF->SBS->blend->red-blue
    await sleep(2 * STATS_TICK_MS);
    check(`stereo ${name} (static)`, await stats());
    if (SHOT_DIR) await page.screenshot({ path: `${SHOT_DIR}/stereo_${name}.png` });
}

// motion: strafe + walk while in red-blue, counts must hold
await page.keyboard.down('KeyW');
await page.keyboard.down('KeyD');
await sleep(1500);
await page.keyboard.up('KeyW');
await page.keyboard.up('KeyD');
await sleep(2 * STATS_TICK_MS);
check('stereo red-blue (after motion)', await stats());

// IPD extremes: zero baseline (eyes coincide) and max
for (const ipd of [0, 0.5]) {
    await page.evaluate((v) => {
        const sl = document.getElementById('stereoIpdSlider');
        sl.value = v; sl.dispatchEvent(new Event('input'));
    }, ipd);
    await sleep(2 * STATS_TICK_MS);
    check(`stereo red-blue IPD=${ipd}`, await stats());
}

// Cull Orphans toggle: survivor counts are pre-filter, so stats must hold,
// and the render path must stay clean while orphans are hidden.
await page.click('#cullOrphansBtn');
await sleep(2 * STATS_TICK_MS);
check('red-blue + Cull Orphans', await stats());
if (SHOT_DIR) await page.screenshot({ path: `${SHOT_DIR}/stereo_cull_orphans.png` });
await page.click('#cullOrphansBtn');

// back to OFF: mono must still be healthy
await page.click('#stereoBtn');
await sleep(2 * STATS_TICK_MS);
const s = await stats();
const monoOk = s.inBoundsFrac === 1 && s.minOverMean > 0.7 && s.maxOverMean < 1.3;
if (!monoOk) failures++;
console.log(`mono after stereo: uniformity ${s.minOverMean.toFixed(2)}/${s.maxOverMean.toFixed(2)}`
    + `  ${monoOk ? 'PASS' : 'FAIL'}`);

console.log(issues.length ? `GPU/JS ISSUES:\n${issues.join('\n')}` : 'zero GPU/JS issues');
const pass = failures === 0 && issues.length === 0;
console.log(pass ? 'STEREO TEST PASSED' : `STEREO TEST FAILED (${failures} check failures)`);
await browser.close();
process.exit(pass ? 0 : 1);
