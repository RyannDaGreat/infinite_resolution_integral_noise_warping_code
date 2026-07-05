/**
 * GPU cycle-consistency test for the graveyard ("cactus test"): walk backward
 * (crowding kills stars near scene center), walk forward again (deficit opens),
 * and measure how many of the ORIGINAL identities are present afterwards.
 * Graveyard ON should recover far more identities than OFF, and the GPU
 * resurrection counter must actually move.
 *
 * Usage: node test_graveyard_cycle.mjs   (server on :8082, like test_headless)
 */
import puppeteer from 'puppeteer';
import { setTimeout as sleep } from 'timers/promises';

const PAGE_URL = 'http://localhost:8082/web_demo_v3/';
const WALK_MS = 1200;          // per leg; short enough that ghosts outlive the trip
const STATS_TICK_MS = 1100;    // stats refresh cadence is 60 frames (~1 s)

const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'],
});
const page = await browser.newPage();
const issues = [];
page.on('console', m => {
    const t = m.text();
    if ((m.type() === 'error' || m.type() === 'warning') && !t.includes('404') &&
        !t.includes('deprecated parameters')) issues.push(t.slice(0, 200));
});
page.on('pageerror', e => issues.push(String(e).slice(0, 200)));

await page.goto(PAGE_URL, { waitUntil: 'networkidle0', timeout: 30000 });
await sleep(2500);
await page.click('[data-mode="6"]');
await sleep(2 * STATS_TICK_MS);

const snap = () => page.evaluate(() => ({
    ids: window.__starIds, stats: window.__starStats,
}));

async function walkCycle(label) {
    await sleep(2 * STATS_TICK_MS);
    const before = await snap();
    await page.keyboard.down('KeyS');           // back away: crowding deaths
    await sleep(WALK_MS);
    await page.keyboard.up('KeyS');
    await sleep(STATS_TICK_MS);
    const mid = await snap();
    await page.keyboard.down('KeyW');           // walk forward: deficit births
    await sleep(WALK_MS);
    await page.keyboard.up('KeyW');
    await sleep(2 * STATS_TICK_MS);
    const after = await snap();

    const a = new Set(before.ids);
    const survivedMid = mid.ids.filter(i => a.has(i)).length;
    const survivedEnd = after.ids.filter(i => a.has(i)).length;
    const deaths = after.stats.deaths - before.stats.deaths;
    const res = after.stats.resurrections - before.stats.resurrections;
    console.log(`${label}: identities kept mid=${(survivedMid / a.size * 100).toFixed(1)}% ` +
        `end=${(survivedEnd / a.size * 100).toFixed(1)}%  ` +
        `deaths=${deaths} resurrections=${res} (${(res / Math.max(deaths, 1) * 100).toFixed(1)}%)`);
    return { end: survivedEnd / a.size, res };
}

const on = await walkCycle('graveyard ON ');
await page.click('#starGraveyardBtn');          // OFF
await sleep(STATS_TICK_MS);
const off = await walkCycle('graveyard OFF');

console.log(issues.length ? `GPU/JS ISSUES:\n${issues.join('\n')}` : 'no GPU/JS issues');
const pass = on.res > 0 && on.end > off.end && issues.length === 0;
console.log(pass ? 'GRAVEYARD CYCLE TEST PASSED'
                 : 'GRAVEYARD CYCLE TEST FAILED (resurrections must fire and ON must beat OFF)');
await browser.close();
process.exit(pass ? 0 : 1);
