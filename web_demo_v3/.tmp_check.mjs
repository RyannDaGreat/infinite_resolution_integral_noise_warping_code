import puppeteer from 'puppeteer';
import { setTimeout as sleep } from 'timers/promises';
const browser = await puppeteer.launch({ headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'] });
const page = await browser.newPage();
const issues = [];
page.on('console', m => {
    const t = m.text();
    if ((m.type() === 'error' || m.type() === 'warning') && !t.includes('404') && !t.includes('deprecated parameters')) issues.push(t.slice(0, 200));
});
page.on('pageerror', e => issues.push(String(e).slice(0, 200)));
await page.goto('http://localhost:8082/web_demo_v3/', { waitUntil: 'networkidle0', timeout: 30000 });
await sleep(2500);
await page.click('[data-mode="6"]');
await sleep(1500);
const SHOTS = '/private/tmp/claude-501/-Users-ryan-CleanCode-Sandbox-RP-Dumps-StarWarp/dea3fb35-3541-4d85-89a0-1a263f22b76a/scratchpad';
await page.click('#starColorQBtn'); await sleep(1500);
await page.screenshot({ path: `${SHOTS}/qcolor.png` });
await page.click('#starSizeQBtn'); await sleep(1500);
await page.screenshot({ path: `${SHOTS}/qcolor_qsize.png` });
await page.click('#starColorQBtn');                       // q-color off, size on
await page.click('#starEmojiBtn'); await sleep(1500);     // emoji + q-size
await page.screenshot({ path: `${SHOTS}/emoji_qsize.png` });
console.log('stats:', JSON.stringify(await page.evaluate(() => window.__starStats)));
console.log(issues.length ? `ISSUES:\n${issues.join('\n')}` : 'zero GPU/JS issues');
await browser.close();
process.exit(issues.length ? 1 : 0);
