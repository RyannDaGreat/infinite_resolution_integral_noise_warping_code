/**
 * Performance test suite for V2 WebGPU demo.
 * Collects N frames of GPU timestamps, computes mean±σ per phase.
 * Runs on REAL GPU (not SwiftShader) for accurate hardware measurements.
 *
 * Usage: node perf_test.mjs [--frames=200] [--warmup=50] [--swiftshader]
 */
import puppeteer from 'puppeteer';

const args = process.argv.slice(2);
const getArg = (name, def) => {
    const m = args.find(a => a.startsWith(`--${name}=`));
    return m ? m.split('=')[1] : def;
};

const NUM_FRAMES = parseInt(getArg('frames', '200'));
const WARMUP = parseInt(getArg('warmup', '50'));
const USE_SWIFTSHADER = args.includes('--swiftshader');
const URL = `http://localhost:8082/web_demo_v2/`;

function stats(arr) {
    const n = arr.length;
    if (n === 0) return { mean: 0, std: 0, n: 0 };
    const mean = arr.reduce((a, b) => a + b, 0) / n;
    const std = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
    return { mean, std, n };
}

async function main() {
    const chromeArgs = [
        '--enable-unsafe-webgpu',
        '--no-sandbox',
        '--disable-gpu-sandbox',
    ];
    if (USE_SWIFTSHADER) {
        chromeArgs.push('--use-webgpu-adapter=swiftshader');
        chromeArgs.push('--enable-features=Vulkan');
    }

    console.log(`Config: ${NUM_FRAMES} frames, ${WARMUP} warmup, GPU=${USE_SWIFTSHADER ? 'SwiftShader' : 'HARDWARE'}`);

    const browser = await puppeteer.launch({
        headless: true,
        args: chromeArgs,
    });

    const page = await browser.newPage();

    const errors = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto(URL, { waitUntil: 'domcontentloaded' });

    // Wait for warmup + measurement frames
    const totalWait = Math.ceil((WARMUP + NUM_FRAMES) / 60) * 1000 + 3000;
    console.log(`Waiting ${totalWait}ms for ${WARMUP + NUM_FRAMES} frames...`);
    await new Promise(r => setTimeout(r, totalWait));

    // Collect timing data
    const data = await page.evaluate((warmup, numFrames) => {
        const history = window.__gpuTimingHistory || [];
        const noiseStats = window.__noiseStats || { mean: NaN, std: NaN };
        const uiStats = window.__stats || {};

        // Skip warmup frames, take numFrames
        const measured = history.slice(warmup, warmup + numFrames);

        return { measured, noiseStats, fps: uiStats.fps, totalSamples: history.length };
    }, WARMUP, NUM_FRAMES);

    // Check for errors
    if (errors.length > 0) {
        console.error('Page errors:', errors);
    }

    const { measured, noiseStats, totalSamples } = data;
    console.log(`\nCollected ${measured.length} timing samples (total available: ${totalSamples})`);

    if (measured.length === 0) {
        console.error('ERROR: No GPU timing data collected. timestamp-query may not be available on this GPU.');
        console.log('FPS:', data.fps);
        console.log('Noise stats:', noiseStats);

        // Fallback: report CPU-only timing
        const cpuData = await page.evaluate(() => window.__stats?.cpuFrameMs || []);
        if (cpuData.length > 0) {
            const cpuStats = stats(cpuData);
            console.log(`CPU frame time: ${cpuStats.mean.toFixed(3)}±${cpuStats.std.toFixed(3)} ms (n=${cpuStats.n})`);
        }

        await browser.close();
        process.exit(0);
    }

    // Compute per-phase statistics
    const phases = Object.keys(measured[0]);
    console.log('\n=== GPU TIMING RESULTS (ms) ===');
    console.log(`${'Phase'.padEnd(14)} ${'Mean'.padStart(8)} ${'± σ'.padStart(8)} ${'Min'.padStart(8)} ${'Max'.padStart(8)} n`);
    console.log('-'.repeat(56));

    const results = {};
    for (const phase of phases) {
        const vals = measured.map(t => t[phase]);
        const s = stats(vals);
        const min = Math.min(...vals);
        const max = Math.max(...vals);
        results[phase] = s;
        console.log(
            `${phase.padEnd(14)} ${s.mean.toFixed(3).padStart(8)} ${('±' + s.std.toFixed(3)).padStart(8)} ${min.toFixed(3).padStart(8)} ${max.toFixed(3).padStart(8)} ${s.n}`
        );
    }

    console.log('\n=== NOISE STATS ===');
    console.log(`mean: ${noiseStats.mean.toFixed(6)}  std: ${noiseStats.std.toFixed(6)}`);
    const meanOK = Math.abs(noiseStats.mean) < 0.15;
    const stdOK = Math.abs(noiseStats.std - 1.0) < 0.3;
    console.log(`Validation: mean ${meanOK ? 'PASS' : 'FAIL'}  std ${stdOK ? 'PASS' : 'FAIL'}`);

    // Generate commit-ready summary line
    const total = results.total;
    const brown = results.brownian;
    console.log('\n=== COMMIT SUMMARY ===');
    console.log(`V2 GPU total: ${total.mean.toFixed(2)}±${total.std.toFixed(2)}ms (n=${total.n}), brownian: ${brown.mean.toFixed(2)}±${brown.std.toFixed(2)}ms, noise: mean=${noiseStats.mean.toFixed(3)} std=${noiseStats.std.toFixed(3)}`);

    // Screenshot
    await page.screenshot({ path: 'web_demo_v2/perf_screenshot.png' });

    await browser.close();

    if (!meanOK || !stdOK) {
        console.error('\nFAILED: Noise stats out of range');
        process.exit(1);
    }
    console.log('\nPASS');
}

main().catch(e => { console.error(e); process.exit(1); });
