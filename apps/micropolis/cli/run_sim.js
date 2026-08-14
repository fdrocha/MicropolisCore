#!/usr/bin/env tsx

/**
 * Standalone headless simulation runner.
 *
 * Loads one of the cities packaged inside the WASM engine, ticks it forward
 * a fixed number of times, and records per-tick city stats as JSONL. Every
 * engine callback (messages, budget/date/fund updates, tool results,
 * disasters, etc.) is also logged as JSONL to a companion .log file next to
 * the stats output. Pass --verbose-log to also include the high-frequency,
 * low-signal events (simulateRobots, simulateChurch, updateMap,
 * updateHistory, updateDate), which are skipped by default.
 *
 * KNOWN ISSUE: the "deadwood" builtin city intermittently crashes the WASM
 * engine with "RuntimeError: memory access out of bounds" (a real
 * out-of-bounds memory access in the C++ engine, not a bug in this script).
 * It's nondeterministic — reruns with the same --seed and tick count
 * sometimes succeed and sometimes crash. Avoid it until the underlying
 * engine bug is fixed.
 *
 * Usage:
 *   pnpm run run-sim --city haight --ticks 200
 *   pnpm run run-sim --help   # lists all builtin city choices
 *
 * Output goes to <output-base-dir>/<city>/log-seed<seed>[-nodisasters].jsonl,
 * with the event log next to it as events-seed<seed>[-nodisasters].jsonl.
 */

import fs from 'node:fs';
import path from 'node:path';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { loadMicropolisMainModule } from '../src/lib/wasm/node.ts';
import { callbackMethodNames } from '../src/lib/wasm/callbacks.ts';
import { messageText } from '../src/lib/engineMessages.ts';

// Events that fire every tick (or nearly so) and rarely carry information
// useful for offline analysis. Skipped unless --verbose-log is passed.
const LOW_SIGNAL_EVENTS = new Set(['simulateRobots', 'simulateChurch', 'updateMap', 'updateHistory', 'updateDate']);

// updateEvaluation/updateOptions carry no args of their own — they're a
// signal that fields on the Micropolis object were just refreshed, so we
// read the relevant fields straight off the engine when they fire.
const EVALUATION_FIELDS = ['cityPop', 'cityPopDelta', 'cityAssessedValue', 'cityClass', 'cityScore', 'cityScoreDelta', 'cityYes', 'gameLevel'];
const OPTIONS_FIELDS = ['autoGoto', 'autoBudget', 'autoBulldoze', 'enableSound', 'enableDisasters', 'doAnimation', 'doMessages', 'doNotices'];

// Mirrors MicropolisSimulator.ts's cityFileNames — the cities preloaded into
// the WASM module's virtual filesystem (the only ones loadCity() can reach).
const BUILTIN_CITIES = [
	'about',
	'badnews',
	'bluebird',
	'bruce',
	'deadwood',
	'finnigan',
	'freds',
	'haight',
	'happisle',
	'joffburg',
	'kamakura',
	'kobe',
	'kowloon',
	'kyoto',
	'linecity',
	'med_isle',
	'ndulls',
	'neatmap',
	'radial',
	'scenario_bern',
	'scenario_boston',
	'scenario_detroit',
	'scenario_dullsville',
	'scenario_hamburg',
	'scenario_rio_de_janeiro',
	'scenario_san_francisco',
	'scenario_tokyo',
	'senri',
	'southpac',
	'splats',
	'wetcity',
	'yokohama'
];

const argv = yargs(hideBin(process.argv))
	.scriptName('run_sim')
	.usage('$0 --city <name> --ticks <n> --output-base-dir <dir>')
	.option('city', {
		alias: 'c',
		type: 'string',
		choices: BUILTIN_CITIES,
		default: 'haight',
		describe:
			'Builtin city to load. KNOWN ISSUE: "deadwood" intermittently crashes the WASM engine ' +
			'("RuntimeError: memory access out of bounds") — a nondeterministic out-of-bounds memory ' +
			'access in the C++ engine, not this script. Avoid it until fixed.'
	})
	.option('ticks', {
		alias: 't',
		type: 'number',
		describe: 'Number of simTick() calls to run (default: 100 if --turns is not given)'
	})
	.option('turns', {
		type: 'number',
		describe:
			'Number of in-game turns to run instead of --ticks (1 turn = 16 ticks = 1 cityTime increment — ' +
			'see simulate.cpp). Without --log-every-tick this logs exactly N rows. Mutually exclusive with --ticks.'
	})
	.conflicts('ticks', 'turns')
	.option('output-base-dir', {
		alias: 'o',
		type: 'string',
		default: './sim-runs/',
		describe:
			'Base directory to write output under. Stats go to ' +
			'<output-base-dir>/<city>/log-seed<seed>-[no]disasters.jsonl, with the event log next to it ' +
			'as events-seed<seed>-[no]disasters.jsonl.'
	})
	.option('log-every-tick', {
		type: 'boolean',
		default: false,
		describe:
			'Log every simTick(), instead of just the last tick before cityTime increments (cityTime only advances once per 16-tick phaseCycle — see simulate.cpp).'
	})
	.option('verbose-log', {
		type: 'boolean',
		default: false,
		describe:
			'Include high-frequency, low-signal events in the event log (simulateRobots, simulateChurch, updateMap, updateHistory, updateDate). Off by default.'
	})
	.option('disasters', {
		type: 'boolean',
		default: true,
		describe: 'Random disasters (fires, floods, monster, etc.). Pass --no-disasters to turn them off.'
	})
	.option('seed', {
		type: 'number',
		default: 42,
		describe:
			'Seed the RNG right after loadCity() (applied every run — a fixed default keeps output paths and results reproducible). Caveat: this makes the raw RNG draws reproducible, but the simulation itself has other unexplained non-determinism — repeated runs with the same seed still diverge slightly.'
	})
	.example('$0 --city haight --ticks 500', 'Run haight for 500 ticks')
	.example('$0 --city haight --turns 500', 'Run haight for 500 turns (8000 ticks), logging exactly 500 rows')
	.help()
	.alias('help', 'h')
	.strict()
	.parseSync();

// A turn is one cityTime increment — 16 ticks (see simulate.cpp's 16-phase
// phaseCycle) — so --turns N logs exactly N rows without --log-every-tick.
const TICKS_PER_TURN = 16;
const totalTicks = argv.turns !== undefined ? argv.turns * TICKS_PER_TURN : argv.ticks ?? 100;

// Unwraps embind enum-style values ({value: N}) down to a plain number;
// passes plain numbers/strings/booleans through untouched.
function unwrap(value) {
	if (value && typeof value === 'object' && 'value' in value) return value.value;
	return value;
}

function snapshot(micropolis, tick) {
	return {
		tick, // number of simTick() calls made so far (script-side counter, not an engine field)
		cityName: String(micropolis.cityName ?? ''), // name of the loaded city
		cityTime: micropolis.cityTime, // in-game clock, advances once per 16-tick phase cycle
		cityYear: micropolis.cityYear, // current in-game year
		cityMonth: micropolis.cityMonth, // current in-game month
		cityClass: unwrap(micropolis.cityClass), // city size classification (village/town/city/capital/metropolis/megalopolis)
		cityScore: micropolis.cityScore, // overall city evaluation score
		cityScoreDelta: micropolis.cityScoreDelta, // change in city score since last evaluation
		cityPop: micropolis.cityPop, // total city population
		cityPopDelta: micropolis.cityPopDelta, // change in population since last evaluation
		totalFunds: micropolis.totalFunds, // city treasury balance
		cashFlow: micropolis.cashFlow, // net income/expense this budget cycle
		cityTax: micropolis.cityTax, // current property tax rate (0-20)
		cityTaxAverage: micropolis.cityTaxAverage, // time-weighted average tax rate over the current budget cycle
		trafficAverage: micropolis.trafficAverage, // city-wide average traffic density
		pollutionAverage: micropolis.pollutionAverage, // city-wide average pollution level
		crimeAverage: micropolis.crimeAverage, // city-wide average crime level
		landValueAverage: micropolis.landValueAverage, // city-wide average land value
		resPop: micropolis.resPop, // residential zone population
		comPop: micropolis.comPop, // commercial zone population
		indPop: micropolis.indPop, // industrial zone population
		roadTotal: micropolis.roadTotal, // total count of road tiles
		railTotal: micropolis.railTotal, // total count of rail tiles
		policeStationPop: micropolis.policeStationPop, // number of police stations
		fireStationPop: micropolis.fireStationPop, // number of fire stations
		hospitalPop: micropolis.hospitalPop, // number of hospitals
		stadiumPop: micropolis.stadiumPop, // number of stadiums
		seaportPop: micropolis.seaportPop, // number of seaports
		airportPop: micropolis.airportPop, // number of airports
		coalPowerPop: micropolis.coalPowerPop, // number of coal power plants
		nuclearPowerPop: micropolis.nuclearPowerPop, // number of nuclear power plants
		poweredZoneCount: micropolis.poweredZoneCount, // count of zone tiles currently receiving power
		unpoweredZoneCount: micropolis.unpoweredZoneCount, // count of zone tiles currently lacking power
		externalMarket: micropolis.externalMarket, // external market demand multiplier for industry
		roadEffect: micropolis.roadEffect, // road funding effectiveness multiplier (funding level -> service quality)
		policeEffect: micropolis.policeEffect, // police funding effectiveness multiplier
		fireEffect: micropolis.fireEffect // fire funding effectiveness multiplier
	};
}

// Base fields every logged event carries, so the event log can be joined
// against the stats JSONL on tick/cityTime.
function eventBase(micropolis, tick, name) {
	return {
		tick,
		event: name,
		cityTime: micropolis.cityTime,
		cityYear: micropolis.cityYear,
		cityMonth: micropolis.cityMonth
	};
}

function fieldSnapshot(micropolis, fields) {
	const out = {};
	for (const field of fields) {
		out[field] = unwrap(micropolis[field]);
	}
	return out;
}

// Names for each callback's extra args, in order, per the C++ call sites
// (packages/micropolis-engine/src/*.cpp — see callback.h for the canonical
// parameter names). Callbacks not listed here take no extra args.
const ARG_NAMES = {
	// doAutoGoto(x, y, message) — message.cpp: tile coords + the message that triggered the goto.
	autoGoto: ['x', 'y', 'message'],
	// generateSomeCity(seed) — generate.cpp: RNG seed used to generate the map.
	didGenerateMap: ['seed'],
	// fileio.cpp: path of the city file just loaded.
	didLoadCity: ['fileName'],
	// loadScenario(s, name, fname) — fileio.cpp: scenario city name + file path loaded.
	didLoadScenario: ['scenarioName', 'fileName'],
	// fileio.cpp: path of the city file just saved.
	didSaveCity: ['fileName'],
	// tool.cpp: tool name + tile coords where it was applied.
	didTool: ['toolName', 'x', 'y'],
	// fileio.cpp: path of the city file that failed to load.
	didntLoadCity: ['fileName'],
	// fileio.cpp: path of the city file that failed to save.
	didntSaveCity: ['fileName'],
	// micropolis.cpp: sound channel ("city"/"interface"), sound effect name, tile coords (-1,-1 = non-positional).
	makeSound: ['channel', 'sound', 'x', 'y'],
	// fileio.cpp: path the city is being saved as.
	saveCityAs: ['fileName'],
	// sendMessage(messageNum, x, y, pictureFlag, important) — text.h has the message enum;
	// messageText is derived from messageNum, not a positional arg (see namedArgs below).
	sendMessage: ['messageNum', 'x', 'y', 'pictureFlag', 'important'],
	// tool.cpp: zone status popup — string-table indices per category, plus tile coords.
	showZoneStatus: [
		'tileCategoryIndex',
		'populationDensityIndex',
		'landValueIndex',
		'crimeRateIndex',
		'pollutionIndex',
		'growthRateIndex',
		'x',
		'y'
	],
	// zone.cpp: church zone tile coords + which church tile/sprite variant (0-7).
	simulateChurch: ['x', 'y', 'churchNumber'],
	// micropolis.cpp doEarthquake(strength): earthquake magnitude.
	startEarthquake: ['strength'],
	// micropolis.cpp doStartScenario(scenario): Scenario enum id being started.
	startScenario: ['scenario'],
	// utilities.cpp: city name.
	updateCityName: ['cityName'],
	// updateDemand(r, c, i) — update.cpp setDemand: residential/commercial/industrial demand valve values.
	updateDemand: ['residentialDemand', 'commercialDemand', 'industrialDemand'],
	// update.cpp: current totalFunds.
	updateFunds: ['totalFunds'],
	// utilities.cpp: GameLevel enum value.
	updateGameLevel: ['gameLevel'],
	// utilities.cpp setPasses: simPasses, sub-passes run per tick.
	updatePasses: ['passes'],
	// utilities.cpp pause()/resume(): simPaused.
	updatePaused: ['paused'],
	// utilities.cpp setSpeed: simSpeed level (0-3).
	updateSpeed: ['speed'],
	// updateTaxRate(cityTax): current tax rate.
	updateTaxRate: ['taxRate'],
	// update.cpp: cityYear/cityMonth as of this date update (redundant with the
	// base event fields, named for clarity when --verbose-log is on).
	updateDate: ['cityYear', 'cityMonth']
};

function namedArgs(name, args) {
	const names = ARG_NAMES[name];
	const out = {};
	args.forEach((value, i) => {
		const key = names?.[i] ?? `arg${i}`;
		out[key] = unwrap(value);
	});
	if (name === 'sendMessage') {
		out.messageText = messageText(out.messageNum);
	}
	return out;
}

function makeLoggingCallback(engine, micropolis, getTick, logStream, verbose) {
	const handlers = {};
	for (const name of callbackMethodNames) {
		if (!verbose && LOW_SIGNAL_EVENTS.has(name)) {
			handlers[name] = () => { };
			continue;
		}
		handlers[name] = (...allArgs) => {
			// Every callback's first two args are the Micropolis* and an opaque
			// user-data handle — not useful to log, so they're dropped here.
			const args = allArgs.slice(2);
			const entry = eventBase(micropolis, getTick(), name);
			if (name === 'updateEvaluation') {
				Object.assign(entry, fieldSnapshot(micropolis, EVALUATION_FIELDS));
			} else if (name === 'updateOptions') {
				Object.assign(entry, fieldSnapshot(micropolis, OPTIONS_FIELDS));
			} else {
				Object.assign(entry, namedArgs(name, args));
			}
			logStream.write(JSON.stringify(entry) + '\n');
		};
	}
	return new engine.JSCallback(handlers);
}

// Builds the stats and event log paths for this run:
//   <outputBaseDir>/<city>/log-seed<seed>[-no]disasters.jsonl
//   <outputBaseDir>/<city>/events-seed<seed>[-no]disasters.jsonl
function outputPathsFor(outputBaseDir, city, seed, disastersEnabled) {
	const suffix = disastersEnabled ? '-disasters' : '-nodisasters';
	const dir = path.join(outputBaseDir, city);
	return {
		dir,
		statsPath: path.join(dir, `log-seed${seed}${suffix}.jsonl`),
		logPath: path.join(dir, `events-seed${seed}${suffix}.jsonl`)
	};
}

async function main() {
	const cityPath = `/cities/${argv.city}.cty`;
	const engine = await loadMicropolisMainModule();
	const micropolis = new engine.Micropolis();

	const { dir, statsPath, logPath } = outputPathsFor(argv.outputBaseDir, argv.city, argv.seed, argv.disasters);
	fs.mkdirSync(dir, { recursive: true });

	const logStream = fs.createWriteStream(logPath, { flags: 'w' });
	const tickRef = { current: 0 };
	const callback = makeLoggingCallback(engine, micropolis, () => tickRef.current, logStream, argv.verboseLog);
	micropolis.setCallback(callback, {});
	micropolis.init();

	// init() unconditionally sets enableDisasters = true (micropolis.cpp), so
	// this must run after it; loadCity() doesn't touch the flag, so it's safe
	// to set before or after that call.
	if (!argv.disasters) {
		micropolis.enableDisasters = false;
	}

	const loaded = micropolis.loadCity(cityPath);
	if (!loaded) {
		console.error(`Failed to load builtin city "${argv.city}" (${cityPath})`);
		micropolis.delete();
		process.exitCode = 1;
		return;
	}

	// loadCity() reseeds the RNG from the clock internally, so --seed only
	// takes effect if applied after it — setting it before would be
	// silently overwritten.
	micropolis.seedRandom(argv.seed);

	console.log(`Loaded ${argv.city} — running ${totalTicks} ticks, writing stats to ${statsPath} and events to ${logPath}`);
	if (!argv.logEveryTick) {
		console.log('Logging one row per cityTime value (pass --log-every-tick to log every tick instead).');
	}

	// phaseCycle wraps 0..15 (simulate.cpp). Phase 0 increments cityTime and
	// runs cityEvaluation() (cityPop, cityScore, cityClass) on the *first*
	// cycle after load; phases 1..14 rebuild the census/map scans. Logging at
	// phaseCycle === 0 catches the tick before any of that ran for this
	// cityTime value (cityPop still 0). phaseCycle === 15 is the last phase
	// of the cycle, after every scan for the current cityTime has completed.
	const shouldLog = (m) => argv.logEveryTick || m.phaseCycle === 15;

	const out = fs.createWriteStream(statsPath, { flags: 'w' });
	let rowsWritten = 0;
	if (shouldLog(micropolis)) {
		out.write(JSON.stringify(snapshot(micropolis, 0)) + '\n');
		rowsWritten++;
	}

	for (let tick = 1; tick <= totalTicks; tick++) {
		tickRef.current = tick;
		micropolis.simTick();
		if (shouldLog(micropolis)) {
			out.write(JSON.stringify(snapshot(micropolis, tick)) + '\n');
			rowsWritten++;
		}
	}

	await Promise.all([
		new Promise((resolve, reject) => out.end((err) => (err ? reject(err) : resolve()))),
		new Promise((resolve, reject) => logStream.end((err) => (err ? reject(err) : resolve())))
	]);

	micropolis.delete();
	console.log(`Done. Wrote ${rowsWritten} rows to ${statsPath}`);
}

await main();
