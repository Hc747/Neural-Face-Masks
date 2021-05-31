/* eslint global-require: off, no-console: off */

/**
 * This module executes inside of electron's main process. You can start
 * electron renderer process from here and communicate with the other processes
 * through IPC.
 *
 * When running `yarn build` or `yarn build:main`, this file is compiled to
 * `./src/main.prod.js` using webpack. This gives us some performance wins.
 */
import 'core-js/stable';
import 'regenerator-runtime/runtime';
import path from 'path';
import { app, BrowserWindow, ipcMain } from 'electron';

const production: boolean = process.env.NODE_ENV === 'production';
const development: boolean = process.env.NODE_ENV === 'development';
const debug: boolean = development || process.env.DEBUG_PROD === 'true';

const error = (e: any) => console.log(e);

const event_init: 'activate' = 'activate';
const event_stop: 'window-all-closed' = 'window-all-closed';

if (production) {
  const sourceMapSupport = require('source-map-support');
  sourceMapSupport.install();
}

if (debug) {
  require('electron-debug')();
}

const install = async () => {
  const installer = require('electron-devtools-installer');
  const forceDownload = !!process.env.UPGRADE_EXTENSIONS;
  const extensions = ['REACT_DEVELOPER_TOOLS'];

  return installer
    .default(extensions.map((name) => installer[name]), forceDownload)
    .catch(error);
};

const start = async () => {
  if (debug) {
    await install();
  }

  const RESOURCES_PATH = app.isPackaged
    ? path.join(process.resourcesPath, 'assets')
    : path.join(__dirname, '../assets');

  const getAssetPath = (...paths: string[]): string => {
    return path.join(RESOURCES_PATH, ...paths);
  };

  const browser = new BrowserWindow({
    show: false,
    icon: getAssetPath('icon.png'),
    webPreferences: {
        nodeIntegration: true,
        preload: path.join(__dirname, 'preload.js')
    },
  });

  await browser.loadURL(`file://${__dirname}/index.html`);

  browser.maximize();
  browser.show();
};

const activate = async () => {
  if (BrowserWindow.getAllWindows().length <= 0) {
    await start();
  }
};

const stop = () => app.quit();

/**
 * Add event listeners...
 */
app.whenReady().then(start).catch(error);
app.on(event_init, activate);
app.on(event_stop, stop);

// const configuration = {
//     // backend: 'webgl',
//     backend: 'humangl',
//     modelBasePath: '../node_modules/@vladmandic/human/models',
//     face: {
//         gesture: { enabled: false },
//         mesh: { enabled: false },
//         iris: { enabled: false },
//         description: { enabled: false },
//         emotion: { enabled: false }
//     },
//     body: { enabled: false },
//     hand: { enabled: false },
//     object: { enabled: false }
// };
//
// const human = new Human(configuration);
//
// ipcMain.handle('inference', async (event, image) => {
//     console.log('image', image);
//     const tensor = human.tf.tensor(image);
//     const detections = await human.detect(tensor.expandDims(0));
//     // const tensor = human.tf.tensor
//     return [detections, 456];
// });
