import React, {useCallback, useEffect} from 'react';
import Human from '@vladmandic/human';
import * as tfd from '@tensorflow/tfjs-data';
import { setIntervalAsync } from 'set-interval-async/dynamic';
import { clearIntervalAsync } from 'set-interval-async';


const interval = 1000; //milliseconds
const run = setIntervalAsync
const stop = clearIntervalAsync
const delay = async ms => new Promise(resolve => setTimeout(resolve, ms));

const media = async constraints => {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    const track = stream.getVideoTracks()[0];
    const capture = new ImageCapture(track);
    return [stream, capture, track];
};

//TODO: make dynamic and render after mask detection has been performed
const options = {
    color: 'red',
    lineHeight: 1,
    lineWidth: 1,
    roundRect: 0,
    useRawBoxes: true
}

const bind_lower = (v, t) => [v, (v < t) ? t - v : 0];
const bind_upper = (v, t) => [v, (v > t) ? -(v - t) : 0];

const bind = (v0, v1, lower, upper) => {
    const [l, lo] = bind_lower(v0, lower);
    const [u, hi] = bind_upper(v1, upper);
    return [l + hi, u + lo];
};

const adjust = (target, value, x, y, lower, upper) => {
    const d = (target - value) / 2.0;
    const dx = Math.ceil(d);
    const dy = Math.floor(d);
    const minima = Math.max(lower, Math.round(x) + (-1 * dx));
    const maxima = Math.min(upper, Math.round(y) + dy);
    console.log('target', target, 'value', value, 'x', x, 'y', y, 'd', d, 'dx', dx, 'dy', dy, 'min', minima, 'max', maxima);
    return [minima, maxima];
};

const shift = (left, top, right, bottom, target, max_width, max_height) => {
    const goal = target - 1;
    let [l, t, r, b] = [0, 0, 0, 0];

    const w = () => Math.round((right + r) - (left + l));
    const h = () => Math.round((bottom + b) - (top + t));

    let width;
    while ((width = w()) !== goal) {
        [l, r] = adjust(target, width, l, r, 0, max_width);
        console.log('width', width, 'l', l, 'r', r);
    }

    let height;
    while ((height = h()) !== goal) {
        [t, b] = adjust(target, height, t, b, 0, max_height);
        console.log('height', height, 't', t, 'b', b);
    }

    const [le, ri] = bind(left + l, right + r, 0, max_width);
    const [to, bo] = bind(top + t, bottom + b, 0, max_height);
    return [le, to, ri, bo];
};

const predict = async (mask, input) => {
    return new Promise(async resolve => {
        const result = mask.predict(input);
        return resolve(result);
    });
}

//TODO: render / draw function
const inference = async (human, mask, input, output, ctx) => {
    const image = await input.capture();
    // const data = await image.data();
    // const [detections, predictions] = await ipcRenderer.invoke('inference', data);
    // console.log('detections', detections);
    // console.log('predictions', predictions);
    const [height, width] = image.shape;
    console.log('image', image);
    try {
        const tensor = image.expandDims(0);
        const faces = await human.detect(tensor);
        const boxes = faces.face.map(face => face.box);
        //TODO: dispose when done...
        const coordinates = boxes.map(([left, top, bottom, right]) => {
            // const width = right - left;
            // const height = bottom - top;

            // const [l, t, r, b] = shift(left, top, right, bottom, 224, width, height); //TODO:
            // const w = r - l
            // const h = b - t;

            //[left, top], [width, height]
            //width, height
            return image.slice([left, top], [224, 224]);
        });
        console.log('coordinates', coordinates);
        try {
            //TODO: blocking... make asynchronous
            //TODO: predict via IPC..?
            if (coordinates.length) {
                const batch = human.tf.stack(coordinates);
                const predictions = await predict(mask, batch);
                console.log('predictions', predictions);
            }
        } catch (e) {
            console.log(e)
        }
        console.log('faces', faces);
        ctx.clearRect(0, 0, output.width, output.height);
        await human.draw.face(output, faces.face, options);
    } finally {
        image.dispose();
    }
};

const process = async (detector, masks, input, output, ctx) => {
    try {
        return await inference(detector, masks, input, output, ctx);
    } catch (e) {
        console.log(e);
    }
}

const configuration = {
    backend: 'webgl',
    modelBasePath: '../node_modules/@vladmandic/human/models',
    face: {
        gesture: { enabled: false },
        mesh: { enabled: false },
        iris: { enabled: false },
        description: { enabled: false },
        emotion: { enabled: false }
    },
    body: { enabled: false },
    hand: { enabled: false },
    object: { enabled: false }
};

export const App = () => {
    //TODO: ability to switch detectors
    //TODO: controls / configuration
    const load = useCallback(async () => {
        const human = new Human(configuration);
        const mask = await human.tf.loadLayersModel('./model/model.json');
        const [stream, capture, track] = await media({video: true});

        console.log('human', human);
        console.log('mask', mask);

        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');

        const video = document.getElementById('video');
        video.srcObject = stream;
        const webcam = await tfd.webcam(video);

        await delay(150);

        // TODO: animation frame API...?

        // const render = async () => {
        //     await process(human, mask, video, canvas, ctx);
        //     return requestAnimationFrame(render);
        // }
        //
        // await render();

        const render = async () => await process(human, mask, webcam, canvas, ctx);
        return run(render, interval);
    }, [])

    useEffect(() => {
        const timer = load();
        const video = document.getElementById('video');
        return function cleanup() {
            video.srcObject = null;
            timer.then(id => stop(id));
        }
    }, [load]);

    return (
        <div>
            <div id='container' style={{ position: 'relative', width: '100%', height: '100%' }}>
                <canvas id='overlay' style={{ position: 'absolute', top: '0px', left: '0px', width: '100%', height: '100%', zIndex: 2 }}/>
                <video id='video' autoPlay style={{ position: 'relative', width: '100%', height: '100%', zIndex: 1 }}/>
            </div>
            <div id='controls'>
                <div>Controls</div>
            </div>
        </div>
    );
}
