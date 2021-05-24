import React, {useCallback, useEffect} from 'react';
import * as faceapi from 'face-api.js';
import { setIntervalAsync } from 'set-interval-async/dynamic';
import { clearIntervalAsync } from 'set-interval-async';

const { TinyFaceDetector } = faceapi;

//TODO: use redux sagas?
const base = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/';
const interval = 20; //milliseconds

const patch = () => {
    faceapi.env.monkeyPatch({
        Canvas: HTMLCanvasElement,
        Image: HTMLImageElement,
        ImageData: ImageData,
        Video: HTMLVideoElement,
        createCanvasElement: () => document.createElement('canvas'),
        createImageElement: () => document.createElement('img')
    })
};

patch();

const initialise = async network => {
    await network.loadFromUri(base);
    return network;
};

const media = async constraints => {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    const track = stream.getVideoTracks()[0];
    const capture = new ImageCapture(track);
    return [stream, capture, track];
};

const inference = async (capture, detector, canvas) => {
    const blob = await capture.takePhoto();
    const image = await faceapi.bufferToImage(blob);
    const detections = await detector.detect(image);

    faceapi.matchDimensions(canvas, image);

    const options = { boxColor: 'red', lineWidth: 1, label: 'Face' };

    for (const detection of detections) {
        const box = new faceapi.draw.DrawBox(detection.box, options);
        box.draw(canvas);
    }
};

const process = async (capture, detector, canvas) => {
    try {
        return inference(capture, detector, canvas);
    } catch (e) {
        console.log(e);
    }
}

export const App = () => {
    const load = useCallback(async () => {
        const options = new faceapi.TinyFaceDetectorOptions();
        const detector = await initialise(new TinyFaceDetector(options))
        const [stream, capture, track] = await media({video: true});

        const canvas = document.getElementById('overlay');
        const video = document.getElementById('video');

        video.srcObject = stream;

        return setIntervalAsync(async () => process(capture, detector, canvas), interval);
    })

    useEffect(() => {
        const timer = load();
        const video = document.getElementById('video');
        return function cleanup() {
            video.srcObject = null;
            timer.then(id => clearIntervalAsync(id));
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
