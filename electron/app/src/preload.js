window.addEventListener('DOMContentLoaded', () => {
    const development = process?.env?.NODE_ENV === 'development';

    if (development) {
        process.once('loaded', () => {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = './dist/style.css';
            document.getElementsByTagName('head')[0].appendChild(link);
        });
    }

    const scripts = [];

    if (development) {
        // Dynamically insert the DLL script in development env in the
        // renderer process
        scripts.push('../.erb/dll/renderer.dev.dll.js');

        // Dynamically insert the bundled app script in the renderer process
        const port = process.env.PORT || 1212;
        scripts.push(`http://localhost:${port}/dist/renderer.dev.js`);
    } else {
        scripts.push('./dist/renderer.prod.js');
    }

    if (scripts.length) {
        for (const script of scripts) {
            const element = document.createElement('script');
            element.defer = true;
            element.src = script;
            document.getElementsByTagName('body')[0].appendChild(element);
        }
    }

    //TODO: prevent TF from detecting environment auto-magically...
    delete process?.versions?.node;
});
