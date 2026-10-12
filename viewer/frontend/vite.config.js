import {defineConfig} from 'vite';
export default defineConfig({
 base:'/',
 esbuild:{jsx:'automatic'},
 server:{host:'127.0.0.1',proxy:{'/api':'http://127.0.0.1:8765'}},
 build:{target:'es2022',rollupOptions:{output:{manualChunks:{three:['three'],react:['react','react-dom']}}}}
});
