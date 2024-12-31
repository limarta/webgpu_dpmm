import rawPlugin from 'vite-raw-plugin';

export default {
  plugins: [
    rawPlugin({
      fileRegex: /\.wgsl$/,
    }),
  ],
  optimizeDeps: {
    exclude: ['plotly.js'],
  }
};
